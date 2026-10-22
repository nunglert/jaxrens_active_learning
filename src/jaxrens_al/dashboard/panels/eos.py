"""Project panel: multi-phase equation of state (ported from EOSEnthalpy).

For each *phase* (a reference volume-scan extxyz) the panel fits a
Birch-Murnaghan EOS to both the DFT energies (``REF_energy``) and the model
energies (evaluated through the jaxrens backend), then draws three sub-panels:

  1. ``E(V)``        -- energy vs volume, DFT (o) and model (x) per phase;
  2. ``H(P)``        -- enthalpy vs pressure (DFT solid, model dotted);
  3. ``H - H_ref(P)``-- enthalpy relative to the first phase (phase stability).

Heavy work (model evaluation + curve fits) runs once in ``compute_data`` and
alviz caches the result as JSON. The model call is behind the ``_evaluate``
seam so unit tests need neither jaxrens nor a real model.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Panel spec.
# ---------------------------------------------------------------------------
from alviz import Entry, JsonDict, PanelSpec, register_panel
from pydantic import model_validator

from jaxrens_al.dashboard.panels.model_eval import (
    ModelEvalConsumer,
    ModelEvalNeed,
    model_eval_keys,
)
from jaxrens_al.dashboard.panels.reference import (
    ReferenceConsumer,
    ReferenceNeed,
)

# 1 eV / Angstrom^3 expressed in GPa.
_EVA3_TO_GPA = 160.21766208

# Pressure grid (GPa) the per-phase enthalpies are interpolated onto, so all
# phases share an x-axis for the H(P) / relative-H panels.
_P_GRID = np.linspace(-2.0, 25.0, 100)


def _expand_reference_paths(references):
    """Normalise EOS reference input to a flat list of per-phase scan files.

    Accepts a single path or a list. Any entry that is a **directory** is
    expanded to its sorted ``eos_*.extxyz`` files (falling back to ``*.extxyz``
    if none match that convention), so a user can point at *the directory the
    reference scans live in* instead of listing each phase file. Plain-file
    entries are kept as given. Returns ``None`` when ``references`` is None.
    """
    from pathlib import Path

    if references is None:
        return None
    if isinstance(references, (str, Path)):
        references = [references]
    out: list = []
    for entry in references:
        p = Path(entry)
        if p.is_dir():
            files = sorted(p.glob("eos_*.extxyz")) or sorted(
                p.glob("*.extxyz")
            )
            if not files:
                raise ValueError(
                    f"EOS reference directory {p} contains no *.extxyz scan "
                    "files."
                )
            out.extend(str(f) for f in files)
        else:
            out.append(str(p))
    return out


def _birch_murnaghan(V, E0, V0, B0, B0p):
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1) ** 3 * B0p + (eta - 1) ** 2 * (6 - 4 * eta)
    )


def _bm_pressure(V, B0, V0, B0p):
    eta = (V0 / V) ** (2.0 / 3.0)
    return (
        (3.0 * B0 / 2.0)
        * ((eta**7 / 3.0) - (eta**5 / 3.0))
        * (1 + (3.0 / 4.0) * (B0p - 4) * (eta - 1))
    )


def _fit_enthalpy(volumes, energies):
    """Fit Birch-Murnaghan and return H(P) interpolated onto ``_P_GRID``.

    Volumes/energies are per atom; pressures in GPa, enthalpies in eV/atom.
    """
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit

    volumes = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    popt, _ = curve_fit(
        _birch_murnaghan,
        volumes,
        energies,
        p0=[energies.mean(), volumes.mean(), 0.5, 4.0],
        maxfev=10000,
    )
    _, V0, B0, B0p = popt
    V = np.linspace(volumes.min(), volumes.max(), 200)
    E = _birch_murnaghan(V, *popt)
    p_eva3 = _bm_pressure(V, B0, V0, B0p)
    H = E + p_eva3 * V
    p_gpa = p_eva3 * _EVA3_TO_GPA
    order = np.argsort(p_gpa)
    return interp1d(
        p_gpa[order], H[order], bounds_error=False, fill_value="extrapolate"
    )(_P_GRID)


class EquationOfStatePanel(ReferenceConsumer, ModelEvalConsumer, Entry):
    """Multi-phase EOS: E(V), H(P) and relative enthalpy under the model + DFT.

    Two ways to supply the per-phase volume scans (give exactly one):

      * ``references`` -- explicit, already-DFT-labelled scan files; or
      * ``element`` -- declare a :class:`~jaxrens_al.dashboard.panels.reference.
        ReferenceNeed` the run-side ``resolve_references`` fetches (Materials
        Project) + DFT-labels for you, filling the files via
        :meth:`receive_references` before rendering.

    The **model** energy curve is computed either locally (loading the backend
    in-process, the default for CPU backends) or -- when ``remote_eval`` -- on a
    GPU cluster: the panel declares a
    :class:`~jaxrens_al.dashboard.panels.model_eval.ModelEvalNeed` and reads the model
    energy back from the annotated reference files. A MACE-jax bundle needs the
    latter (its ``cuequivariance`` build is GPU-only).

    Args:
        model_file: This iteration's trained-model bundle (the model curve).
        references: One path or a list of per-phase volume-scan extxyz
            (``REF_energy`` on each frame is the DFT energy), **or a dir**
            holding them -- a directory is expanded to its sorted
            ``eos_*.extxyz`` files.
        element: Chemical symbol -> declare an EOS reference need instead.
        vol_scalings, target_cutoff, level: parameters of the declared need
            (the volume grid, the supercell cutoff, and an opaque DFT-level tag
            used only to key the build for reuse).
        labels: Optional phase labels (default: the reference file stems).
        backend_type: jaxrens backend to load the model with.
        remote_eval: Evaluate the model on a GPU cluster (via the ``model_eval``
            processor) instead of in-process. Required for a MACE-jax bundle.
        max_neighbors: Static neighbor-buffer size for the backend call (the
            starting bucket when ``max_neighbors_list`` is set).
        max_neighbors_list: Neighbour-bucket ladder (e.g. ``[20, 40, 60, 90,
            120]``). Set it and the buffer is sized per batch from each frame's
            geometry and grown on overflow, instead of pinned at
            ``max_neighbors`` -- the compressed end of a volume scan packs far
            more neighbours than the expanded end. ``None`` keeps it static.
        max_neighbors_offset: Headroom above the observed neighbour count.
        batch_size: Frames per vmapped backend call (``None`` = one per call).
    """

    name = "eos"
    colspan = 3

    def __init__(
        self,
        *args,
        model_file,
        references=None,
        element: str | None = None,
        vol_scalings=None,
        target_cutoff: float = 4.0,
        level: str = "dft",
        labels=None,
        backend_type: str = "neuralil",
        remote_eval: bool = False,
        max_neighbors: int = 50,
        max_neighbors_list=None,
        max_neighbors_offset: int = 4,
        batch_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from pathlib import Path

        if (references is None) == (element is None):
            raise ValueError(
                "EOS panel needs exactly one of references= (explicit "
                "per-phase scans) or element= (declare a reference need)."
            )
        self._references = _expand_reference_paths(references)
        self._model_file = str(model_file)
        self._labels_override = list(labels) if labels is not None else None
        self._backend_type = backend_type
        self._remote_eval = bool(remote_eval)
        self._model_eval_annotated: dict = {}
        self._max_neighbors = max_neighbors
        self._max_neighbors_list = (
            sorted(int(b) for b in max_neighbors_list)
            if max_neighbors_list
            else None
        )
        self._max_neighbors_offset = int(max_neighbors_offset)
        self._batch_size = None if batch_size is None else int(batch_size)
        # Need-mode inputs: the resolver fetches + DFT-labels these and fills
        # ``_references`` via ``receive_references``.
        self._element = element
        self._vol_scalings = (
            [float(v) for v in vol_scalings]
            if vol_scalings is not None
            else None
        )
        self._target_cutoff = float(target_cutoff)
        self._level = level

    def _need(self) -> ReferenceNeed:
        params = {
            "element": self._element,
            "target_cutoff": self._target_cutoff,
            "level": self._level,
        }
        if self._vol_scalings is not None:
            params["vol_scalings"] = self._vol_scalings
        return ReferenceNeed.make("eos", params)

    def reference_needs(self) -> list:
        if self._element is None or self._references is not None:
            return []
        return [self._need()]

    def receive_references(self, mapping: dict) -> None:
        need = self._need()
        if (
            self._element is not None
            and self._references is None
            and need in mapping  # skipped (needs a DFT evaluator) -> unbuilt
        ):
            self._references = [str(p) for p in mapping[need]]

    def model_eval_needs(self) -> list:
        """Score the reference scans with this iteration's model, on cluster."""
        if not self._remote_eval or self._references is None:
            return []
        return [
            ModelEvalNeed.make(
                self._refs(),
                self._model_file,
                self._backend_type,
                want_forces=False,
                max_neighbors=self._max_neighbors,
                min_image_cutoff=self._target_cutoff,
                **self._bucket_kwargs(),
            )
        ]

    def receive_model_eval(self, mapping: dict) -> None:
        if self._remote_eval and self._references is not None:
            by_model = mapping.get(self._model_file, {})
            self._model_eval_annotated = {
                r: by_model[r] for r in self._refs() if r in by_model
            }

    def _refs(self) -> list:
        if self._references is None:
            raise RuntimeError(
                "EOS panel has no references yet: pass references=, or run "
                "resolve_references() over the dashboard (element= declares "
                "the need)."
            )
        return self._references

    def _evaluate(self, model_file, atoms_list) -> list:
        """Per-input-cell energies (eV) of each structure under the model.

        With ``target_cutoff`` set, each frame is scored on a cutoff-hosting
        supercell and the energy is scaled back to the input cell's atom count,
        so the caller's per-atom ``/n`` (n = input-cell atoms) stays correct and
        a unit-cell DFT reference remains comparable.
        """
        from jaxrens_al.dashboard.panels.model_eval import score_atoms

        energies, _ = score_atoms(
            model_file,
            self._backend_type,
            atoms_list,
            want_forces=False,
            max_neighbors=self._max_neighbors,
            min_image_cutoff=self._target_cutoff,
            **self._bucket_kwargs(),
        )
        return energies

    def _bucket_kwargs(self) -> dict:
        """The neighbour-bucketing knobs, shared by local + remote scoring."""
        return dict(
            batch_size=self._batch_size,
            max_neighbors_list=self._max_neighbors_list,
            max_neighbors_offset=self._max_neighbors_offset,
        )

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 3, wspace=0.35)
        self.ax_ev = fig.add_subplot(self.gs[0, 0])
        self.ax_hp = fig.add_subplot(self.gs[0, 1])
        self.ax_dh = fig.add_subplot(self.gs[0, 2])

    def _phase_data(self, ref_file) -> dict:
        import ase.io

        if self._remote_eval:
            # Model energies were written onto the reference frames by the
            # cluster job; read both curves from the annotated file.
            annotated = self._model_eval_annotated.get(str(ref_file))
            if annotated is None:
                raise RuntimeError(
                    "EOS panel is remote_eval but the reference scans are not "
                    "annotated yet: run resolve_model_evaluations() over the "
                    "dashboard before rendering."
                )
            energy_key, _ = model_eval_keys(self._model_file)
            frames = ase.io.read(str(annotated), index=":")
            n = len(frames[0])
            e_nnff = [float(at.info[energy_key]) / n for at in frames]
        else:
            frames = ase.io.read(ref_file, index=":")
            n = len(frames[0])
            e_nnff = [e / n for e in self._evaluate(self._model_file, frames)]
        volumes = [at.get_volume() / n for at in frames]
        e_dft = [float(at.info["REF_energy"]) / n for at in frames]
        return dict(
            volumes=volumes,
            e_dft=e_dft,
            e_nnff=e_nnff,
            h_dft=_fit_enthalpy(volumes, e_dft).tolist(),
            h_nnff=_fit_enthalpy(volumes, e_nnff).tolist(),
        )

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        from pathlib import Path

        refs = self._refs()
        labels = self._labels_override or [Path(p).stem for p in refs]
        phases = [self._phase_data(f) for f in refs]
        return JsonDict(
            labels=labels,
            p_grid=_P_GRID.tolist(),
            phases=phases,
        )

    def plot_data(self):
        import matplotlib.pyplot as plt

        phases = self.data["phases"]
        labels = self.data["labels"]
        p_grid = self.data["p_grid"]
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(phases))]

        # (1) E(V): DFT (o) + model (x) per phase.
        for i, ph in enumerate(phases):
            order = np.argsort(ph["volumes"])
            v = np.asarray(ph["volumes"])[order]
            self.ax_ev.plot(
                v,
                np.asarray(ph["e_dft"])[order],
                "o-",
                color=colors[i],
                markersize=4,
                label=labels[i],
            )
            self.ax_ev.plot(
                v,
                np.asarray(ph["e_nnff"])[order],
                "x--",
                color=colors[i],
                markersize=5,
            )
        self.ax_ev.set_xlabel(r"$V$ [$\AA^3$/atom]")
        self.ax_ev.set_ylabel(r"$E$ [eV/atom]")
        self.ax_ev.set_title("E(V)")
        self.ax_ev.legend(fontsize="x-small", loc="best")

        # (2) H(P): DFT solid, model dotted.
        for i, ph in enumerate(phases):
            self.ax_hp.plot(p_grid, ph["h_dft"], "-", color=colors[i])
            self.ax_hp.plot(p_grid, ph["h_nnff"], ":", color=colors[i])
        self.ax_hp.set_xlabel(r"$P$ [GPa]")
        self.ax_hp.set_ylabel(r"$H$ [eV/atom]")
        self.ax_hp.set_title("H(P)")

        # (3) Relative enthalpy vs the first phase's DFT curve.
        ref = np.asarray(phases[0]["h_dft"]) if phases else 0.0
        for i, ph in enumerate(phases):
            self.ax_dh.plot(
                p_grid, np.asarray(ph["h_dft"]) - ref, "-", color=colors[i]
            )
            self.ax_dh.plot(
                p_grid, np.asarray(ph["h_nnff"]) - ref, ":", color=colors[i]
            )
        self.ax_dh.set_xlabel(r"$P$ [GPa]")
        self.ax_dh.set_ylabel(r"$H - H_{\mathrm{ref}}$ [eV/atom]")
        self.ax_dh.set_title(r"$\Delta H(P)$")


class EOSSpec(PanelSpec):
    """Equation-of-state panel: explicit references or a declared build.

    Not a curated default -- it renders only from an explicit ``add:`` entry,
    which must carry everything EOS-specific itself (there are no run-global
    EOS settings to inherit):

    * exactly one of ``references`` (per-phase volume-scan extxyz, already
      DFT-labelled, or a directory holding them) or ``element`` (declare a
      reference need the run-side ``resolve_references`` fetches + DFT-labels,
      with on-disk reuse);
    * ``target_cutoff`` -- the cutoff (Å, the model's r_max) each unit-cell
      reference frame is replicated to host before scoring. Required: a wrong
      value silently mis-scores every frame, so there is no default.

    ``backend`` / ``remote_eval`` default to the run's *model* settings (the
    auto-detected backend and ``--remote-model-eval``, shared with the
    model-parity panel) -- they describe the model, not the EOS.
    """

    type: str = "eos"
    references: list | str | None = None
    element: str | None = None
    vol_scalings: list | None = None
    n_vol: int = 11
    vol_min: float = 0.9
    vol_max: float = 1.1
    target_cutoff: float
    level: str = "dft"
    backend: str | None = None
    remote_eval: bool | None = None
    max_neighbors: int = 50
    max_neighbors_list: list | None = None
    max_neighbors_offset: int = 4
    batch_size: int | None = None

    @model_validator(mode="after")
    def _one_reference_source(self):
        if (self.references is None) == (self.element is None):
            raise ValueError(
                "eos panel needs exactly one of 'references' (volume-scan "
                "file(s) or a directory) or 'element' (build them)."
            )
        return self


def _build_eos(spec, ctx, c):
    if not ctx.model_file:
        return None
    backend = spec.backend or ctx.eos_backend
    remote_eval = (
        spec.remote_eval
        if spec.remote_eval is not None
        else ctx.remote_model_eval
    )
    # The panel replicates each (unit-cell) reference frame to a supercell
    # hosting this cutoff before scoring the model; it drives BOTH the
    # reference (element=) build and the explicit-file model re-score.
    common = dict(
        model_file=ctx.model_file,
        backend_type=backend,
        remote_eval=remote_eval,
        target_cutoff=spec.target_cutoff,
        max_neighbors=spec.max_neighbors,
        max_neighbors_list=spec.max_neighbors_list,
        max_neighbors_offset=spec.max_neighbors_offset,
        batch_size=spec.batch_size,
        **c,
    )
    if spec.element:
        vol = spec.vol_scalings
        if vol is None:
            vol = np.linspace(spec.vol_min, spec.vol_max, spec.n_vol).tolist()
        return EquationOfStatePanel(
            element=spec.element,
            vol_scalings=vol,
            level=spec.level,
            **common,
        )
    # A scalar (one file / one directory) must not be iterated char by char.
    references = spec.references
    if isinstance(references, str):
        references = [references]
    return EquationOfStatePanel(references=references, **common)


register_panel("eos", EOSSpec, _build_eos)
