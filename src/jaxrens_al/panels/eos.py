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
from alviz import Entry, JsonDict

# 1 eV / Angstrom^3 expressed in GPa.
_EVA3_TO_GPA = 160.21766208

# Pressure grid (GPa) the per-phase enthalpies are interpolated onto, so all
# phases share an x-axis for the H(P) / relative-H panels.
_P_GRID = np.linspace(-2.0, 25.0, 100)


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


class EquationOfStatePanel(Entry):
    """Multi-phase EOS: E(V), H(P) and relative enthalpy under the model + DFT.

    Args:
        references: One path or a list of paths -- each is a phase's
            volume-scan extxyz; ``REF_energy`` on each frame is the DFT energy.
        model_file: This iteration's trained-model bundle.
        labels: Optional phase labels (default: the reference file stems).
        backend_type: jaxrens backend to load the model with.
        max_neighbors: Static neighbor-buffer size for the backend call.
    """

    name = "eos"
    colspan = 3

    def __init__(
        self,
        references,
        model_file,
        *args,
        labels=None,
        backend_type: str = "neuralil",
        max_neighbors: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from pathlib import Path

        if isinstance(references, (str, Path)):
            references = [references]
        self._references = [str(p) for p in references]
        self._model_file = str(model_file)
        self._labels = (
            list(labels)
            if labels is not None
            else [Path(p).stem for p in self._references]
        )
        self._backend_type = backend_type
        self._max_neighbors = max_neighbors

    def _evaluate(self, model_file, atoms_list) -> list:
        """Total energies (eV) of each structure under the model (seam)."""
        from jaxrens.backends.loader import load_backend
        from jaxrens.io.formats import ase_atoms_to_walker

        if self._backend_type == "mace":
            backend = load_backend("mace", model_path=model_file)
        else:
            backend = load_backend(self._backend_type, pickle_file=model_file)

        symbol_map = getattr(backend, "symbol_map", None)
        if symbol_map is None:
            from ase.data import chemical_symbols

            symbol_map = {
                chemical_symbols[int(z)]: i
                for i, z in enumerate(backend.atomic_numbers)
            }

        energies = []
        for atoms in atoms_list:
            w = ase_atoms_to_walker(atoms, symbol_map)
            energies.append(
                float(
                    backend(w.positions, w.types, w.cell, self._max_neighbors)[
                        0
                    ]
                )
            )
        return energies

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 3, wspace=0.35)
        self.ax_ev = fig.add_subplot(self.gs[0, 0])
        self.ax_hp = fig.add_subplot(self.gs[0, 1])
        self.ax_dh = fig.add_subplot(self.gs[0, 2])

    def _phase_data(self, ref_file) -> dict:
        import ase.io

        frames = ase.io.read(ref_file, index=":")
        n = len(frames[0])
        volumes = [at.get_volume() / n for at in frames]
        e_dft = [float(at.info["REF_energy"]) / n for at in frames]
        e_nnff = [e / n for e in self._evaluate(self._model_file, frames)]
        return dict(
            volumes=volumes,
            e_dft=e_dft,
            e_nnff=e_nnff,
            h_dft=_fit_enthalpy(volumes, e_dft).tolist(),
            h_nnff=_fit_enthalpy(volumes, e_nnff).tolist(),
        )

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        phases = [self._phase_data(f) for f in self._references]
        return JsonDict(
            labels=self._labels,
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
