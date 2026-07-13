"""Project panel: PCA map of atomic environments (QUESTS descriptors).

Reads a set of configs, computes *per-atom* structural descriptors with the
``quests`` package, projects them to 2D, and scatters them coloured by species
-- a read on which local environments the sampler is exploring. Distinct from
``QuestsPCAPanel`` (which is per-*structure* and highlights the resampled
ones); this is per-*atom* environment coverage and is computed independently of
whether a QUESTS selector ran.

By default the PCA is fit on the mapped configs themselves, so the axes drift
between renders. Given a **reference database** (elemental phases drawn from the
Materials Project + rattled), the panel instead fits the PCA on the reference's
environments and projects the sampled ones into that fixed space -- a stable,
physically-grounded map -- with the reference drawn as a faint background so the
sampled coverage is read *against* it. The reference is supplied either as
explicit prebuilt configs (``reference_configs=``, a file/dir/list; works with
the light CLI) or declared as a
:class:`~jaxrens_al.dashboard.panels.reference.ReferenceNeed` via
``reference_element=`` that the run-side ``resolve_references`` fetches (the
``mp_elemental`` kind) -- the same dual the EOS panel exposes. Without either,
it falls back to the sampled-fit above.

The sampler is now a pure producer (its output is the *full* NS dead-point
pool), so descriptors over every config would be enormous. The panel therefore
evenly **subsamples** the configs down to ``max_configs`` before computing
descriptors -- the visualization analogue of the stride the old sampler applied
itself, kept here (not routed through the selector, whose choice would bias the
picture). Heavy work (descriptors) runs in ``compute_data`` and alviz caches
the result as JSON, so later renders just reload it. The descriptor call sits
behind the ``_descriptors`` seam so unit tests need neither ``quests`` nor the
full compute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Panel spec.
# ---------------------------------------------------------------------------
from alviz import Entry, JsonDict, PanelSpec, register_panel

from jaxrens_al.dashboard.panels.reference import (
    ReferenceConsumer,
    ReferenceNeed,
)


def _expand_reference_paths(paths):
    """Normalise a file / dir / list of reference configs to a list of files.

    A directory expands to its sorted ``*.extxyz`` files; a single path becomes
    a one-element list; ``None`` stays ``None`` (no reference).
    """
    if paths is None:
        return None
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            return [str(f) for f in sorted(p.glob("*.extxyz"))]
        return [str(p)]
    return [str(p) for p in paths]


def _fit_pca_2d(fit_desc: np.ndarray):
    """Fit a (≤2)-component PCA on ``fit_desc``; return a transform ``fn``.

    ``fn(desc) -> (n, 2)`` projects rows of ``desc`` onto the components learnt
    from ``fit_desc`` (second column zero when only one component is available,
    e.g. a single environment or a 1-D descriptor). An empty/degenerate
    ``fit_desc`` yields an all-zero transform.
    """
    if fit_desc.ndim != 2 or fit_desc.shape[0] == 0:
        return lambda desc: np.zeros(
            (desc.shape[0] if desc.ndim == 2 else 0, 2)
        )
    from sklearn.decomposition import PCA

    n = min(2, fit_desc.shape[0], fit_desc.shape[1])
    pca = PCA(n_components=n).fit(fit_desc)

    def _transform(desc: np.ndarray) -> np.ndarray:
        if desc.ndim != 2 or desc.shape[0] == 0:
            return np.zeros((0, 2))
        proj = pca.transform(desc)
        if proj.shape[1] < 2:
            proj = np.column_stack([proj[:, 0], np.zeros(proj.shape[0])])
        return proj

    return _transform


class AtomicEnvsPanel(ReferenceConsumer, Entry):
    """Per-atom QUESTS-descriptor PCA of a set of configs, coloured by species.

    Args:
        config_files: extxyz files whose atomic environments are mapped.
        k: QUESTS descriptor neighbour count.
        cutoff: QUESTS descriptor cutoff radius (Angstrom).
        max_configs: Cap on how many configs are descriptor-mapped (per set); if
            the input has more, they are evenly subsampled down to this many
            (``None`` disables the cap). Bounds the cost on the full NS pool.
        reference_configs: Prebuilt reference configs (file/dir/list) to fit the
            PCA on instead of the sampled data -- the sampled envs are projected
            into this fixed space and the reference is drawn as a background.
        reference_element: Instead of explicit files, declare a ``mp_elemental``
            :class:`ReferenceNeed` (Materials-Project elemental phases, rattled)
            the run-side resolver fetches and fills in.
        reference_target_cutoff / reference_theoretical: parameters of the
            declared need (supercell cutoff for rattling; include theoretical
            MP phases).
    """

    name = "atomic_envs"

    def __init__(
        self,
        config_files,
        *args,
        k: int = 32,
        cutoff: float = 5.0,
        max_configs: int = 500,
        reference_configs=None,
        reference_element: str | None = None,
        reference_target_cutoff: float = 3.0,
        reference_theoretical: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._config_files = [str(p) for p in config_files]
        self._k = k
        self._cutoff = cutoff
        self._max_configs = max_configs
        self._reference_configs = _expand_reference_paths(reference_configs)
        self._reference_element = reference_element
        self._reference_target_cutoff = float(reference_target_cutoff)
        self._reference_theoretical = bool(reference_theoretical)

    # -- reference (PCA-fit) wiring -----------------------------------------
    def _ref_need(self) -> ReferenceNeed:
        return ReferenceNeed.make(
            "mp_elemental",
            {
                "element": self._reference_element,
                "target_cutoff": self._reference_target_cutoff,
                "theoretical": self._reference_theoretical,
            },
        )

    def reference_needs(self) -> list:
        if self._reference_element is None or self._reference_configs:
            return []
        return [self._ref_need()]

    def receive_references(self, mapping: dict) -> None:
        need = self._ref_need()
        if (
            self._reference_element is not None
            and not self._reference_configs
            and need in mapping  # skipped (e.g. unresolvable) -> self-fit
        ):
            self._reference_configs = [str(p) for p in mapping[need]]

    def _descriptors(self, atoms_list):
        """Per-atom QUESTS descriptors (seam for testing)."""
        from quests.descriptor import get_descriptors

        return get_descriptors(atoms_list, k=self._k, cutoff=self._cutoff)

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def _read(self, paths):
        import ase.io

        atoms_list = []
        for path in paths:
            atoms_list.extend(ase.io.read(path, index=":"))
        return atoms_list

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        atoms_list = self._read(self._config_files)
        n_total = len(atoms_list)
        atoms_list = self._subsample(atoms_list)
        desc = np.asarray(self._descriptors(atoms_list), dtype=float)
        symbols = [s for at in atoms_list for s in at.get_chemical_symbols()]

        # Fit the PCA on the reference envs when available, else on the sampled
        # ones (the legacy self-fit). Project the sampled envs into that space;
        # also project the reference for the background scatter.
        ref_desc = self._reference_descriptors()
        if ref_desc is not None and ref_desc.shape[0]:
            transform = _fit_pca_2d(ref_desc)
            ref_coords = transform(ref_desc)
            fitted_on = "reference"
        else:
            transform = _fit_pca_2d(desc)
            ref_coords = np.zeros((0, 2))
            fitted_on = "sampled"
        coords = transform(desc)

        return JsonDict(
            pc1=coords[:, 0].tolist(),
            pc2=coords[:, 1].tolist(),
            species=symbols,
            ref_pc1=ref_coords[:, 0].tolist(),
            ref_pc2=ref_coords[:, 1].tolist(),
            fitted_on=fitted_on,
            n_total=n_total,
            n_used=len(atoms_list),
        )

    def _reference_descriptors(self):
        """Descriptors of the (subsampled) reference set, or ``None``."""
        if not self._reference_configs:
            return None
        ref_atoms = self._subsample(self._read(self._reference_configs))
        if not ref_atoms:
            return None
        return np.asarray(self._descriptors(ref_atoms), dtype=float)

    def _subsample(self, atoms_list):
        """Evenly thin ``atoms_list`` to at most ``max_configs`` entries."""
        n = len(atoms_list)
        if not self._max_configs or n <= self._max_configs:
            return atoms_list
        idx = np.unique(
            np.linspace(0, n - 1, self._max_configs).round().astype(int)
        )
        return [atoms_list[i] for i in idx]

    def plot_data(self):
        pc1 = np.asarray(self.data["pc1"], dtype=float)
        pc2 = np.asarray(self.data["pc2"], dtype=float)
        species = np.asarray(self.data["species"])
        ref_pc1 = np.asarray(self.data.get("ref_pc1", []), dtype=float)
        ref_pc2 = np.asarray(self.data.get("ref_pc2", []), dtype=float)

        # Reference environments as a faint grey background, sampled on top.
        if ref_pc1.size:
            self.ax.scatter(
                ref_pc1,
                ref_pc2,
                s=8,
                alpha=0.25,
                color="0.6",
                linewidths=0,
                zorder=1,
                label="reference",
            )
        if pc1.size:
            for sym in sorted(set(species.tolist())):
                m = species == sym
                self.ax.scatter(
                    pc1[m], pc2[m], s=6, alpha=0.5, zorder=2, label=sym
                )
            self.ax.legend(title="species", fontsize="small", loc="best")
        self.ax.set_xlabel("PC1")
        self.ax.set_ylabel("PC2")
        self.ax.set_title(self._title())

    def _title(self) -> str:
        n_used = self.data.get("n_used")
        n_total = self.data.get("n_total")
        suffix = (
            " [PCA on reference]"
            if self.data.get("fitted_on") == "reference"
            else ""
        )
        if n_used is not None and n_total is not None and n_used < n_total:
            return (
                f"Atomic environments ({n_used} of {n_total} configs){suffix}"
            )
        return f"Atomic environments{suffix}"


class AtomicEnvsSpec(PanelSpec):
    """PCA of atomic environments for one config set.

    ``source`` names the :class:`RunContext` attribute holding the configs to
    map (the sampled pool by default; a DFT-labelled or training set in the
    section dashboards), so one panel type serves every consumer.

    ``reference_configs`` (prebuilt files/dir) or ``reference_element`` (declare
    an ``mp_elemental`` reference need) fit the PCA on a reference database
    instead of the mapped configs; unset -> the legacy self-fit.
    """

    type: str = "atomic_envs"
    source: str = "sampled_files"
    reference_configs: list | str | None = None
    reference_element: str | None = None
    reference_target_cutoff: float = 3.0
    reference_theoretical: bool = False


def _build_atomic_envs(spec, ctx, c):
    files = getattr(ctx, spec.source, None)
    if files and not ctx.is_summary:
        return AtomicEnvsPanel(
            config_files=files,
            reference_configs=spec.reference_configs,
            reference_element=spec.reference_element,
            reference_target_cutoff=spec.reference_target_cutoff,
            reference_theoretical=spec.reference_theoretical,
            **c,
        )
    return None


register_panel("atomic_envs", AtomicEnvsSpec, _build_atomic_envs)
