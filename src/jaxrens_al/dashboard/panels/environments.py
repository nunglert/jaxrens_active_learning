"""Project panel: PCA map of atomic environments (QUESTS descriptors).

Reads a set of configs, computes *per-atom* structural descriptors with the
``quests`` package, projects them to 2D, and scatters them coloured by species
-- a read on which local environments the sampler is exploring. Distinct from
``QuestsPCAPanel`` (which is per-*structure* and highlights the resampled
ones); this is per-*atom* environment coverage and is computed independently of
whether a QUESTS selector ran.

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

import numpy as np

# ---------------------------------------------------------------------------
# Panel spec.
# ---------------------------------------------------------------------------
from alviz import Entry, JsonDict, PanelSpec, register_panel


def _pca_2d(desc: np.ndarray) -> np.ndarray:
    """Project rows of ``desc`` onto their first two principal components.

    Returns an ``(n, 2)`` array (second column zero if only one component is
    available, e.g. a single environment or a 1-D descriptor).
    """
    if desc.ndim != 2 or desc.shape[0] == 0:
        return np.zeros((desc.shape[0] if desc.ndim == 2 else 0, 2))
    from sklearn.decomposition import PCA

    n = min(2, desc.shape[0], desc.shape[1])
    proj = PCA(n_components=n).fit_transform(desc)
    if proj.shape[1] < 2:
        proj = np.column_stack([proj[:, 0], np.zeros(proj.shape[0])])
    return proj


class AtomicEnvsPanel(Entry):
    """Per-atom QUESTS-descriptor PCA of a set of configs, coloured by species.

    Args:
        config_files: extxyz files whose atomic environments are mapped.
        k: QUESTS descriptor neighbour count.
        cutoff: QUESTS descriptor cutoff radius (Angstrom).
        max_configs: Cap on how many configs are descriptor-mapped; if the
            input has more, they are evenly subsampled down to this many
            (``None`` disables the cap). Bounds the cost on the full NS pool.
    """

    name = "atomic_envs"

    def __init__(
        self,
        config_files,
        *args,
        k: int = 32,
        cutoff: float = 5.0,
        max_configs: int = 500,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._config_files = [str(p) for p in config_files]
        self._k = k
        self._cutoff = cutoff
        self._max_configs = max_configs

    def _descriptors(self, atoms_list):
        """Per-atom QUESTS descriptors (seam for testing)."""
        from quests.descriptor import get_descriptors

        return get_descriptors(atoms_list, k=self._k, cutoff=self._cutoff)

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        import ase.io

        atoms_list = []
        for path in self._config_files:
            atoms_list.extend(ase.io.read(path, index=":"))

        n_total = len(atoms_list)
        atoms_list = self._subsample(atoms_list)

        desc = np.asarray(self._descriptors(atoms_list), dtype=float)
        symbols = [s for at in atoms_list for s in at.get_chemical_symbols()]
        coords = _pca_2d(desc)
        return JsonDict(
            pc1=coords[:, 0].tolist(),
            pc2=coords[:, 1].tolist(),
            species=symbols,
            n_total=n_total,
            n_used=len(atoms_list),
        )

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
        if pc1.size:
            for sym in sorted(set(species.tolist())):
                m = species == sym
                self.ax.scatter(pc1[m], pc2[m], s=6, alpha=0.5, label=sym)
            self.ax.legend(title="species", fontsize="small", loc="best")
        self.ax.set_xlabel("PC1")
        self.ax.set_ylabel("PC2")
        n_used = self.data.get("n_used")
        n_total = self.data.get("n_total")
        if n_used is not None and n_total is not None and n_used < n_total:
            self.ax.set_title(
                f"Atomic environments ({n_used} of {n_total} configs)"
            )
        else:
            self.ax.set_title("Atomic environments")


class AtomicEnvsSpec(PanelSpec):
    """PCA of atomic environments for one config set.

    ``source`` names the :class:`RunContext` attribute holding the configs to
    map (the sampled pool by default; a DFT-labelled or training set in the
    section dashboards), so one panel type serves every consumer.
    """

    type: str = "atomic_envs"
    source: str = "sampled_files"


def _build_atomic_envs(spec, ctx, c):
    files = getattr(ctx, spec.source, None)
    if files and not ctx.is_summary:
        return AtomicEnvsPanel(config_files=files, **c)
    return None


register_panel("atomic_envs", AtomicEnvsSpec, _build_atomic_envs)
