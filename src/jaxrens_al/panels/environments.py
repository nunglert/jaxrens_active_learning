"""Project panel: PCA map of atomic environments (QUESTS descriptors).

Reads a set of configs, computes *per-atom* structural descriptors with the
``quests`` package, projects them to 2D, and scatters them coloured by species
-- a read on which local environments the sampler is exploring. Distinct from
``QuestsPCAPanel`` (which is per-*structure* and highlights the resampled
ones); this is per-*atom* environment coverage and is computed independently of
whether a QUESTS selector ran.

Heavy work (descriptors) runs in ``compute_data`` and alviz caches the result
as JSON, so later renders just reload it. The descriptor call sits behind the
``_descriptors`` seam so unit tests need neither ``quests`` nor the full
compute.
"""

import numpy as np
from alviz import Entry, JsonDict


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
    """

    name = "atomic_envs"

    def __init__(
        self, config_files, *args, k: int = 32, cutoff: float = 5.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._config_files = [str(p) for p in config_files]
        self._k = k
        self._cutoff = cutoff

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

        desc = np.asarray(self._descriptors(atoms_list), dtype=float)
        symbols = [s for at in atoms_list for s in at.get_chemical_symbols()]
        coords = _pca_2d(desc)
        return JsonDict(
            pc1=coords[:, 0].tolist(),
            pc2=coords[:, 1].tolist(),
            species=symbols,
        )

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
        self.ax.set_title("Atomic environments")
