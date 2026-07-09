"""Project panels for the QUESTS resampling stage of an AL iteration.

When the jaxrens NS sampler selects candidates with a ``QUESTSSelector`` it
leaves two per-iteration artifacts under ``<i>/ns_sampler/results/`` (located
via :mod:`jaxrens_al.dashboard.run_layout`):

* ``quests_pca.json`` -- every candidate's PCA-projected descriptor + a
  ``selected`` flag (written by the selector). Read by :class:`QuestsPCAPanel`.
* ``resampled_provenance.json`` -- the *resampled* configs with their
  ``run_id`` (replica) and ``iter`` (NS dead-point iteration), written by the
  sampler. Read by :class:`ResampledGridPanel`.

Both are pure (json + numpy) and subclass ``alviz.Entry``, so alviz stays
ignorant of the run layout.
"""

import json

import numpy as np

# ---------------------------------------------------------------------------
# Panel specs.
# ---------------------------------------------------------------------------
from alviz import Entry, JsonDict, register_simple


class QuestsPCAPanel(Entry):
    """QUESTS descriptor PCA map, highlighting the resampled configurations.

    Scatters every candidate at its (PC1, PC2); selected (resampled) configs
    are drawn on top so you can see *where* in descriptor space they were
    picked from -- a quick read on whether the selector spread across the
    population (diversity) or clustered (novelty).

    Args:
        pca_json: Path to a ``quests_pca.json`` written by ``QUESTSSelector``.
    """

    name = "quests_pca"

    def __init__(self, pca_json, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pca_json = str(pca_json)

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        with open(self._pca_json) as f:
            data = json.load(f)
        structs = data.get("structures", [])
        return JsonDict(
            pc1=[float(s["pc1"]) for s in structs],
            pc2=[float(s["pc2"]) for s in structs],
            selected=[bool(s.get("selected")) for s in structs],
            mode=data.get("mode", ""),
            method=data.get("method", ""),
            evr=[float(x) for x in data.get("explained_variance_ratio", [])],
        )

    def plot_data(self):
        pc1 = np.asarray(self.data["pc1"], dtype=float)
        pc2 = np.asarray(self.data["pc2"], dtype=float)
        sel = np.asarray(self.data["selected"], dtype=bool)

        if pc1.size:
            self.ax.scatter(
                pc1[~sel],
                pc2[~sel],
                s=8,
                c="0.7",
                alpha=0.5,
                label="candidates",
            )
            self.ax.scatter(
                pc1[sel],
                pc2[sel],
                s=20,
                c="tab:red",
                edgecolors="k",
                linewidths=0.3,
                label="resampled",
            )
            self.ax.legend(loc="best", fontsize="small")

        evr = self.data["evr"]
        xlab = f"PC1 ({evr[0]:.0%})" if len(evr) >= 1 else "PC1"
        ylab = f"PC2 ({evr[1]:.0%})" if len(evr) >= 2 else "PC2"
        self.ax.set_xlabel(xlab)
        self.ax.set_ylabel(ylab)
        mode, method = self.data["mode"], self.data["method"]
        suffix = f" ({mode}/{method})" if mode or method else ""
        self.ax.set_title(f"QUESTS map{suffix}")


class ResampledGridPanel(Entry):
    """Resampled configs scattered on a (NS iteration, replica) grid.

    Each resampled config is placed at its ``iter`` (x) and ``run_id`` (y),
    showing which replicas and which parts of the NS run the labelled configs
    were drawn from. Reads the *selected* records of
    ``resampled_provenance.json``.

    Args:
        provenance_json: Path to a ``resampled_provenance.json``.
    """

    name = "resampled_grid"

    def __init__(self, provenance_json, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._provenance_json = str(provenance_json)

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        with open(self._provenance_json) as f:
            data = json.load(f)
        iters, run_ids = [], []
        for rec in data.get("selected", []):
            it, run = rec.get("iter"), rec.get("run_id")
            if it is None or run is None:
                continue
            iters.append(int(it))
            run_ids.append(int(run))
        return JsonDict(iters=iters, run_ids=run_ids)

    def plot_data(self):
        iters = self.data["iters"]
        run_ids = self.data["run_ids"]
        if iters:
            self.ax.scatter(iters, run_ids, s=18, c="tab:red", alpha=0.7)
            # One integer tick per replica that actually appears.
            unique_runs = sorted(set(run_ids))
            self.ax.set_yticks(unique_runs)
        self.ax.set_xlabel("NS iteration")
        self.ax.set_ylabel("replica (run_id)")
        self.ax.set_title("Resampled configs")


def _build_quests_pca(ctx, c):
    return (
        QuestsPCAPanel(pca_json=ctx.quests_pca, **c)
        if ctx.quests_pca
        else None
    )


def _build_resampled_grid(ctx, c):
    if ctx.resampled_provenance:
        return ResampledGridPanel(
            provenance_json=ctx.resampled_provenance, **c
        )
    return None


register_simple("quests_pca", _build_quests_pca)
register_simple("resampled_grid", _build_resampled_grid)
