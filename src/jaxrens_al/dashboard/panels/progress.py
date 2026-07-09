"""Project-specific dashboard panel: training-set growth across iterations.

This panel lives in the *project* (not in alviz) because it understands
remote_handler's run layout -- it reads ``AL_progress.json`` and counts the
configurations evaluated in each AL iteration. It subclasses ``alviz.Entry``
and stays pure (ase + numpy), so alviz remains ignorant of the orchestrator.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Panel spec.
# ---------------------------------------------------------------------------
from alviz import Entry, JsonDict, register_simple

from jaxrens_al.dashboard.run_layout import evaluated_configs_per_iteration


class DatasetGrowthPanel(Entry):
    """Per-iteration and cumulative count of evaluated configurations.

    Args:
        calchome: Path to a remote_handler run directory.
    """

    name = "dataset_growth"

    def __init__(self, calchome, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calchome = calchome

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])
        self.ax.set_title("Training-set growth")

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        import ase.io

        per_iter = []
        for paths in evaluated_configs_per_iteration(self._calchome):
            n = 0
            for path in paths:
                n += len(ase.io.read(path, index=":"))
            per_iter.append(n)

        return JsonDict(
            iterations=list(range(len(per_iter))),
            per_iteration=per_iter,
            cumulative=np.cumsum(per_iter, dtype=int).tolist(),
        )

    def plot_data(self):
        it = self.data["iterations"]
        self.ax.bar(
            it,
            self.data["per_iteration"],
            color="tab:blue",
            alpha=0.6,
            label="per iteration",
        )
        self.ax.set_xlabel("AL iteration")
        self.ax.set_ylabel("# configs added", color="tab:blue")

        ax2 = self.ax.twinx()
        ax2.plot(
            it,
            self.data["cumulative"],
            color="tab:red",
            marker="o",
            label="cumulative",
        )
        ax2.set_ylabel("# configs total", color="tab:red")


def _build_dataset_growth(ctx, c):
    # Cumulative training-set growth -> summary page only.
    return (
        DatasetGrowthPanel(calchome=ctx.calchome, **c)
        if ctx.is_summary
        else None
    )


register_simple("dataset_growth", _build_dataset_growth)
