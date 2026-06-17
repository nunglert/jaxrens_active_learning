"""Project panel: per-move MCMC acceptance rates across an NS run.

jaxrens writes a ``<prefix>.acc_rates.h5`` log -- per iteration, per run, per
move it records accepted/proposed counts (replica-exchange and chain moves
alike). This panel reads it through jaxrens' own reader (lazy import, behind
the ``_load_log`` seam so unit tests need neither jaxrens nor h5py) and plots
each move's acceptance rate vs NS iteration: a quick read on whether moves are
mixing or a step size needs tuning.

Heavy/optional work is fine here -- ``compute_data`` runs once and alviz caches
the result as JSON, so later renders just reload it.
"""

import numpy as np
from alviz import Entry, JsonDict


class MoveAcceptancePanel(Entry):
    """Acceptance rate of each NS move vs iteration (from ``.acc_rates.h5``).

    Args:
        acc_rates_file: Path to a jaxrens ``<prefix>.acc_rates.h5`` log.
    """

    name = "acceptance_rates"

    def __init__(self, acc_rates_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acc_rates_file = str(acc_rates_file)

    def _load_log(self, path):
        """Read the acc-rates log (seam for testing)."""
        from jaxrens.io.acc_rates_log import AccRatesLogger

        return AccRatesLogger.read(path)

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        log = self._load_log(self._acc_rates_file)
        # acceptance_rates: (n_entries, n_runs, n_moves); average over runs.
        rates = np.asarray(log.acceptance_rates, dtype=float)
        per_move = rates.mean(axis=1) if rates.ndim == 3 else rates
        return JsonDict(
            iterations=np.asarray(log.iterations).tolist(),
            move_names=list(log.move_names),
            rates=np.atleast_2d(per_move).tolist(),
        )

    def plot_data(self):
        it = self.data["iterations"]
        rates = np.asarray(self.data["rates"], dtype=float)
        names = self.data["move_names"]
        for j, name in enumerate(names):
            self.ax.plot(it, rates[:, j], marker=".", lw=1, label=name)
        self.ax.set_xlabel("NS iteration")
        self.ax.set_ylabel("acceptance rate")
        self.ax.set_ylim(0, 1)
        if names:
            self.ax.legend(loc="best", fontsize="small")
        self.ax.set_title("Move acceptance rates")


class RESwapAcceptancePanel(Entry):
    """Inter-replica-exchange swap acceptance vs iteration, per replica pair.

    Reads jaxrens' ``<prefix>.re_stats.h5`` (per swap *fire*, per adjacent
    replica pair: accepted/attempted counts) via its own reader, and renders it
    with jaxrens' own ``plot_re_acceptance`` so the figure matches what
    ``jaxrens plot`` produces -- a read on whether the replica ladder is
    exchanging well (too-low rates mean the pressures/temperatures are spaced
    too far apart).

    Args:
        re_stats_file: Path to a jaxrens ``<prefix>.re_stats.h5`` log.
    """

    name = "re_swap_rates"

    def __init__(self, re_stats_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._re_stats_file = str(re_stats_file)

    def _load_log(self, path):
        """Read the RE-stats log (seam for testing)."""
        from jaxrens.io.re_stats_log import RELogger

        return RELogger.read(path)

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        log = self._load_log(self._re_stats_file)
        rates = np.atleast_2d(np.asarray(log.acceptance_rates, dtype=float))
        return JsonDict(
            iterations=np.asarray(log.iterations).tolist(),
            rates=rates.tolist(),
            n_pairs=int(log.n_pairs),
            flavor=str(getattr(log, "flavor", "")),
        )

    def plot_data(self):
        flavor = self.data["flavor"]
        if not self.data["n_pairs"]:
            self.ax.set_title("RE swap acceptance (single replica)")
            return

        # Plot exactly the way jaxrens does: rebuild an RELog from the cached
        # rates (n_attempted=1 so RELog.acceptance_rates returns them as-is)
        # and hand it to jaxrens' own per-pair plotter.
        from jaxrens.io.re_stats_log import RELog
        from jaxrens.postprocess.plotting import plot_re_acceptance

        rates = np.asarray(self.data["rates"], dtype=float)
        re_log = RELog(
            iterations=np.asarray(self.data["iterations"]),
            n_accepted_per_pair=rates,
            n_attempted_per_pair=np.ones_like(rates),
            n_pairs=int(self.data["n_pairs"]),
            flavor=flavor,
        )
        plot_re_acceptance(re_log, ax=self.ax, per_pair=True)
        self.ax.grid(alpha=0.3)
        self.ax.set_title(f"RE swap acceptance ({flavor})")
