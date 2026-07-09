"""Project panels: adaptive-move diagnostics from the NS adaptation trace.

jaxrens writes a ``<prefix>.adaptation.h5`` log recording, per adaptation
record, per replica, per move: the tuned step size and the acceptance rate it
was tuned against (shape ``(n_records, n_runs, n_moves)``). These two panels
read it through jaxrens' own ``AdaptationLogger`` (lazy import, behind the
``_load_log`` seam so unit tests need neither jaxrens nor h5py) and render with
jaxrens' own ``plot_step_sizes`` / ``plot_acceptance_rates`` so the figures are
identical to ``jaxrens plot`` -- the mean across replicas with a shaded spread
(``per_run=False``), or one line per replica (``per_run=True``).

This is distinct from
:class:`~jaxrens_al.dashboard.panels.acceptance.MoveAcceptancePanel`, which reads a
separate ``<prefix>.acc_rates.h5`` log; the adaptation trace is what current
multi-run NS jobs emit.

``compute_data`` serializes the raw arrays to JSON (alviz caches it), and
``plot_data`` rebuilds an ``AdaptationLog`` from that cache before calling the
jaxrens plotter -- so a render off the JSON cache never reopens the HDF5 file.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Panel specs (parameterized: per_run one-line-per-replica toggle).
# ---------------------------------------------------------------------------
from alviz import Entry, JsonDict, PanelSpec, register_panel


class _AdaptationTracePanel(Entry):
    """Shared reader/cache for the two adaptation-trace panels.

    Args:
        adaptation_file: Path to a jaxrens ``<prefix>.adaptation.h5`` log.
        per_run: Plot one line per replica instead of the mean-across-replicas
            band.
    """

    data_class = JsonDict

    def __init__(
        self, adaptation_file, *args, per_run: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._adaptation_file = str(adaptation_file)
        self._per_run = per_run

    def _load_log(self, path):
        """Read the adaptation log (seam for testing)."""
        from jaxrens.io.adaptation_log import AdaptationLogger

        return AdaptationLogger.read(path)

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        log = self._load_log(self._adaptation_file)
        return JsonDict(
            iterations=np.asarray(log.iterations).tolist(),
            step_sizes=np.asarray(log.step_sizes, dtype=float).tolist(),
            acceptance_rates=np.asarray(
                log.acceptance_rates, dtype=float
            ).tolist(),
            move_names=list(log.move_names),
            n_runs=int(log.n_runs),
            n_moves=int(log.n_moves),
        )

    def _rebuild_log(self):
        """Reconstruct an ``AdaptationLog`` from the cached arrays."""
        from jaxrens.io.adaptation_log import AdaptationLog

        return AdaptationLog(
            iterations=np.asarray(self.data["iterations"]),
            step_sizes=np.asarray(self.data["step_sizes"], dtype=float),
            acceptance_rates=np.asarray(
                self.data["acceptance_rates"], dtype=float
            ),
            move_names=list(self.data["move_names"]),
            n_runs=int(self.data["n_runs"]),
            n_moves=int(self.data["n_moves"]),
        )


class StepSizePanel(_AdaptationTracePanel):
    """Per-move adaptive step size vs NS iteration (mean across replicas)."""

    name = "step_sizes"

    def plot_data(self):
        if not self.data["move_names"]:
            self.ax.set_title("step sizes (no adaptation trace)")
            return
        from jaxrens.postprocess.plotting import plot_step_sizes

        plot_step_sizes(self._rebuild_log(), ax=self.ax, per_run=self._per_run)
        self.ax.set_yscale("log")
        self.ax.set_title("Adaptive step size per move")
        self.ax.grid(alpha=0.3)


class AdaptationAcceptancePanel(_AdaptationTracePanel):
    """Per-move MCMC acceptance rate vs NS iteration (mean across replicas)."""

    name = "adaptation_acceptance"

    def plot_data(self):
        if not self.data["move_names"]:
            self.ax.set_title("move acceptance (no adaptation trace)")
            return
        from jaxrens.postprocess.plotting import plot_acceptance_rates

        plot_acceptance_rates(
            self._rebuild_log(), ax=self.ax, per_run=self._per_run
        )
        self.ax.set_title("Move acceptance rate per move")
        self.ax.grid(alpha=0.3)


class StepSizeSpec(PanelSpec):
    type: str = "step_sizes"
    per_run: bool = False


class AdaptationAcceptanceSpec(PanelSpec):
    type: str = "adaptation_acceptance"
    per_run: bool = False


def _build_step_sizes(spec, ctx, c):
    if not ctx.adaptation:
        return None
    return StepSizePanel(
        adaptation_file=ctx.adaptation, per_run=spec.per_run, **c
    )


def _build_adaptation_acceptance(spec, ctx, c):
    if not ctx.adaptation:
        return None
    return AdaptationAcceptancePanel(
        adaptation_file=ctx.adaptation, per_run=spec.per_run, **c
    )


register_panel("step_sizes", StepSizeSpec, _build_step_sizes)
register_panel(
    "adaptation_acceptance",
    AdaptationAcceptanceSpec,
    _build_adaptation_acceptance,
)
