"""Project panel: dead-point NS energy vs iteration, one line per replica.

A quick read on whether every replica of a multi-run NS job is descending the
energy landscape in step, or one has stalled / diverged. Reads the per-replica
energy logs through jaxrens' ``MonitorCollection.from_multi_run_directory``
(lazy import, behind the ``_load_monitors`` seam so unit tests need neither
jaxrens nor a real ``.energies`` file).

The raw NS log is one entry per dead point (tens of thousands);
``compute_data`` subsamples to at most ``_TRACE_MAX_POINTS`` per replica so the
cached JSON and the PDF panel stay light.
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Panel spec: this panel wired from a RunContext (see jaxrens_al.dashboard.panel_specs).
# ---------------------------------------------------------------------------
from alviz import Entry, JsonDict, register_simple

# Cap on points plotted per replica -- the dead-point log is far denser than a
# PDF panel needs.
_TRACE_MAX_POINTS = 2000


class NSEnergyTracePanel(Entry):
    """Dead-point NS energy vs iteration, one viridis line per replica.

    Args:
        result_dirs: One or more multi-run NS output dirs (each holding
            ``<prefix>.runNN.energies``). Every replica is one line. Empty ->
            nothing to plot.
        prefix: jaxrens ``out_file_prefix`` of the run files.
        energy_unit: Axis-label unit for the energies (default ``eV``).
    """

    name = "ns_energy_trace"
    data_class = JsonDict

    def __init__(
        self,
        result_dirs,
        *args,
        prefix: str = "ns",
        energy_unit="eV",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._result_dirs = [str(p) for p in result_dirs]
        self._prefix = prefix
        self._energy_unit = energy_unit

    def _load_monitors(self, result_dir):
        """Monitors for one NS output dir, one per replica (test seam)."""
        from jaxrens.postprocess import MonitorCollection

        return list(
            MonitorCollection.from_multi_run_directory(
                result_dir, prefix=self._prefix
            )
        )

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        traces = []
        for result_dir in self._result_dirs:
            for mon in self._load_monitors(result_dir):
                it = np.asarray(mon.iteration_trace)
                en = np.asarray(mon.energy_trace, dtype=float)
                step = max(1, len(it) // _TRACE_MAX_POINTS)
                traces.append(
                    dict(
                        label=str(mon.label),
                        iterations=it[::step].tolist(),
                        energies=en[::step].tolist(),
                    )
                )
        return JsonDict(traces=traces, energy_unit=self._energy_unit)

    def plot_data(self):
        traces = self.data["traces"]
        cmap = plt.get_cmap("viridis")
        n = max(1, len(traces) - 1)
        for i, tr in enumerate(traces):
            self.ax.plot(
                tr["iterations"],
                tr["energies"],
                color=cmap(i / n),
                lw=0.8,
                label=tr["label"],
            )
        self.ax.set_xlabel("NS iteration")
        self.ax.set_ylabel(f"dead-point energy [{self.data['energy_unit']}]")
        self.ax.set_title("Energy vs iteration (per replica)")
        self.ax.grid(alpha=0.3)
        if 0 < len(traces) <= 10:
            self.ax.legend(fontsize=6, ncol=2)


def _build_ns_energy_trace(ctx, c):
    if ctx.ns_dirs and not ctx.is_summary:
        return NSEnergyTracePanel(
            result_dirs=ctx.ns_dirs, prefix=ctx.ns_prefix, **c
        )
    return None


register_simple("ns_energy_trace", _build_ns_energy_trace)
