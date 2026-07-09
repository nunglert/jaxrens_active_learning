"""Project panel: nested-sampling thermodynamics heatmap.

Equips alviz's abstract ``HeatmapPanel`` (which owns the (P, T) pcolormesh
rendering + ``HeatmapData`` contract, but ships no reader) with the
nested-sampling computation. The numbers come from jaxrens'
``MonitorCollection.from_multi_run_directory`` -- jaxrens NS jobs are
*multi-run* (one combined checkpoint plus per-replica
``<prefix>.runNN.energies`` logs), so this reads one ``Monitor`` per replica
and evaluates
``heat_capacity`` / ``expectation`` over a temperature grid. That postprocess
path is pure NumPy/SciPy, so building this dashboard does not pull in JAX; the
``jaxrens`` import is deferred to ``compute_data`` and only the actual NS
postprocessing needs it.

Each replica is one heatmap row (a replica-exchange ladder is typically a
pressure or temperature sweep, so the rows *are* the sweep). A genuine
single-run output is just the one-replica edge case -> a degenerate (1, n_T)
heatmap that still reads as a C_v(T) / <V>(T) curve. Pass several run dirs to
stack their replicas into one grid.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Panel spec (parameterized: which thermodynamic quantity).
# ---------------------------------------------------------------------------
from alviz import PanelSpec, register_panel
from alviz.panels import HeatmapData, HeatmapPanel, heatmap_pcolormesh

_QUANTITIES = ("heat_capacity", "volume")


class NSThermodynamicsHeatmap(HeatmapPanel):
    """Heat-capacity or mean-volume vs temperature from jaxrens NS runs.

    Args:
        result_dirs: One or more multi-run NS output dirs, each holding the
            per-replica ``<prefix>.runNN.energies`` + a combined
            ``<prefix>.final.checkpoint.h5``. Every replica across every dir is
            one heatmap row. Empty -> nothing to plot.
        quantity: ``"heat_capacity"`` or ``"volume"``.
        temperatures: Explicit temperature array, or None to build
            ``n_T`` points spanning ``tlim``.
        tlim: (T_min, T_max) when ``temperatures`` is None.
        n_T: Number of temperature points when ``temperatures`` is None.
        k_B: Boltzmann constant in the energy units of the stored energies
            (1.0 for reduced units; 8.617e-5 for eV with T in Kelvin).
        prefix: jaxrens ``out_file_prefix`` of the run files.
        pressures: Optional row labels (one per replica row); defaults to row
            indices.
        swap_axes: If True, render with pressure on the x-axis and temperature
            on the y-axis (the grid is transposed and the labels swapped).
            Default keeps alviz's T-on-x / P-on-y orientation.
    """

    name = "ns_thermodynamics"

    def __init__(
        self,
        result_dirs,
        *args,
        quantity: str = "heat_capacity",
        temperatures=None,
        tlim=(0.05, 2.0),
        n_T: int = 50,
        k_B: float = 1.0,
        prefix: str = "ns",
        pressures=None,
        swap_axes: bool = False,
        plot_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(*args, plot_kwargs=plot_kwargs, **kwargs)
        if quantity not in _QUANTITIES:
            raise ValueError(
                f"quantity must be one of {_QUANTITIES}, got {quantity!r}"
            )
        self._result_dirs = [str(p) for p in result_dirs]
        self._quantity = quantity
        if temperatures is not None:
            self._temperatures = np.asarray(temperatures, dtype=float)
        else:
            self._temperatures = np.linspace(tlim[0], tlim[1], n_T)
        self._k_B = k_B
        self._prefix = prefix
        self._pressures = pressures
        self._swap_axes = swap_axes

    def _load_monitors(self, result_dir):
        """Monitors for one NS output dir, one per replica (test seam).

        Multi-run is the only loader: a single-run output is just the
        one-replica edge case of ``from_multi_run_directory``.
        """
        from jaxrens.postprocess import MonitorCollection

        return list(
            MonitorCollection.from_multi_run_directory(
                result_dir, prefix=self._prefix
            )
        )

    def compute_data(self, shared_ressources: dict) -> HeatmapData:
        if not self._result_dirs:
            raise RuntimeError(
                "NSThermodynamicsHeatmap: no NS result directories given."
            )

        T = self._temperatures
        rows = []
        for result_dir in self._result_dirs:
            for mon in self._load_monitors(result_dir):
                if self._quantity == "heat_capacity":
                    obs = mon.heat_capacity(T, k_B=self._k_B)
                else:  # volume
                    obs = mon.expectation(mon.dead_volumes, T, k_B=self._k_B)
                rows.append(np.asarray(obs, dtype=float).tolist())

        if self._pressures is not None:
            p = [float(x) for x in self._pressures]
        else:
            p = [float(i) for i in range(len(rows))]

        return HeatmapData(
            t=T.tolist(), p=p, obs=rows, quantity=self._quantity
        )

    def plot_data(self):
        # Default orientation (T on x, P on y) is alviz's; defer to it.
        if not self._swap_axes:
            return super().plot_data()

        # P on x, T on y: transpose the grid and swap the axis labels, but
        # still render through alviz's helper (colorbar/scaling/cmap reused).
        d = self.data
        kw = dict(self._plot_kwargs)
        if d.quantity and "label" not in kw:
            kw["label"] = d.quantity
        t_label = kw.pop("T_label", r"$T$")
        p_label = kw.pop("P_label", r"$\log P$")
        obs_t = np.asarray(d.obs, dtype=float).T  # (len(t), len(p))
        # heatmap_pcolormesh plots pcolormesh(arg1, arg2, obs) with
        # xlabel=T_label, ylabel=P_label -> feed pressures as x, temps as y
        # and route the labels accordingly.
        heatmap_pcolormesh(
            self.ax, d.p, d.t, obs_t, T_label=p_label, P_label=t_label, **kw
        )
        if d.quantity:
            self.ax.set_title(d.quantity)


class ThermodynamicsSpec(PanelSpec):
    type: str = "ns_thermodynamics"
    quantity: str = "heat_capacity"
    # NS-global settings a panel opts into; unset -> inherit the run global
    # (alviz.apply_globals), set -> override this instance.
    swap_axes: bool | None = None
    k_B: float | None = None
    temperatures: list | None = None


def _build_thermodynamics(spec, ctx, c):
    if not ctx.ns_dirs:
        return None
    return NSThermodynamicsHeatmap(
        result_dirs=ctx.ns_dirs,
        quantity=spec.quantity,
        temperatures=spec.temperatures,
        k_B=spec.k_B,
        prefix=ctx.ns_prefix,
        swap_axes=spec.swap_axes,
        **c,
    )


register_panel("ns_thermodynamics", ThermodynamicsSpec, _build_thermodynamics)
