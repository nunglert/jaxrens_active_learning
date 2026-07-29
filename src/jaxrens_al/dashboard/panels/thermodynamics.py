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
from alviz.panels import HeatmapData, HeatmapPanel

from jaxrens_al.dashboard.panels.ns_expectation import (
    SwapAxesHeatmapMixin,
    load_monitors,
)

_QUANTITIES = ("heat_capacity", "volume")

# Colorbar/title label per quantity (``HeatmapData.quantity``).
_QUANTITY_LABEL = {
    "heat_capacity": "heat_capacity",
    "volume": r"$\langle V \rangle$ ($\mathrm{\AA}^3$)",
}


class NSThermodynamicsHeatmap(SwapAxesHeatmapMixin, HeatmapPanel):
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
        return load_monitors(result_dir, self._prefix)

    @staticmethod
    def _volumes(mon):
        """The per-dead-point cell volumes of one replica.

        ``MonitorCollection.from_multi_run_directory`` deliberately leaves
        ``dead_volumes`` None: for an NPT (``EnsembleBackend``) run the stored
        energy is already the enthalpy ``H = U + P*V``, so feeding the volume
        column into the NS weights would double-count ``P*V``. It stashes the
        bare cell-volume series as ``volume_trace`` instead -- which is exactly
        what an ``<V>(T)`` expectation needs. ``dead_volumes`` remains the
        fallback for a Monitor built by ``Monitor.from_directory``.
        """
        vols = getattr(mon, "volume_trace", None)
        if vols is None:
            vols = getattr(mon, "dead_volumes", None)
        if vols is None:
            raise ValueError(
                "NSThermodynamicsHeatmap(quantity='volume'): replica "
                f"{getattr(mon, 'label', '?')} carries no volume series "
                "('volume_trace' / 'dead_volumes') -- an NVT run has no "
                "volume to average."
            )
        return np.asarray(vols, dtype=float)

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
                    obs = mon.expectation(self._volumes(mon), T, k_B=self._k_B)
                rows.append(np.asarray(obs, dtype=float).tolist())

        if self._pressures is not None:
            p = [float(x) for x in self._pressures]
        else:
            p = [float(i) for i in range(len(rows))]

        return HeatmapData(
            t=T.tolist(),
            p=p,
            obs=rows,
            quantity=_QUANTITY_LABEL.get(self._quantity, self._quantity),
        )


class ThermodynamicsSpec(PanelSpec):
    type: str = "ns_thermodynamics"
    quantity: str = "heat_capacity"
    # NS-global settings a panel opts into; unset -> inherit the run global
    # (alviz.apply_globals), set -> override this instance.
    swap_axes: bool | None = None
    k_B: float | None = None
    temperatures: list | None = None


def _thermo_name(spec):
    """Quantity-keyed name, so several quantities coexist on one page.

    The heat-capacity instance keeps the historical ``ns_thermodynamics``
    name (config ``disable:`` / ``only:`` keys and JSON caches refer to it); a
    second quantity added via ``add:`` becomes ``ns_<quantity>`` --
    ``build_entries`` rejects duplicate names, so they must differ.
    """
    if spec.name:
        return spec.name
    if spec.quantity == "heat_capacity":
        return "ns_thermodynamics"
    return f"ns_{spec.quantity}"


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
        name=_thermo_name(spec),
        **c,
    )


register_panel("ns_thermodynamics", ThermodynamicsSpec, _build_thermodynamics)
