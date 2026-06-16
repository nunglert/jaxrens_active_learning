"""Project panel: nested-sampling thermodynamics heatmap.

Equips alviz's abstract ``HeatmapPanel`` (which owns the (P, T) pcolormesh
rendering + ``HeatmapData`` contract, but ships no reader) with the
nested-sampling computation. The numbers come from jaxrens'
``Monitor.from_directory`` -- it reads a run's ``<prefix>.energies`` log and
``<prefix>.final.checkpoint.h5`` (dead + live energies) and evaluates
``heat_capacity`` / ``expectation`` over a temperature grid. That postprocess
path is pure NumPy/SciPy, so building this dashboard does not pull in JAX; the
``jaxrens`` import is deferred to ``compute_data`` and only the actual NS
postprocessing needs it.

A single run gives one pressure row -> a degenerate (1, n_T) heatmap that
still reads as a C_v(T) / <V>(T) curve; pass several run dirs (e.g. an NPT
pressure sweep) to get a real (n_P, n_T) grid.
"""

import numpy as np
from alviz.panels import HeatmapData, HeatmapPanel

_QUANTITIES = ("heat_capacity", "volume")


class NSThermodynamicsHeatmap(HeatmapPanel):
    """Heat-capacity or mean-volume vs temperature from jaxrens NS runs.

    Args:
        result_dirs: One or more directories each holding a jaxrens
            ``<prefix>.energies`` + ``<prefix>.final.checkpoint.h5`` (one row
            of the heatmap per dir). Empty -> nothing to plot.
        quantity: ``"heat_capacity"`` or ``"volume"``.
        temperatures: Explicit temperature array, or None to build
            ``n_T`` points spanning ``tlim``.
        tlim: (T_min, T_max) when ``temperatures`` is None.
        n_T: Number of temperature points when ``temperatures`` is None.
        k_B: Boltzmann constant in the energy units of the stored energies
            (1.0 for reduced units; 8.617e-5 for eV with T in Kelvin).
        prefix: jaxrens ``out_file_prefix`` of the run files.
        pressures: Optional row labels (one per ``result_dirs`` entry);
            defaults to row indices.
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

    def _load_monitor(self, result_dir):
        """Build a jaxrens Monitor for one run dir (seam for testing)."""
        from jaxrens.postprocess import Monitor

        return Monitor.from_directory(result_dir, prefix=self._prefix)

    def compute_data(self, shared_ressources: dict) -> HeatmapData:
        if not self._result_dirs:
            raise RuntimeError(
                "NSThermodynamicsHeatmap: no NS result directories given."
            )

        T = self._temperatures
        rows = []
        for result_dir in self._result_dirs:
            mon = self._load_monitor(result_dir)
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
            t=T.tolist(),
            p=p,
            obs=rows,
            quantity=self._quantity,
        )
