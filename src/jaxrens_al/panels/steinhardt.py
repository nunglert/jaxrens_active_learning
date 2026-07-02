"""Project panels: Steinhardt order parameters, computed once and reused.

The (expensive, remote) Steinhardt computation lives in the run-side
:class:`jaxrens_al.processors.steinhardt.SteinhardtProcessor` (a
``remote_handler`` trajectory-processing step), run via
``remote_handler.processing.RemoteTrajectoryProcessor``; it writes an annotated
trajectory (per-atom ``q<l>`` / ``w<l>`` arrays + per-frame ``q<l>_mean`` /
``w<l>_mean`` in ``atoms.info``). These panels are pure **readers** of that
artifact (ase + numpy, no cluster, no ``remote_handler`` import).

Each panel gets its data from one of two sources:

  * ``annotated_file=`` -- an already-annotated extxyz (pure read); or
  * ``configs=`` + Steinhardt params (``r_cut`` ...) -- the *raw* pool to
    annotate. The panel then declares a :class:`~jaxrens_al.panels.processing.
    ProcessingNeed` (plain data, no cluster), and the run-side
    :func:`jaxrens_al.processing_resolver.resolve_processing` -- called once
    over the whole dashboard -- runs a single processing job and hands the
    annotated path back via ``receive_annotations``. Identical needs across
    panels deduplicate to one step, so the panels below share one job.

Either way the panels themselves stay pure readers -- ``remote_handler`` enters
only through the resolver, never here.

Panels:
  * :class:`SteinhardtDistributionPanel` -- per-atom ``q_l`` / ``w_l``
    distribution (joint scatter for two l-values, else per-l histograms);
  * :class:`SteinhardtHeatmapPanel` -- expectation ``<q_l>`` (or ``<w_l>``) as
    a heatmap over a 2D grid of two per-config keys (e.g. volume vs energy);
  * :class:`SteinhardtExpectationHeatmap` -- the *nested-sampling* expectation
    ``<q_l>(T)`` per replica (like the heat-capacity heatmap), combining a
    jaxrens ``Monitor`` with the annotated pool.

The first two are pure ase+numpy readers. :class:`SteinhardtExpectationHeatmap`
additionally uses the jaxrens ``Monitor`` (deferred behind a ``_load_monitors``
seam, as in :mod:`jaxrens_al.panels.thermodynamics`), so building the dashboard
still pulls in no JAX -- only the actual NS postprocessing does.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from alviz import Entry, JsonDict
from alviz.panels import HeatmapData, HeatmapPanel, heatmap_pcolormesh

from jaxrens_al.panels.processing import ProcessingConsumer, ProcessingNeed

_PARAMS = ("q", "w")


def _read_frames(annotated) -> list:
    import ase.io

    return ase.io.read(str(annotated), index=":")


def _require_array(at, key, annotated):
    if key not in at.arrays:
        raise ValueError(
            f"Annotated trajectory {annotated} is missing per-atom array "
            f"'{key}'. Did the Steinhardt processor write it?"
        )
    return np.asarray(at.arrays[key], float)


class _SteinhardtSource(ProcessingConsumer):
    """Shared source resolution for the reader panels (mix in before Entry).

    Source is either a pre-annotated ``annotated_file`` (pure read) or a raw
    ``configs`` pool that the run-side resolver annotates (declared as a
    :class:`ProcessingNeed`, delivered via :meth:`receive_annotations`).
    """

    def _init_source(
        self, annotated_file, configs, l_values, param, r_cut, r_cut_min
    ):
        if (annotated_file is None) == (configs is None):
            raise ValueError(
                "Steinhardt panel needs exactly one of annotated_file= or "
                "configs=."
            )
        if configs is not None and r_cut is None:
            raise ValueError(
                "Steinhardt panel with configs= needs r_cut= (the Steinhardt "
                "neighbour cutoff) so it can declare its processing need."
            )
        # ``_annotated_file`` may start None and be filled by the resolver.
        self._annotated_file = annotated_file
        self._configs = None if configs is None else str(configs)
        self._r_cut = r_cut
        self._r_cut_min = r_cut_min
        self._l_values = [int(l) for l in l_values]
        if not self._l_values:
            raise ValueError("Steinhardt panel needs at least one l value.")
        if param not in _PARAMS:
            raise ValueError(f"param must be one of {_PARAMS}; got {param!r}.")
        self._param = param

    def _annotated(self) -> Path:
        if self._annotated_file is None:
            raise RuntimeError(
                "Steinhardt panel has no annotated file yet: pass "
                "annotated_file=, or run resolve_processing() over the "
                "dashboard (configs= declares the need)."
            )
        return Path(self._annotated_file)

    def processing_needs(self) -> list:
        if self._configs is None or self._annotated_file is not None:
            return []
        return [
            ProcessingNeed.make(
                (self._configs,),
                "steinhardt",
                {
                    "l_values": tuple(self._l_values),
                    "r_cut": self._r_cut,
                    "r_cut_min": self._r_cut_min,
                },
            )
        ]

    def receive_annotations(self, mapping: dict) -> None:
        if self._configs is not None:
            self._annotated_file = mapping[self._configs]


class SteinhardtDistributionPanel(_SteinhardtSource, Entry):
    """Per-atom ``q_l`` / ``w_l`` distribution of a set of configs.

    With exactly two ``l_values`` (e.g. ``(4, 6)``) draws the classic joint
    ``X_{l0}``-vs-``X_{l1}`` scatter coloured by species (``X`` = ``param``);
    otherwise one histogram per ``l``.

    Args:
        annotated_file: A Steinhardt-annotated extxyz (pure read); or
        configs: the raw pool to annotate (the resolver fills the annotated
            path). Give exactly one of the two.
        l_values: Steinhardt degrees (default 4, 6).
        param: ``"q"`` (default) or ``"w"``.
        r_cut, r_cut_min: neighbour cutoffs -- required with ``configs=`` (they
            define the processing need).
    """

    name = "steinhardt"

    def __init__(
        self,
        *args,
        annotated_file: Optional[os.PathLike] = None,
        configs: Optional[os.PathLike] = None,
        l_values: Sequence[int] = (4, 6),
        param: str = "q",
        r_cut: Optional[float] = None,
        r_cut_min: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_source(
            annotated_file, configs, l_values, param, r_cut, r_cut_min
        )
        self._joint = len(self._l_values) == 2
        self.colspan = 1 if self._joint else len(self._l_values)

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        annotated = self._annotated()
        frames = _read_frames(annotated)
        per_l = {p: {l: [] for l in self._l_values} for p in _PARAMS}
        species: list[str] = []
        for at in frames:
            for l in self._l_values:
                for p in _PARAMS:
                    per_l[p][l].extend(
                        _require_array(at, f"{p}{l}", annotated).tolist()
                    )
            species.extend(at.get_chemical_symbols())
        out = {
            p: {str(l): per_l[p][l] for l in self._l_values} for p in _PARAMS
        }
        return JsonDict(
            l_values=self._l_values,
            param=self._param,
            q=out["q"],
            w=out["w"],
            species=species,
        )

    def allocate_panels(self, fig):
        self._fig = fig
        if self._joint:
            self.gs = self.subspec.subgridspec(1, 1)
            self.ax = fig.add_subplot(self.gs[0, 0])
            self.axes = [self.ax]
        else:
            self.gs = self.subspec.subgridspec(
                1, len(self._l_values), wspace=0.35
            )
            self.axes = [
                fig.add_subplot(self.gs[0, i])
                for i in range(len(self._l_values))
            ]

    def plot_data(self):
        import matplotlib.pyplot as plt

        param = self.data.get("param", "q")
        vals_by_l = self.data[param]
        l_values = self.data["l_values"]
        species = np.asarray(self.data["species"])

        if self._joint:
            la, lb = l_values
            xa = np.asarray(vals_by_l[str(la)], float)
            xb = np.asarray(vals_by_l[str(lb)], float)
            ax = self.axes[0]
            if xa.size:
                for sym in sorted(set(species.tolist())):
                    m = species == sym
                    ax.scatter(xa[m], xb[m], s=6, alpha=0.5, label=sym)
                ax.legend(fontsize="x-small", loc="best")
            ax.set_xlabel(rf"${param}_{{{la}}}$")
            ax.set_ylabel(rf"${param}_{{{lb}}}$")
            ax.set_title("Steinhardt order parameters")
            return

        cmap = plt.get_cmap("tab10")
        for i, l in enumerate(l_values):
            ax = self.axes[i]
            vals = np.asarray(vals_by_l[str(l)], float)
            if vals.size:
                ax.hist(vals, bins=40, color=cmap(i % 10), alpha=0.8)
            ax.set_xlabel(rf"${param}_{{{l}}}$")
            if i == 0:
                ax.set_ylabel("count")
        self.axes[0].set_title("Steinhardt order parameters")


# Back-compat alias (the panel was introduced as ``SteinhardtPanel``).
SteinhardtPanel = SteinhardtDistributionPanel


def _config_axis(at, key: str) -> float:
    """A per-config scalar for a heatmap axis: ``volume`` / ``volume_per_atom``
    are computed; anything else is read from ``atoms.info``.
    """
    if key in ("volume", "vol"):
        return float(at.get_volume())
    if key in ("volume_per_atom", "vol_per_atom"):
        return float(at.get_volume() / len(at))
    if key in at.info:
        return float(at.info[key])
    raise ValueError(
        f"Heatmap axis key {key!r} is neither 'volume[_per_atom]' nor an "
        f"atoms.info key present on the frame (have: {sorted(at.info)})."
    )


class SteinhardtHeatmapPanel(_SteinhardtSource, HeatmapPanel):
    """Expectation ``<q_l>`` (or ``<w_l>``) over a 2D grid of two config keys.

    Each config contributes its per-frame mean ``<param_l>`` (the CLI's
    ``param<l>_mean``, or the mean of the per-atom array as a fallback).
    Configs are binned by ``(x_key, y_key)`` and each cell shows the *mean* of
    per-config expectations landing in it -- i.e. ``E[param_l | x, y]``. Empty
    cells render transparent (NaN).

    Reads the *same* annotated artifact as
    :class:`SteinhardtDistributionPanel`; pass both panels one output file to
    compute once.

    Args:
        annotated_file: annotated extxyz to read; or ``configs``: the raw pool
            (resolver-annotated). Give exactly one (see module doc).
        l: Single Steinhardt degree to map (default 6).
        param: ``"q"`` (default) or ``"w"``.
        r_cut, r_cut_min: neighbour cutoffs -- required with ``configs=``.
        x_key, y_key: per-config axis keys -- ``"volume"`` /
            ``"volume_per_atom"`` or any ``atoms.info`` key (default volume per
            atom vs ``ns_energy``).
        n_bins: bins per axis (int, or ``(nx, ny)``).
    """

    name = "steinhardt_heatmap"

    def __init__(
        self,
        *args,
        annotated_file: Optional[os.PathLike] = None,
        configs: Optional[os.PathLike] = None,
        l: int = 6,
        param: str = "q",
        r_cut: Optional[float] = None,
        r_cut_min: float = 0.0,
        x_key: str = "volume_per_atom",
        y_key: str = "ns_energy",
        n_bins=24,
        plot_kwargs: dict = None,
        **kwargs,
    ):
        plot_kwargs = dict(plot_kwargs or {})
        plot_kwargs.setdefault("T_label", x_key)
        plot_kwargs.setdefault("P_label", y_key)
        super().__init__(*args, plot_kwargs=plot_kwargs, **kwargs)
        self._init_source(
            annotated_file, configs, [l], param, r_cut, r_cut_min
        )
        self._l = int(l)
        self._x_key = x_key
        self._y_key = y_key
        nx, ny = (n_bins, n_bins) if np.isscalar(n_bins) else n_bins
        self._nx, self._ny = int(nx), int(ny)

    def _config_value(self, at, annotated) -> float:
        """Per-config expectation ``<param_l>``: CLI mean, else computed."""
        key = f"{self._param}{self._l}_mean"
        if key in at.info:
            return float(at.info[key])
        arr = _require_array(at, f"{self._param}{self._l}", annotated)
        return float(arr.mean())

    def compute_data(self, shared_ressources: dict) -> HeatmapData:
        annotated = self._annotated()
        frames = _read_frames(annotated)
        if not frames:
            raise RuntimeError(
                f"SteinhardtHeatmapPanel: {annotated} has no frames."
            )

        x = np.array([_config_axis(at, self._x_key) for at in frames], float)
        y = np.array([_config_axis(at, self._y_key) for at in frames], float)
        v = np.array(
            [self._config_value(at, annotated) for at in frames], float
        )

        counts, xe, ye = np.histogram2d(x, y, bins=(self._nx, self._ny))
        sums, _, _ = np.histogram2d(x, y, bins=(xe, ye), weights=v)
        with np.errstate(invalid="ignore"):
            mean = np.where(counts > 0, sums / counts, np.nan)  # (nx, ny)

        x_centers = 0.5 * (xe[:-1] + xe[1:])
        y_centers = 0.5 * (ye[:-1] + ye[1:])
        return HeatmapData(
            t=x_centers.tolist(),
            p=y_centers.tolist(),
            obs=mean.T.tolist(),  # rows = y bins (p), cols = x bins (t)
            quantity=f"<{self._param}{self._l}>",
        )


class SteinhardtExpectationHeatmap(ProcessingConsumer, HeatmapPanel):
    """Nested-sampling expectation ``<q_l>(T)`` per replica (like the C_v map).

    The NS analogue of :class:`SteinhardtHeatmapPanel`: rather than binning
    per-config means over two arbitrary keys, it reweights a Steinhardt
    observable onto a temperature grid with each replica's jaxrens ``Monitor``
    -- the same machinery as
    :class:`~jaxrens_al.panels.thermodynamics.NSThermodynamicsHeatmap`, but
    with ``<q_l>`` / ``<w_l>`` as the observable instead of ``C_v`` / ``<V>``.

    The panel is self-sufficient about its inputs: from each result dir it
    discovers the per-replica ``<prefix>.run*.traj.extxyz`` (the same dir +
    prefix it loads Monitors from), in replica order. Each such trajectory *is*
    one replica; it declares a :class:`ProcessingNeed` on each (with ``r_cut``)
    and the run-side resolver annotates them, delivering the annotated paths
    back. Given ``r_cut=None`` the discovered trajectories are assumed already
    annotated and read directly.

    Per replica: the per-config ``q<l>_mean`` and dead-point index
    ``info['iter']``. A jaxrens trajectory is usually a regular *slice* of the
    dead points (``output.traj_interval``), so each retained value is held
    across its slice window to reconstruct the per-dead-point observable (a
    good approximation -- consecutive dead points barely differ); a
    full-resolution trajectory is used directly. ``mon.expectation(obs, T)``
    then gives
    ``<q_l>`` over the temperature grid with the full NS weights.

    Args:
        result_dirs: NS output dirs (per-replica ``<prefix>.run*.energies`` +
            ``.run*.traj.extxyz`` + combined checkpoint); every replica across
            every dir is one heatmap row.
        prefix: jaxrens ``out_file_prefix`` of the run files.
        r_cut: Steinhardt neighbour cutoff (Angstrom). Given -> the panel
            declares a processing need on each trajectory (the resolver
            annotates them). ``None`` -> the trajectories are read as already
            annotated.
        r_cut_min: Lower Steinhardt neighbour cutoff (default 0).
        l: Steinhardt degree (default 6).
        param: ``"q"`` (default) or ``"w"``.
        temperatures: Explicit temperature array, or None to build ``n_T``
            points spanning ``tlim``.
        tlim: (T_min, T_max) when ``temperatures`` is None.
        n_T: Number of temperature points when ``temperatures`` is None.
        k_B: Boltzmann constant in the energy units of the stored energies.
        pressures: Optional row labels (one per replica row); default row
            indices.
        swap_axes: If True, render pressure on x and temperature on y.
    """

    name = "steinhardt_expectation"

    def __init__(
        self,
        result_dirs,
        *args,
        prefix: str = "ns",
        r_cut: Optional[float] = None,
        r_cut_min: float = 0.0,
        l: int = 6,
        param: str = "q",
        temperatures=None,
        tlim=(0.05, 2.0),
        n_T: int = 50,
        k_B: float = 1.0,
        pressures=None,
        swap_axes: bool = False,
        plot_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(*args, plot_kwargs=plot_kwargs, **kwargs)
        if param not in _PARAMS:
            raise ValueError(f"param must be one of {_PARAMS}; got {param!r}.")
        self._result_dirs = [str(p) for p in result_dirs]
        self._prefix = prefix
        self._r_cut = r_cut
        self._r_cut_min = r_cut_min
        self._l = int(l)
        self._param = param
        # {raw trajectory -> annotated trajectory}, filled by the resolver.
        self._annotations: dict = {}
        if temperatures is not None:
            self._temperatures = np.asarray(temperatures, dtype=float)
        else:
            self._temperatures = np.linspace(tlim[0], tlim[1], n_T)
        self._k_B = k_B
        self._pressures = pressures
        self._swap_axes = swap_axes

    def _traj_files(self, result_dir) -> list:
        """The per-replica trajectories in ``result_dir``, in replica order.

        Globs ``<prefix>.run*.traj.extxyz`` (which excludes the ``.snap.``
        snapshot files); sorted order matches the Monitors' ``runNN`` order.
        """
        return [
            str(p)
            for p in sorted(
                Path(result_dir).glob(f"{self._prefix}.run*.traj.extxyz")
            )
        ]

    def _load_monitors(self, result_dir):
        """Monitors for one NS output dir, one per replica (test seam).

        Same seam/loader as
        :meth:`~jaxrens_al.panels.thermodynamics.NSThermodynamicsHeatmap.
        _load_monitors`; a single-run output is the one-replica edge case.
        """
        from jaxrens.postprocess import MonitorCollection

        return list(
            MonitorCollection.from_multi_run_directory(
                result_dir, prefix=self._prefix
            )
        )

    def processing_needs(self) -> list:
        if self._r_cut is None:
            return []  # trajectories read as already annotated
        params = {
            "l_values": (self._l,),
            "r_cut": self._r_cut,
            "r_cut_min": self._r_cut_min,
        }
        return [
            ProcessingNeed.make((traj,), "steinhardt", params)
            for result_dir in self._result_dirs
            for traj in self._traj_files(result_dir)
        ]

    def receive_annotations(self, mapping: dict) -> None:
        self._annotations = dict(mapping)

    def _read_observable(self, src):
        """Per-config ``<param_l>`` + dead-point index for one replica traj.

        Returns ``(values, indices)``; ``indices`` (from ``info['iter']``) is
        None if no frame carries it. Each value is the per-frame mean
        ``param<l>_mean`` (falling back to the mean of the per-atom array).
        """
        frames = _read_frames(src)
        key = f"{self._param}{self._l}_mean"
        arr_key = f"{self._param}{self._l}"
        vals: list = []
        idxs: list = []
        have_iter = True
        for at in frames:
            if key in at.info:
                vals.append(float(at.info[key]))
            elif arr_key in at.arrays:
                arr = np.asarray(at.arrays[arr_key], float)
                vals.append(float(np.mean(arr)))
            else:
                raise ValueError(
                    f"Annotated trajectory {src} is missing '{key}' and "
                    f"per-atom '{arr_key}'. Did the Steinhardt processor "
                    f"write it?"
                )
            it = at.info.get("iter")
            if it is None:
                have_iter = False
            idxs.append(it)
        return (
            np.asarray(vals, float),
            np.asarray(idxs, int) if have_iter else None,
        )

    def _to_dead_point_observable(self, vals, idx, n_dead, replica):
        """Reconstruct a per-dead-point observable of length ``n_dead``.

        A jaxrens trajectory is usually a regular *slice* of the dead points
        (``output.traj_interval``), so ``vals`` is shorter than ``n_dead``. NS
        reweighting needs a value at every dead point, but consecutive dead
        points barely differ, so we hold each retained value across its slice
        window (piecewise constant, keyed by ``iter``) -- a good approximation
        that lets ``Monitor.expectation`` reweight with the full NS weights.
        A full-resolution trajectory (``len(vals) == n_dead``) is used as-is.
        """
        if len(vals) == n_dead:
            return vals
        if idx is None:
            raise ValueError(
                f"SteinhardtExpectationHeatmap: replica {replica} has "
                f"{len(vals)} annotated configs but the NS run has {n_dead} "
                f"dead points, and the pool carries no 'iter' index to align "
                f"the subsampled trajectory. Re-annotate from a jaxrens "
                f"trajectory (its frames carry 'iter')."
            )
        if idx.min() < 0 or idx.max() >= n_dead:
            raise ValueError(
                f"SteinhardtExpectationHeatmap: replica {replica} dead-point "
                f"indices ('iter') span [{idx.min()}, {idx.max()}] but the NS "
                f"run has {n_dead} dead points -- pool and run mismatched."
            )
        # For each dead point, the most recent retained config's value.
        pos = np.searchsorted(idx, np.arange(n_dead), side="right") - 1
        pos = np.clip(pos, 0, len(vals) - 1)
        return np.asarray(vals, float)[pos]

    def compute_data(self, shared_ressources: dict) -> HeatmapData:
        if not self._result_dirs:
            raise RuntimeError(
                "SteinhardtExpectationHeatmap: no NS result directories given."
            )

        T = self._temperatures
        rows = []
        for result_dir in self._result_dirs:
            trajs = self._traj_files(result_dir)
            if not trajs:
                raise RuntimeError(
                    f"SteinhardtExpectationHeatmap: no "
                    f"'{self._prefix}.run*.traj.extxyz' under {result_dir}."
                )
            monitors = self._load_monitors(result_dir)
            if len(trajs) != len(monitors):
                raise ValueError(
                    f"SteinhardtExpectationHeatmap: {len(trajs)} trajectories "
                    f"but {len(monitors)} replicas in {result_dir}."
                )
            for i, (traj, mon) in enumerate(zip(trajs, monitors)):
                src = self._annotations.get(traj)
                if src is None:
                    if self._r_cut is not None:
                        raise RuntimeError(
                            "SteinhardtExpectationHeatmap declared a "
                            "processing need but no annotation was delivered "
                            "-- run resolve_processing() over the dashboard "
                            "before rendering."
                        )
                    src = traj  # r_cut None -> trajectory already annotated
                vals, idx = self._read_observable(src)
                n_dead = getattr(mon, "n_dead", len(vals))
                obs = self._to_dead_point_observable(vals, idx, n_dead, i)
                row = mon.expectation(obs, T, k_B=self._k_B)
                rows.append(np.asarray(row, dtype=float).tolist())

        if self._pressures is not None:
            p = [float(x) for x in self._pressures]
        else:
            p = [float(i) for i in range(len(rows))]

        return HeatmapData(
            t=T.tolist(),
            p=p,
            obs=rows,
            quantity=f"<{self._param}{self._l}>",
        )

    def plot_data(self):
        # Default orientation (T on x, P on y) is alviz's; defer to it.
        if not self._swap_axes:
            return super().plot_data()

        # P on x, T on y: transpose the grid and swap the axis labels, still
        # rendering through alviz's helper (colorbar/scaling/cmap reused).
        d = self.data
        kw = dict(self._plot_kwargs)
        if d.quantity and "label" not in kw:
            kw["label"] = d.quantity
        t_label = kw.pop("T_label", r"$T$")
        p_label = kw.pop("P_label", r"$\log P$")
        obs_t = np.asarray(d.obs, dtype=float).T  # (len(t), len(p))
        heatmap_pcolormesh(
            self.ax, d.p, d.t, obs_t, T_label=p_label, P_label=t_label, **kw
        )
        if d.quantity:
            self.ax.set_title(d.quantity)
