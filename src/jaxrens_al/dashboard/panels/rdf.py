"""Project panel: the nested-sampling expectation of the RDF, ``<g(r)>(T)``.

The (remote) producer is ``remote_handler``'s built-in ``rdf`` processor, which
annotates every frame with ``r_rdf`` (bin centres) + ``rdf_total`` (total
``g(r)``). As with the Steinhardt panels, this module is a pure **reader**: it
declares a :class:`~jaxrens_al.dashboard.panels.processing.ProcessingNeed` per
replica trajectory and the run-side resolver runs the one annotation job.

Per replica the annotated per-frame RDFs are reweighted onto the temperature
grid with that replica's jaxrens ``Monitor`` (see
:mod:`jaxrens_al.dashboard.panels.ns_expectation`), giving the full
``<g(r)>(T)`` surface -- ``n_T x n_bins`` numbers per replica, all of which are
cached in the panel's JSON (``r`` + ``rdf``) so the raw curves stay available
for ad-hoc plotting.

**Projecting the curve back to a scalar.** ``<g(r)>(T)`` is vector-valued, so
it cannot go on a (replica, T) heatmap directly. The panel therefore reduces
each temperature's curve to one number with a selectable ``reduction``:

``peak_height``
    ``max_r <g(r)>`` -- the first-peak height. The classic structural-order
    indicator (Hansen-Verlet: a liquid freezes near ``g_max ~ 2.85``), and the
    default.
``peak_position``
    the ``r`` of that peak -- the mean nearest-neighbour distance, which grows
    with thermal expansion.
``order``
    ``tau = int |<g(r)> - 1| dr / dr_window`` -- the translational order
    parameter: ~0 for an ideal gas, larger the more structured the fluid/solid.
    A single dimensionless number for "how ordered is it at this T".
``coordination``
    ``N(r) = 4 pi rho int r^2 <g(r)> dr`` over the window -- the coordination
    number. ``rho = N_atoms / <V>(T)`` uses the *same* NS weights (the cell
    volume is reweighted alongside the RDF), so it is consistent at every T.
    Its window defaults to the first minimum after the first peak.
``mean``
    plain mean of ``<g(r)>`` over ``[r_min, r_max]`` -- the escape hatch for
    watching one shell.

Several reductions can coexist: the panel is repeatable and its entries are
named ``rdf_<reduction>_rc<cutoff>``, and they share one annotation job (the
processing need only depends on ``cutoff``/``nbins``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from alviz import PanelSpec, register_panel
from alviz.panels import HeatmapData, HeatmapPanel

from jaxrens_al.dashboard.panels.ns_expectation import (
    SwapAxesHeatmapMixin,
    hold_to_dead_points,
    load_monitors,
    replica_traj_files,
    reweight,
)
from jaxrens_al.dashboard.panels.processing import (
    ProcessingConsumer,
    ProcessingNeed,
)

# The keys ``remote_handler``'s ``rdf`` processor writes into ``atoms.info``.
R_KEY = "r_rdf"
G_KEY = "rdf_total"

_REDUCTIONS = (
    "peak_height",
    "peak_position",
    "order",
    "coordination",
    "mean",
)

# ``np.trapz`` was renamed in numpy 2.0; support either.
_trapz = getattr(np, "trapezoid", None) or np.trapz


class RDFExpectationData(HeatmapData):
    """A (replica, T) grid of the reduced scalar **plus** the full RDFs.

    ``obs`` is the reduction that gets plotted; ``rdf`` keeps the whole
    ``<g(r)>(T)`` surface per replica (shape ``(len(p), len(t), len(r))``) so
    the cached JSON holds the raw observable, not just its projection.
    """

    r: list
    rdf: list
    reduction: str
    r_window: list


def _smooth(g, width: int = 5):
    """Boxcar-smoothed copy of ``g`` (peak/minimum detection only).

    The window is capped at an eighth of the profile, so a coarse RDF is left
    alone rather than being smoothed into a featureless line.
    """
    width = min(width, max(1, g.size // 8))
    if width < 2:
        return g
    kernel = np.ones(width) / width
    return np.convolve(g, kernel, mode="same")


def _first_minimum(r, g):
    """``r`` of the first local minimum after the first peak, or None.

    Detected on a lightly smoothed profile so histogram noise does not create
    a spurious minimum two bins past the peak.
    """
    s = _smooth(np.asarray(g, dtype=float))
    i_peak = int(np.argmax(s))
    for i in range(i_peak + 1, len(s) - 1):
        if s[i] <= s[i - 1] and s[i] <= s[i + 1]:
            return float(r[i])
    return None


def _reduce(grid, r, mask, reduction: str, density=None):
    """Project ``<g(r)>(T)`` (n_T, n_bins) onto one scalar per temperature."""
    rw = r[mask]
    gw = grid[:, mask]
    if reduction == "peak_height":
        return gw.max(axis=1)
    if reduction == "peak_position":
        return rw[gw.argmax(axis=1)]
    if reduction == "mean":
        return gw.mean(axis=1)
    if reduction == "order":
        span = rw[-1] - rw[0]
        integral = _trapz(np.abs(gw - 1.0), rw, axis=1)
        return integral / span if span > 0 else integral
    if reduction == "coordination":
        integral = _trapz(gw * rw[None, :] ** 2, rw, axis=1)
        return 4.0 * np.pi * np.asarray(density, dtype=float) * integral
    raise ValueError(
        f"reduction must be one of {_REDUCTIONS}; got {reduction!r}."
    )


def _quantity_label(reduction: str, window) -> str:
    lo, hi = window
    if reduction == "peak_height":
        return r"$\max_r \langle g(r) \rangle$"
    if reduction == "peak_position":
        return r"$r_\mathrm{peak}$ ($\mathrm{\AA}$)"
    if reduction == "order":
        return r"$\tau = \int |\langle g \rangle - 1|\,dr / \Delta r$"
    if reduction == "coordination":
        return rf"$N_\mathrm{{coord}}(r < {hi:.2f}\,\mathrm{{\AA}})$"
    return rf"$\langle g \rangle_{{[{lo:.2f}, {hi:.2f}]}}$"


class NSRDFExpectationHeatmap(
    ProcessingConsumer, SwapAxesHeatmapMixin, HeatmapPanel
):
    """Nested-sampling ``<g(r)>(T)`` per replica, reduced to a scalar map.

    Discovers each result dir's per-replica ``<prefix>.run*.traj.extxyz`` (the
    same dir + prefix it loads Monitors from), declares an ``rdf``
    :class:`ProcessingNeed` on each (given ``cutoff``), reweights the annotated
    per-frame RDFs onto the temperature grid, and renders one heatmap row per
    replica of the chosen scalar ``reduction`` (see the module docstring). The
    full ``<g(r)>(T)`` surface is kept in the panel's cached JSON.

    Args:
        result_dirs: NS output dirs; every replica across every dir is a row.
        prefix: jaxrens ``out_file_prefix`` of the run files.
        cutoff: RDF cutoff (Angstrom) -- the panel declares its processing need
            with it. ``None`` -> the discovered trajectories are read as
            already annotated.
        nbins: RDF histogram bins (part of the processing need).
        reduction: How each temperature's curve becomes one number; one of
            ``peak_height`` (default), ``peak_position``, ``order``,
            ``coordination``, ``mean``.
        r_min, r_max: ``r`` window the reduction is evaluated over. ``r_max``
            None -> the full range, except for ``coordination``, which falls
            back to the first minimum after the first peak (of the lowest-T
            curve of the first replica, so every row shares one window).
        temperatures, tlim, n_T: the temperature grid.
        k_B: Boltzmann constant in the energy units of the stored energies.
        pressures: Optional row labels (one per replica); default row indices.
        swap_axes: If True, render pressure on x and temperature on y.
    """

    name = "rdf_expectation"
    data_class = RDFExpectationData

    def __init__(
        self,
        result_dirs,
        *args,
        prefix: str = "ns",
        cutoff: Optional[float] = None,
        nbins: int = 120,
        reduction: str = "peak_height",
        r_min: float = 0.0,
        r_max: Optional[float] = None,
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
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {_REDUCTIONS}; got {reduction!r}."
            )
        self._result_dirs = [str(p) for p in result_dirs]
        self._prefix = prefix
        self._cutoff = cutoff
        self._nbins = int(nbins)
        self._reduction = reduction
        self._r_min = float(r_min)
        self._r_max = None if r_max is None else float(r_max)
        if temperatures is not None:
            self._temperatures = np.asarray(temperatures, dtype=float)
        else:
            self._temperatures = np.linspace(tlim[0], tlim[1], n_T)
        self._k_B = k_B
        self._pressures = pressures
        self._swap_axes = swap_axes
        # {raw trajectory -> annotated trajectory}, filled by the resolver.
        self._annotations: dict = {}

    # -- inputs ------------------------------------------------------------
    def _traj_files(self, result_dir) -> list:
        return replica_traj_files(result_dir, self._prefix)

    def _load_monitors(self, result_dir):
        """Monitors for one NS output dir, one per replica (test seam)."""
        return load_monitors(result_dir, self._prefix)

    def processing_needs(self) -> list:
        if self._cutoff is None:
            return []  # trajectories read as already annotated
        params = {"cutoff": self._cutoff, "nbins": self._nbins}
        return [
            ProcessingNeed.make((traj,), "rdf", params)
            for result_dir in self._result_dirs
            for traj in self._traj_files(result_dir)
        ]

    def receive_annotations(self, mapping: dict) -> None:
        self._annotations = dict(mapping)

    # -- computation -------------------------------------------------------
    def _read_rdfs(self, src):
        """``(g, r, volumes, n_atoms, iter)`` for one replica trajectory.

        ``g`` is ``(n_frames, n_bins)``; ``volumes`` is the per-frame cell
        volume (reweighted alongside the RDF so ``coordination`` gets a
        thermally consistent density). ``iter`` is None when no frame carries
        the dead-point index.
        """
        import ase.io

        frames = ase.io.read(str(src), index=":")
        if not frames:
            raise ValueError(f"Annotated trajectory {src} has no frames.")

        g, vols, idxs = [], [], []
        r = None
        have_iter = True
        for at in frames:
            if G_KEY not in at.info or R_KEY not in at.info:
                raise ValueError(
                    f"Annotated trajectory {src} is missing '{G_KEY}'/"
                    f"'{R_KEY}' in atoms.info. Did the rdf processor write "
                    f"it?"
                )
            if r is None:
                r = np.asarray(at.info[R_KEY], dtype=float)
            g.append(np.asarray(at.info[G_KEY], dtype=float))
            vols.append(float(at.get_volume()))
            it = at.info.get("iter")
            if it is None:
                have_iter = False
            idxs.append(it)

        g = np.asarray(g, dtype=float)
        if g.shape[1] != r.size:
            raise ValueError(
                f"Annotated trajectory {src}: '{G_KEY}' has {g.shape[1]} bins "
                f"but '{R_KEY}' has {r.size} -- inconsistent annotation."
            )
        return (
            g,
            r,
            np.asarray(vols, dtype=float),
            len(frames[0]),
            np.asarray(idxs, dtype=int) if have_iter else None,
        )

    def _replica_source(self, traj):
        src = self._annotations.get(traj)
        if src is not None:
            return src
        if self._cutoff is not None:
            raise RuntimeError(
                "NSRDFExpectationHeatmap declared a processing need but no "
                "annotation was delivered -- run resolve_processing() over "
                "the dashboard before rendering."
            )
        return traj  # cutoff None -> trajectory already annotated

    def _window(self, r, first_row):
        """The ``(lo, hi)`` r-window every row's reduction is evaluated over.

        Explicit ``r_min``/``r_max`` win. Otherwise the window runs to the end
        of the RDF, except for ``coordination``, whose shell must stop at the
        first minimum after the first peak -- taken from the lowest-temperature
        curve of the first replica so all rows share one definition (and the
        resulting numbers stay comparable).
        """
        hi = self._r_max
        if hi is None and self._reduction == "coordination":
            hi = _first_minimum(r, first_row[0])
        if hi is None:
            hi = float(r[-1])
        return float(self._r_min), float(hi)

    def compute_data(self, shared_ressources: dict) -> RDFExpectationData:
        if not self._result_dirs:
            raise RuntimeError(
                "NSRDFExpectationHeatmap: no NS result directories given."
            )

        T = self._temperatures
        grids, densities = [], []
        r = None
        for result_dir in self._result_dirs:
            trajs = self._traj_files(result_dir)
            if not trajs:
                raise RuntimeError(
                    f"NSRDFExpectationHeatmap: no "
                    f"'{self._prefix}.run*.traj.extxyz' under {result_dir}."
                )
            monitors = self._load_monitors(result_dir)
            if len(trajs) != len(monitors):
                raise ValueError(
                    f"NSRDFExpectationHeatmap: {len(trajs)} trajectories but "
                    f"{len(monitors)} replicas in {result_dir}."
                )
            for i, (traj, mon) in enumerate(zip(trajs, monitors)):
                g, r_i, vols, n_atoms, idx = self._read_rdfs(
                    self._replica_source(traj)
                )
                if r is None:
                    r = r_i
                elif r_i.size != r.size:
                    raise ValueError(
                        "NSRDFExpectationHeatmap: replicas were annotated on "
                        f"different r grids ({r.size} vs {r_i.size} bins) -- "
                        "re-annotate them with one (cutoff, nbins)."
                    )
                n_dead = getattr(mon, "n_dead", len(g))
                panel = type(self).__name__
                # Reweight the volume in the same pass (last column) so the
                # coordination density uses the very same NS weights.
                obs = np.column_stack(
                    [
                        hold_to_dead_points(g, idx, n_dead, i, panel),
                        hold_to_dead_points(vols, idx, n_dead, i, panel),
                    ]
                )
                out = np.asarray(
                    reweight(mon, obs, T, k_B=self._k_B), dtype=float
                )
                grids.append(out[:, :-1])
                densities.append(n_atoms / out[:, -1])

        window = self._window(r, grids[0])
        mask = (r >= window[0]) & (r <= window[1])
        if not mask.any():
            raise ValueError(
                f"NSRDFExpectationHeatmap: the r-window {window} contains no "
                f"RDF bin (r spans [{r[0]:.2f}, {r[-1]:.2f}])."
            )
        rows = [
            _reduce(grid, r, mask, self._reduction, density=dens).tolist()
            for grid, dens in zip(grids, densities)
        ]

        if self._pressures is not None:
            p = [float(x) for x in self._pressures]
        else:
            p = [float(i) for i in range(len(rows))]

        return RDFExpectationData(
            t=T.tolist(),
            p=p,
            obs=rows,
            r=r.tolist(),
            rdf=[grid.tolist() for grid in grids],
            reduction=self._reduction,
            r_window=list(window),
            quantity=_quantity_label(self._reduction, window),
        )


class RDFSpec(PanelSpec):
    """One NS ``<g(r)>(T)`` panel; repeatable at different reductions.

    ``swap_axes`` / ``k_B`` / ``temperatures`` are NS-global settings a panel
    opts into: left unset they inherit the run-global value (via
    ``alviz.apply_globals``); set here they override just this instance.
    """

    type: str = "rdf_expectation"
    cutoff: float
    nbins: int = 120
    reduction: str = "peak_height"
    r_min: float = 0.0
    r_max: float | None = None
    swap_axes: bool | None = None
    k_B: float | None = None
    temperatures: list | None = None


def _rdf_name(spec):
    """Param-keyed name so instances never collide (cache/grid slot)."""
    if spec.name:
        return spec.name
    return f"rdf_{spec.reduction}_rc{spec.cutoff:g}"


def _build_rdf(spec, ctx, c):
    if not (ctx.ns_dirs and not ctx.is_summary):
        return None
    return NSRDFExpectationHeatmap(
        result_dirs=ctx.ns_dirs,
        prefix=ctx.ns_prefix,
        cutoff=spec.cutoff,
        nbins=spec.nbins,
        reduction=spec.reduction,
        r_min=spec.r_min,
        r_max=spec.r_max,
        temperatures=spec.temperatures,
        k_B=spec.k_B,
        swap_axes=spec.swap_axes,
        name=_rdf_name(spec),
        **c,
    )


register_panel("rdf_expectation", RDFSpec, _build_rdf)
