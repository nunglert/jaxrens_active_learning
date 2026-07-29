"""Shared machinery for the nested-sampling *expectation* panels.

Several panels answer the same question for different observables: take a
per-configuration quantity annotated onto a jaxrens trajectory, reweight it
onto a temperature grid with that replica's NS ``Monitor``, and render one
heatmap row per replica (``<q_l>(T)``, ``<g(r)>(T)``, ...). What they share
lives here:

* :func:`replica_traj_files` / :func:`load_monitors` -- a run dir's per-replica
  trajectories and Monitors, in the same (sorted ``runNN``) order;
* :func:`hold_to_dead_points` -- expand a *subsampled* trajectory's values back
  onto every dead point (jaxrens keeps every ``output.traj_interval``-th
  frame);
* :func:`reweight` -- ``<O>(T)`` for a scalar **or vector-valued**
  per-dead-point observable;
* :class:`SwapAxesHeatmapMixin` -- the "P on x, T on y" rendering variant.

Everything here is numpy-only except :func:`load_monitors` and the fast path of
:func:`reweight` (both deferred jaxrens imports), so importing a panel still
pulls in no JAX.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from alviz.panels import heatmap_pcolormesh


def replica_traj_files(result_dir, prefix: str) -> list:
    """The per-replica trajectories in ``result_dir``, in replica order.

    Globs ``<prefix>.run*.traj.extxyz`` (which excludes the ``.snap.``
    snapshot files); sorted order matches the Monitors' ``runNN`` order.
    """
    return [
        str(p)
        for p in sorted(Path(result_dir).glob(f"{prefix}.run*.traj.extxyz"))
    ]


def load_monitors(result_dir, prefix: str) -> list:
    """Monitors for one NS output dir, one per replica (the jaxrens seam).

    Multi-run is the only loader: a single-run output is just the one-replica
    edge case of ``from_multi_run_directory``. Panels expose this as a
    ``_load_monitors`` method so tests can patch it.
    """
    from jaxrens.postprocess import MonitorCollection

    return list(
        MonitorCollection.from_multi_run_directory(result_dir, prefix=prefix)
    )


def hold_to_dead_points(vals, idx, n_dead, replica, panel: str):
    """Reconstruct a per-dead-point observable of length ``n_dead``.

    A jaxrens trajectory is usually a regular *slice* of the dead points
    (``output.traj_interval``), so ``vals`` is shorter than ``n_dead``. NS
    reweighting needs a value at every dead point, but consecutive dead points
    barely differ, so we hold each retained value across its slice window
    (piecewise constant, keyed by ``iter``) -- a good approximation that lets
    the NS weights be applied at full resolution. A full-resolution trajectory
    (``len(vals) == n_dead``) is used as-is.

    ``vals`` is ``(n_frames,)`` or ``(n_frames, n_components)`` (a vector-
    valued observable such as an RDF); the hold is along the frame axis either
    way.
    ``panel`` names the caller in the error messages.
    """
    vals = np.asarray(vals, dtype=float)
    if len(vals) == n_dead:
        return vals
    if idx is None:
        raise ValueError(
            f"{panel}: replica {replica} has {len(vals)} annotated configs "
            f"but the NS run has {n_dead} dead points, and the pool carries "
            f"no 'iter' index to align the subsampled trajectory. "
            f"Re-annotate from a jaxrens trajectory (its frames carry 'iter')."
        )
    idx = np.asarray(idx, dtype=int)
    if idx.min() < 0 or idx.max() >= n_dead:
        raise ValueError(
            f"{panel}: replica {replica} dead-point indices ('iter') span "
            f"[{idx.min()}, {idx.max()}] but the NS run has {n_dead} dead "
            f"points -- pool and run mismatched."
        )
    # For each dead point, the most recent retained config's value.
    pos = np.searchsorted(idx, np.arange(n_dead), side="right") - 1
    pos = np.clip(pos, 0, len(vals) - 1)
    return vals[pos]


def _weight_matrix(mon, T, k_B: float):
    """Normalised NS weights ``(n_T, n_dead + n_live)``, or None.

    Reproduces ``jaxrens.postprocess.thermodynamics.expectation``'s weighting
    (prior-mass log weights for the dead points, the shared live-point weight
    for the live walkers, Boltzmann-reweighted at each beta) *once* per
    replica, so a vector-valued observable costs one matmul instead of a full
    pass per component. Returns None when jaxrens or the Monitor's raw arrays
    are unavailable -- the caller then falls back to ``Monitor.expectation``.
    """
    try:
        from jaxrens.postprocess.thermodynamics import (
            calc_log_weights,
            calc_log_weights_live,
        )
    except ImportError:
        return None

    dead_e = getattr(mon, "dead_energies", None)
    live_e = getattr(mon, "live_energies", None)
    n_live = getattr(mon, "n_live", None)
    if dead_e is None or live_e is None or n_live is None:
        return None
    n_cull = getattr(mon, "n_cull", 1) or 1

    dead_e = np.asarray(dead_e, dtype=float)
    live_e = np.asarray(live_e, dtype=float)
    n_dead = dead_e.shape[0]
    log_w = np.concatenate(
        [
            calc_log_weights(n_dead, n_live, n_cull),
            np.full(
                live_e.shape[0],
                calc_log_weights_live(n_dead, n_live, n_cull),
                dtype=float,
            ),
        ]
    )
    energies = np.concatenate([dead_e, live_e])
    beta = 1.0 / (k_B * np.asarray(T, dtype=float))
    log_unnorm = log_w[None, :] - beta[:, None] * energies[None, :]
    log_unnorm -= log_unnorm.max(axis=1, keepdims=True)
    w = np.exp(log_unnorm)
    return w / w.sum(axis=1, keepdims=True)


def reweight(mon, obs, T, k_B: float = 1.0):
    """``<O>(T)``: NS-reweight a per-dead-point observable onto ``T``.

    ``obs`` is ``(n_dead,)`` (scalar observable -> ``(n_T,)``) or
    ``(n_dead, n_components)`` (vector-valued, e.g. an RDF's bins ->
    ``(n_T, n_components)``). The scalar case is exactly
    ``Monitor.expectation``; the vector case applies the *same* weights to
    every component at once, so a few hundred RDF bins cost one matmul rather
    than a few hundred passes over the dead points.

    Live walkers are included exactly as ``Monitor.expectation`` does it: each
    live point carries the observable's dead-point mean (per component). When
    the Monitor does not expose its raw arrays (a test fake), this falls back
    to per-component ``mon.expectation`` calls.
    """
    obs = np.asarray(obs, dtype=float)
    T = np.atleast_1d(np.asarray(T, dtype=float))
    vector = obs.ndim == 2

    w = _weight_matrix(mon, T, k_B)
    if w is None:
        columns = obs.T if vector else obs[None, :]
        rows = [
            np.atleast_1d(np.asarray(mon.expectation(c, T, k_B=k_B), float))
            for c in columns
        ]
        return np.stack(rows, axis=-1) if vector else rows[0]

    n_live_actual = w.shape[1] - obs.shape[0]
    live = np.broadcast_to(obs.mean(axis=0), (n_live_actual,) + obs.shape[1:])
    return w @ np.concatenate([obs, live], axis=0)


class SwapAxesHeatmapMixin:
    """Render a ``HeatmapPanel`` with P on x and T on y when ``_swap_axes``.

    alviz owns the default (T on x, P on y) orientation; a replica ladder often
    reads better transposed, so every NS heatmap panel here offers the swap.
    The transposed branch still renders through ``heatmap_pcolormesh``
    (colorbar / scaling / cmap reused) -- only the grid and the two axis labels
    swap. Mix in *before* the alviz panel class so the unswapped case defers to
    it.
    """

    def plot_data(self):
        if not getattr(self, "_swap_axes", False):
            return super().plot_data()

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
