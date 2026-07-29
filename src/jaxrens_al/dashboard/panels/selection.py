"""Project panel: which sampled configs the selector kept.

The sampler is a pure producer -- it emits *all* NS dead points that its
trajectories retained (every ``output.traj_interval``-th one) -- and a separate
``SelectorActor`` stage thins that pool down to the candidates actually sent
for DFT. Both artifacts carry jaxrens' per-frame provenance: ``iter`` (the
dead-point index within the NS run) and ``run_id`` (the replica).

That makes the selection legible as a map: the NS iteration on x, the replica
on y, one grey point per config the selector *could* have taken and one
coloured point per config it *did*. Clustering at low ``iter``, a replica that
never contributes, or a selector that ignores the late (low-energy, ordered)
part of the run all show up at a glance.

Pure ase-free numpy: the pool is hundreds of MB and only two integers per frame
are needed, so the extxyz headers are parsed directly (see
:func:`frame_indices`) instead of building an ``Atoms`` object per frame.
"""

from __future__ import annotations

import re

import numpy as np
from alviz import Entry, JsonDict, PanelSpec, register_panel

# ``iter=12`` / ``run_id=3`` in an extxyz comment line (integer-valued keys).
_KEY_RE = {
    key: re.compile(rf"(?:^|\s){key}=(-?\d+)") for key in ("iter", "run_id")
}


def _info_int(comment: str, key: str):
    """The integer value of ``key=`` in an extxyz comment line, or None."""
    m = _KEY_RE[key].search(comment)
    return int(m.group(1)) if m else None


def frame_indices(path):
    """``(iter, run_id)`` arrays for the frames of an extxyz file.

    Only the per-frame comment line carries the keys, so each frame's atom
    block is skipped rather than parsed -- an NS pool runs to hundreds of MB
    and ase would build every ``Atoms`` object just to reach two integers.
    Frames missing either key are dropped (a non-jaxrens pool simply yields
    nothing rather than breaking the page).
    """
    iters, runs = [], []
    with open(path, "r") as fh:
        while True:
            head = fh.readline()
            if not head:
                break
            head = head.strip()
            if not head:
                continue
            try:
                n_atoms = int(head)
            except ValueError as exc:
                raise ValueError(
                    f"{path} is not a valid extxyz: expected an atom count, "
                    f"got {head!r}."
                ) from exc
            comment = fh.readline()
            it = _info_int(comment, "iter")
            run = _info_int(comment, "run_id")
            if it is not None and run is not None:
                iters.append(it)
                runs.append(run)
            for _ in range(n_atoms):
                fh.readline()
    return np.asarray(iters, dtype=int), np.asarray(runs, dtype=int)


def _read_group(paths):
    """Concatenated ``(iter, run_id)`` over several config files."""
    iters, runs = [], []
    for path in paths:
        i, r = frame_indices(path)
        iters.append(i)
        runs.append(r)
    if not iters:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(iters), np.concatenate(runs)


class SelectionMapPanel(Entry):
    """Selected configs on an (NS iteration x replica) grid.

    Args:
        groups: ``[(label, files), ...]`` -- one scatter series per label. The
            per-iteration page passes a single group (that iteration's selector
            output); the run summary passes one group per AL iteration.
        pool_files: The full candidate pool to draw behind the selection (the
            sampler output). Empty -> selection only.
        max_pool_points: Cap on the pool points kept in the cached JSON;
            above it the pool is strided down (the background is context, not
            data to read off).
    """

    name = "selection_map"

    def __init__(
        self,
        groups,
        *args,
        pool_files=(),
        max_pool_points: int = 50000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._groups = [
            (str(label), [str(p) for p in files]) for label, files in groups
        ]
        self._pool_files = [str(p) for p in pool_files]
        self._max_pool_points = int(max_pool_points)

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        groups = []
        for label, files in self._groups:
            it, run = _read_group(files)
            groups.append(
                {"label": label, "iter": it.tolist(), "run_id": run.tolist()}
            )

        pool_it, pool_run = _read_group(self._pool_files)
        n_pool = int(pool_it.size)
        stride = max(1, -(-n_pool // self._max_pool_points))  # ceil division
        return JsonDict(
            groups=groups,
            pool={
                "iter": pool_it[::stride].tolist(),
                "run_id": pool_run[::stride].tolist(),
            },
            n_pool=n_pool,
            pool_stride=stride,
        )

    def plot_data(self):
        import matplotlib.pyplot as plt

        pool = self.data["pool"]
        groups = self.data["groups"]

        if pool["iter"]:
            self.ax.scatter(
                pool["iter"],
                pool["run_id"],
                s=3,
                c="0.8",
                marker=".",
                linewidths=0,
                rasterized=True,
                label="pool",
                zorder=1,
            )

        cmap = plt.get_cmap("tab10")
        run_ids = list(pool["run_id"])
        single = len(groups) == 1
        for i, group in enumerate(groups):
            run_ids.extend(group["run_id"])
            self.ax.scatter(
                group["iter"],
                group["run_id"],
                s=10,
                color=cmap(i % 10),
                alpha=0.7,
                linewidths=0,
                rasterized=True,
                label="selected" if single else f"iteration {group['label']}",
                zorder=2,
            )

        # One tick per replica while that stays readable -- the y axis is a
        # replica index, not a continuous coordinate.
        unique = sorted(set(run_ids))
        if unique and len(unique) <= 24:
            self.ax.set_yticks(unique)
        self.ax.set_xlabel("NS iteration (dead point)")
        self.ax.set_ylabel("replica (run_id)")
        n_sel = sum(len(g["iter"]) for g in groups)
        n_pool = self.data["n_pool"]
        self.ax.set_title(f"Selected configs ({n_sel} of {n_pool})")
        if pool["iter"] or len(groups) > 1:
            self.ax.legend(fontsize="x-small", loc="best", markerscale=3)


class SelectionMapSpec(PanelSpec):
    """The selected-configs map; ``show_pool`` draws the candidate pool."""

    type: str = "selection_map"
    show_pool: bool = True
    max_pool_points: int = 50000


def _build_selection_map(spec, ctx, c):
    selected = list(getattr(ctx, "selected_files", None) or [])
    if not selected:
        return None  # no selector stage recorded (classic run) -> no panel

    if ctx.is_summary:
        from jaxrens_al.dashboard.run_layout import (
            selected_configs_per_iteration,
        )

        groups = [
            (str(i), files)
            for i, files in enumerate(
                selected_configs_per_iteration(ctx.calchome)
            )
            if files
        ]
        pool = []  # every iteration's full pool would swamp the page
    else:
        groups = [(str(ctx.iteration), selected)]
        pool = ctx.sampled_files if spec.show_pool else []

    return SelectionMapPanel(
        groups,
        pool_files=pool,
        max_pool_points=spec.max_pool_points,
        **c,
    )


register_panel("selection_map", SelectionMapSpec, _build_selection_map)
