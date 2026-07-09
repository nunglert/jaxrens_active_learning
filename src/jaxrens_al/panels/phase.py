"""Project panel: nested-sampling phase diagram from relaxed space groups.

Each NS dead point is relaxed to its local minimum and labelled by its space
group (the ``relax`` -> ``spacegroup`` cluster pipeline; see
:func:`jaxrens_al.processing_resolver.resolve_phase_labeling`). This panel reads
the labelled pool, pairs it with the jaxrens ``Monitor`` (exactly as
:class:`~jaxrens_al.panels.steinhardt.SteinhardtExpectationHeatmap` does), and
reweights a per-phase indicator over a temperature grid: ``<1[sg==g]>(T)`` is the
NS population (basin free-energy weight) of phase ``g`` at ``T``. The stable
phase in each ``(P, T)`` cell -- ``argmax_g`` population -- gives a P-T phase
diagram over the phases that actually appear in the run's dead points.

Producer/reader split like the Steinhardt panels: this pure reader declares a
:class:`~jaxrens_al.panels.phase.PhaseNeed`; the run-side resolver runs the
relax+spacegroup job (with *this iteration's* model) and delivers the labelled
trajectory back via :meth:`receive_phase_labels`.

Caveat: NS over one simulation cell gives *relative* basin free energies for the
competing structures sampled in that cell -- a legitimate diagram for those
phases, not a full multi-cell construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from alviz import Entry, JsonDict, PanelSpec, register_panel

from jaxrens_al.keys import SPACEGROUP_KEY, SPACEGROUP_SYMBOL_KEY


@dataclass(frozen=True)
class PhaseNeed:
    """A panel's request to relax + space-group-label a set of trajectories.

    Hashable, so identical needs deduplicate. Grouped by ``model`` run-side
    (each iteration relaxes with its own model -> its own job).

    Attributes:
        inputs: Trajectory file(s) to relax + label -- the grouping key.
        model: Trained-model bundle to relax with (mace-jax dir / neuralil pkl).
        backend: jaxrens backend to load ``model`` with.
        fmax, steps: LBFGS relaxation criteria.
        symprec: spglib distance tolerance for the space-group assignment.
        max_neighbors: Static neighbour-buffer size for the backend call.
    """

    inputs: tuple
    model: str
    backend: str
    fmax: float = 0.05
    steps: int = 200
    symprec: float = 1e-3
    max_neighbors: int = 50

    @classmethod
    def make(
        cls,
        inputs,
        model,
        backend: str,
        *,
        fmax: float = 0.05,
        steps: int = 200,
        symprec: float = 1e-3,
        max_neighbors: int = 50,
    ) -> "PhaseNeed":
        return cls(
            tuple(str(i) for i in inputs),
            str(model),
            backend,
            float(fmax),
            int(steps),
            float(symprec),
            int(max_neighbors),
        )


class PhaseConsumer:
    """Mixin for a panel whose samples are relaxed + labelled on the cluster.

    Override :meth:`phase_needs` to declare the pipeline the panel's
    trajectories need and :meth:`receive_phase_labels` to pick up the resolved
    ``{model -> {input -> labelled}}`` mapping.
    """

    def phase_needs(self) -> list:
        """The phase labelling this panel needs (default: none)."""
        return []

    def receive_phase_labels(self, mapping: dict) -> None:
        """Receive ``{model -> {input -> labelled_file}}`` from the resolver."""


class PhaseDiagramPanel(PhaseConsumer, Entry):
    """P-T phase diagram: the stable (most-populated) space group per cell.

    Args:
        result_dirs: One or more multi-run NS output dirs (per-replica
            ``<prefix>.run*.traj.extxyz`` + the combined checkpoint). Each
            replica is one pressure row.
        model_file: This iteration's trained-model bundle -- the model each
            dead point is relaxed with (a mace-jax bundle dir / neuralil pkl).
        backend_type: jaxrens backend to load ``model_file`` with.
        fmax, steps: LBFGS relaxation criteria.
        symprec: spglib distance tolerance for the space-group assignment.
        temperatures / tlim / n_T: the reweighting temperature grid.
        k_B: Boltzmann constant in the stored energies' units.
        prefix: jaxrens ``out_file_prefix`` of the run files.
        pressures: Optional row labels (one per replica); default row indices.
        swap_axes: If True, render pressure on x and temperature on y.
        max_neighbors: Static neighbour-buffer size for the backend call.
    """

    name = "phase_diagram"
    colspan = 1

    def __init__(
        self,
        result_dirs,
        *args,
        model_file,
        backend_type: str = "mace",
        fmax: float = 0.05,
        steps: int = 200,
        symprec: float = 1e-3,
        temperatures=None,
        tlim=(0.05, 2.0),
        n_T: int = 50,
        k_B: float = 1.0,
        prefix: str = "ns",
        pressures=None,
        swap_axes: bool = False,
        max_neighbors: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._result_dirs = [str(p) for p in result_dirs]
        self._model_file = str(model_file)
        self._backend_type = backend_type
        self._fmax = float(fmax)
        self._steps = int(steps)
        self._symprec = float(symprec)
        self._max_neighbors = int(max_neighbors)
        if temperatures is not None:
            self._temperatures = np.asarray(temperatures, dtype=float)
        else:
            self._temperatures = np.linspace(tlim[0], tlim[1], n_T)
        self._k_B = k_B
        self._prefix = prefix
        self._pressures = pressures
        self._swap_axes = swap_axes
        # {raw trajectory -> relaxed+labelled trajectory}, filled by resolver.
        self._labels: dict = {}

    # -- producer wiring ----------------------------------------------------
    def _traj_files(self, result_dir) -> list:
        return [
            str(p)
            for p in sorted(
                Path(result_dir).glob(f"{self._prefix}.run*.traj.extxyz")
            )
        ]

    def phase_needs(self) -> list:
        trajs = [t for d in self._result_dirs for t in self._traj_files(d)]
        if not trajs:
            return []
        return [
            PhaseNeed.make(
                trajs,
                self._model_file,
                self._backend_type,
                fmax=self._fmax,
                steps=self._steps,
                symprec=self._symprec,
                max_neighbors=self._max_neighbors,
            )
        ]

    def receive_phase_labels(self, mapping: dict) -> None:
        self._labels = dict(mapping.get(self._model_file, {}))

    # -- compute ------------------------------------------------------------
    def _load_monitors(self, result_dir):
        """Monitors for one NS output dir, one per replica (test seam)."""
        from jaxrens.postprocess import MonitorCollection

        return list(
            MonitorCollection.from_multi_run_directory(
                result_dir, prefix=self._prefix
            )
        )

    def _read_labels(self, src):
        """Per-config space-group number + dead-point index for one replica.

        Returns ``(numbers, indices, symbols)``; ``indices`` (from
        ``info['iter']``) is None if no frame carries it, and ``symbols`` maps
        each seen number to its Hermann-Mauguin symbol (for the legend).
        """
        import ase.io

        frames = ase.io.read(str(src), index=":")
        nums: list = []
        idxs: list = []
        symbols: dict = {}
        have_iter = True
        for at in frames:
            if SPACEGROUP_KEY not in at.info:
                raise ValueError(
                    f"Labelled trajectory {src} is missing "
                    f"'{SPACEGROUP_KEY}'. Did the spacegroup processor run?"
                )
            g = int(at.info[SPACEGROUP_KEY])
            nums.append(g)
            symbols.setdefault(g, str(at.info.get(SPACEGROUP_SYMBOL_KEY, g)))
            it = at.info.get("iter")
            if it is None:
                have_iter = False
            idxs.append(it)
        return (
            np.asarray(nums, int),
            np.asarray(idxs, int) if have_iter else None,
            symbols,
        )

    def _to_dead_point(self, vals, idx, n_dead, replica):
        """Reconstruct a per-dead-point observable of length ``n_dead``.

        A jaxrens trajectory is usually a regular slice of the dead points, so
        the retained values are held piecewise-constant across their window
        (keyed by ``iter``) -- the same alignment
        :class:`SteinhardtExpectationHeatmap` uses so ``Monitor.expectation``
        reweights with the full NS weights.
        """
        if len(vals) == n_dead:
            return vals
        if idx is None:
            raise ValueError(
                f"PhaseDiagramPanel: replica {replica} has {len(vals)} "
                f"labelled configs but the NS run has {n_dead} dead points, "
                f"and the pool carries no 'iter' index to align them."
            )
        pos = np.searchsorted(idx, np.arange(n_dead), side="right") - 1
        pos = np.clip(pos, 0, len(vals) - 1)
        return np.asarray(vals)[pos]

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        if not self._result_dirs:
            raise RuntimeError("PhaseDiagramPanel: no NS result dirs given.")
        T = self._temperatures
        rows: list = []
        symbols: dict = {}
        present: set = set()
        for result_dir in self._result_dirs:
            trajs = self._traj_files(result_dir)
            if not trajs:
                raise RuntimeError(
                    f"PhaseDiagramPanel: no '{self._prefix}.run*.traj.extxyz' "
                    f"under {result_dir}."
                )
            monitors = self._load_monitors(result_dir)
            if len(trajs) != len(monitors):
                raise ValueError(
                    f"PhaseDiagramPanel: {len(trajs)} trajectories but "
                    f"{len(monitors)} replicas in {result_dir}."
                )
            for i, (traj, mon) in enumerate(zip(trajs, monitors)):
                src = self._labels.get(traj)
                if src is None:
                    raise RuntimeError(
                        "PhaseDiagramPanel declared a phase need but no "
                        "labels were delivered -- run resolve_phase_labeling() "
                        "over the dashboard before rendering."
                    )
                nums, idx, syms = self._read_labels(src)
                symbols.update(syms)
                n_dead = getattr(mon, "n_dead", len(nums))
                sg_dead = self._to_dead_point(nums, idx, n_dead, i)
                unique = sorted({int(x) for x in sg_dead})
                pops = np.vstack(
                    [
                        np.asarray(
                            mon.expectation(
                                (sg_dead == g).astype(float), T, k_B=self._k_B
                            ),
                            dtype=float,
                        )
                        for g in unique
                    ]
                )
                present.update(unique)
                rows.append([int(unique[j]) for j in np.argmax(pops, axis=0)])

        if self._pressures is not None:
            p = [float(x) for x in self._pressures]
        else:
            p = [float(i) for i in range(len(rows))]
        return JsonDict(
            t=T.tolist(),
            p=p,
            phase_grid=rows,
            phases=sorted(present),
            symbols={str(k): v for k, v in symbols.items()},
            swap_axes=self._swap_axes,
        )

    # -- render -------------------------------------------------------------
    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    def plot_data(self):
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        d = self.data
        phases = d["phases"]
        symbols = d["symbols"]
        index_of = {g: k for k, g in enumerate(phases)}
        grid = np.array(
            [[index_of[g] for g in row] for row in d["phase_grid"]]
        )  # (n_p, n_T)
        t = np.asarray(d["t"], dtype=float)
        p = np.asarray(d["p"], dtype=float)
        cmap = plt.get_cmap("tab20", max(len(phases), 1))

        if not d["swap_axes"]:
            x, y, c, xl, yl = t, p, grid, r"$T$", r"$P$"
        else:
            x, y, c, xl, yl = p, t, grid.T, r"$P$", r"$T$"
        self.ax.pcolormesh(
            x,
            y,
            c,
            cmap=cmap,
            shading="nearest",
            vmin=-0.5,
            vmax=len(phases) - 0.5,
        )
        self.ax.set_xlabel(xl)
        self.ax.set_ylabel(yl)
        self.ax.set_title("stable phase")
        handles = [
            mpatches.Patch(
                color=cmap(k), label=f"{symbols.get(str(g), g)} ({g})"
            )
            for k, g in enumerate(phases)
        ]
        self.ax.legend(
            handles=handles,
            fontsize="x-small",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )


class PhaseDiagramSpec(PanelSpec):
    """NS phase diagram (relaxed-space-group populations). Opt-in, repeatable.

    ``swap_axes`` / ``k_B`` / ``temperatures`` inherit the NS run-globals unless
    set here. ``backend`` inherits the run's ``eos_backend`` when unset; the
    model is this iteration's trained bundle.
    """

    type: str = "phase_diagram"
    backend: str | None = None
    fmax: float = 0.05
    steps: int = 200
    symprec: float = 1e-3
    max_neighbors: int = 50
    swap_axes: bool | None = None
    k_B: float | None = None
    temperatures: list | None = None


def _build_phase_diagram(spec, ctx, c):
    if not (ctx.ns_dirs and not ctx.is_summary and ctx.model_file):
        return None
    return PhaseDiagramPanel(
        result_dirs=ctx.ns_dirs,
        model_file=ctx.model_file,
        backend_type=spec.backend or ctx.eos_backend,
        fmax=spec.fmax,
        steps=spec.steps,
        symprec=spec.symprec,
        max_neighbors=spec.max_neighbors,
        prefix=ctx.ns_prefix,
        temperatures=spec.temperatures,
        k_B=spec.k_B,
        swap_axes=spec.swap_axes,
        **c,
    )


register_panel("phase_diagram", PhaseDiagramSpec, _build_phase_diagram)


__all__ = [
    "PhaseDiagramPanel",
    "PhaseDiagramSpec",
    "PhaseNeed",
    "PhaseConsumer",
]
