"""Project panel: relaxed phase of the most likely NS sample on a (T, P) grid.

For every grid temperature and replica (pressure) of an NS pressure sweep the
run-side :func:`jaxrens_al.run.processing_resolver.resolve_phase_grid` picks the
stored dead point with the largest NPT weight
(:func:`jaxrens_al.run.phase_grid.select_grid_samples`), relaxes it with
torch-sim -- positions at fixed cell, then the cell at the replica's pressure --
and labels the relaxed basin with its spglib space group. This panel reads that
labelled set and shows which crystal each region of the NS phase diagram
belongs to (left), next to the relaxed volumes (right), where a first-order
transition shows up as a volume jump.

Unlike :class:`~jaxrens_al.dashboard.panels.phase.PhaseDiagramPanel` (every
dead point relaxed with the jax model, per-phase populations reweighted), this
looks at a handful of representative structures and relaxes the cell too, so
it resolves pressure-driven transitions at the cost of a coarse grid. Samples
from the liquid show the basin a quench falls into, not a stable phase.

Producer/reader split like the other cluster-backed panels: this pure reader
declares a :class:`PhaseGridNeed` (plain data) and receives the labelled file
via :meth:`PhaseGridConsumer.receive_phase_grid`. A MACE run is relaxed with
torch-sim, so the need names a torch ``.model`` file -- by default the torch
original of the model that drove this iteration's sampler
(:func:`~jaxrens_al.dashboard.run_layout.torch_model_for`). A NeuralIL run's
``.pkl`` is relaxed through its jaxrens backend instead
(``jax_cell_relax``), with the ``supercell_trafo`` and neighbour ladder of the
NS run's ``backend`` block. Set ``model`` on the spec for anything else (e.g.
a foundation model).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from alviz import Entry, JsonDict, PanelSpec, register_panel

from jaxrens_al.keys import SPACEGROUP_KEY, SPACEGROUP_SYMBOL_KEY

logger = logging.getLogger(__name__)

KB_EV = 8.617333262e-5
_LOW_SYMMETRY = {0, 1, 2}  # unknown, P1, P-1: disordered / glassy basins
_MARKERS = ("D", "s", "h", "o", "^", "v", "p", "*")
_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#a45ee5",
    "#d6a02a",
    "#d63a6b",
    "#3ab5c9",
    "#6b8e23",
)
_OTHER_COLOR = "#8a8984"


@dataclass(frozen=True)
class PhaseGridNeed:
    """A panel's request to select, relax and label a (T, P) sample grid.

    Hashable, so identical needs (same run, grid and relaxation) deduplicate
    to one job; the run-side resolver keys its output by the whole need.

    Attributes:
        result_dir: jaxrens NS output dir (``<prefix>.runNN.*`` per replica).
        config: NS config yaml holding the per-replica pressure ladder (GPa).
        prefix: jaxrens ``out_file_prefix``.
        temperatures: Grid temperatures, in the units ``k_B`` implies.
        replicas: Replica indices to use (None = all).
        model: Model file the samples are relaxed with -- a torch
            ``.model`` (torch-sim) or a NeuralIL ``.pkl`` / mace-jax bundle
            (jaxrens backend).
        backend, device: Model backend and (torch-sim) device.
        fmax, steps, stages: relaxation criteria and stage order.
        symprec: spglib tolerance of the stored space-group label.
        k_B: Boltzmann constant in the dead-point energy units.
        supercell_trafo, max_neighbors_list, max_neighbors_offset: jaxrens
            backend settings (the NS run's); unused by torch-sim.
    """

    result_dir: str
    config: str | None
    prefix: str
    temperatures: tuple
    replicas: tuple | None
    model: str
    backend: str = "mace"
    device: str = "cuda"
    fmax: float = 0.02
    steps: int = 1000
    stages: tuple = ("positions", "cell")
    symprec: float = 0.1
    k_B: float = KB_EV
    supercell_trafo: tuple | None = None
    max_neighbors_list: tuple | None = None
    max_neighbors_offset: int = 4

    def __repr__(self) -> str:
        """The dataclass repr, minus the jax-relax fields at their default.

        The resolver keys a grid's output dir by ``repr(need)``; leaving the
        fields a torch-sim grid never sets out keeps those grids' existing
        tags (and their on-disk results) valid.
        """
        parts = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in fields(self)
            if f.name not in _JAX_RELAX_FIELDS
            or getattr(self, f.name) != f.default
        ]
        return f"{type(self).__name__}({', '.join(parts)})"

    @classmethod
    def make(
        cls, result_dir, config, prefix, temperatures, replicas, model, **kw
    ):
        return cls(
            str(result_dir),
            None if config is None else str(config),
            str(prefix),
            tuple(float(t) for t in temperatures),
            None if replicas is None else tuple(int(r) for r in replicas),
            str(model),
            **{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in kw.items()
            },
        )


_JAX_RELAX_FIELDS = (
    "supercell_trafo",
    "max_neighbors_list",
    "max_neighbors_offset",
)


class PhaseGridConsumer:
    """Mixin for a panel whose (T, P) grid is relaxed + labelled run-side."""

    def phase_grid_needs(self) -> list:
        """The phase-grid jobs this panel needs (default: none)."""
        return []

    def receive_phase_grid(self, mapping: dict) -> None:
        """Receive ``{PhaseGridNeed -> labelled extxyz}`` from the resolver."""


def hm_tex(symbol: str) -> str:
    """Hermann-Mauguin symbol as mathtext: ``F-43m`` -> ``F$\\bar{4}$3m``."""
    out = re.sub(r"-(\d)", r"$\\bar{\1}$", symbol)
    return re.sub(r"_(\d)", r"$_{\1}$", out)


def phase_name(number, symbol) -> str:
    """Legend label of one space group: ``F$\\bar{4}$3m (216)``."""
    return f"{hm_tex(symbol)} ({int(number)})"


def phase_labels(numbers, symbols, max_phases: int = 6, pinned=()):
    """Group space groups into at most ``max_phases`` named phases + "other".

    The most frequent non-trivial space groups keep their own label; P1/P-1
    (disordered, glassy or quenched-liquid basins), undetermined cells and
    anything beyond ``max_phases`` collapse into one "other" class. Space
    groups whose number is in ``pinned`` always keep their own label (even
    P1, and on top of ``max_phases``), so a fixed colour map shows them in
    every run. Returns ``(labels, order)``: a label per sample and the legend
    order (most frequent first, "other" last).
    """
    pinned = {int(n) for n in pinned}
    names = [phase_name(n, s) for n, s in zip(numbers, symbols)]
    counts = Counter(names)
    free = Counter(
        name
        for name, n in zip(names, numbers)
        if int(n) not in _LOW_SYMMETRY and int(n) not in pinned
    )
    keep = {name for name, _ in free.most_common(max_phases)}
    keep |= {name for name, n in zip(names, numbers) if int(n) in pinned}
    labels = [
        name if name in keep else "other / low symmetry" for name in names
    ]
    order = [name for name, _ in counts.most_common() if name in keep]
    if len(set(labels)) > len(keep):
        order.append("other / low symmetry")
    return labels, order


def parse_phase_colors(phase_colors) -> dict:
    """Normalise a ``phase_colors`` map to ``{key: (marker | None, color)}``.

    Keys are space-group numbers (``216``, also as the string ``"216"``) or
    spglib symbols (``"F-43m"``); values are a matplotlib colour or a
    ``{color, marker}`` mapping. Raises ``ValueError`` on a bad entry so a
    typo in ``dashboard.yaml`` fails before any cluster job.
    """
    from matplotlib.colors import is_color_like

    out = {}
    for key, val in (phase_colors or {}).items():
        if isinstance(key, str) and key.strip().isdigit():
            key = int(key)
        if isinstance(val, dict):
            unknown = set(val) - {"color", "marker"}
            if unknown or "color" not in val:
                raise ValueError(
                    f"phase_colors[{key!r}]: expected a colour or "
                    f"{{color, marker}}, got {val!r}"
                )
            marker, color = val.get("marker"), val["color"]
        else:
            marker, color = None, val
        if not is_color_like(color):
            raise ValueError(f"phase_colors[{key!r}]: {color!r} is no colour")
        out[key] = (marker, color)
    return out


def phase_styles(order, pins: dict, name_pins: dict) -> dict:
    """``{legend name -> (marker, color)}`` for the phases in ``order``.

    Pinned phases (``name_pins``: legend name -> the space group's
    ``(number, symbol)`` key into ``pins``) take their fixed colour; an unset
    pinned marker is derived from the space-group number, so it too is the
    same in every run. Unpinned phases cycle through the palette, skipping
    the pinned colours, in legend order.
    """
    from matplotlib.colors import to_hex

    taken = {to_hex(color) for _, color in pins.values()}
    free = [c for c in _COLORS if c not in taken] or list(_COLORS)
    style, k = {}, 0
    for name in order:
        if name in name_pins:
            number, marker, color = name_pins[name]
            style[name] = (marker or _MARKERS[number % len(_MARKERS)], color)
        elif name.startswith("other"):
            style[name] = ("X", _OTHER_COLOR)
        else:
            style[name] = (_MARKERS[k % len(_MARKERS)], free[k % len(free)])
            k += 1
    return style


class PhaseGridPanel(PhaseGridConsumer, Entry):
    """Relaxed phase of the most likely NS sample per (T, P) cell.

    Args:
        result_dir: The NS output dir the grid is drawn from.
        model_file: torch MLIP file the samples are relaxed with.
        temperatures: Grid temperatures (a handful; one relaxation per
            replica and temperature).
        config: NS config yaml with the replica pressure ladder.
        prefix: jaxrens ``out_file_prefix``.
        replicas: Replica subset (None = all).
        backend, device, fmax, steps, stages: relaxation settings.
        supercell_trafo, max_neighbors_list, max_neighbors_offset: jaxrens
            backend settings for a NeuralIL / mace-jax model.
        symprec: spglib tolerance of the phase label.
        k_B: Boltzmann constant in the dead-point energy units.
        max_phases: Distinct phases in the legend (rest -> "other").
        phase_colors: Fixed style per space group, ``{number or symbol:
            color or {color, marker}}`` (see :func:`parse_phase_colors`) --
            the same phase gets the same colour in every run.
        swap_axes: If True, pressure on x and temperature on y.
    """

    name = "phase_grid"
    colspan = 2

    def __init__(
        self,
        result_dir,
        *args,
        model_file,
        temperatures,
        config=None,
        prefix: str = "ns",
        replicas=None,
        backend: str = "mace",
        device: str = "cuda",
        fmax: float = 0.02,
        steps: int = 1000,
        stages=("positions", "cell"),
        symprec: float = 0.1,
        k_B: float = KB_EV,
        supercell_trafo=None,
        max_neighbors_list=None,
        max_neighbors_offset: int = 4,
        max_phases: int = 4,
        phase_colors=None,
        swap_axes: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._need = PhaseGridNeed.make(
            result_dir,
            config,
            prefix,
            temperatures,
            replicas,
            model_file,
            backend=backend,
            device=device,
            fmax=float(fmax),
            steps=int(steps),
            stages=tuple(stages),
            symprec=float(symprec),
            k_B=float(k_B),
            supercell_trafo=(
                None
                if supercell_trafo is None
                else tuple(int(x) for x in supercell_trafo)
            ),
            max_neighbors_list=(
                None
                if max_neighbors_list is None
                else tuple(int(x) for x in max_neighbors_list)
            ),
            max_neighbors_offset=int(max_neighbors_offset),
        )
        self._max_phases = int(max_phases)
        # Render-only (not in the cached JSON): recolouring needs no rerun.
        self._pins = parse_phase_colors(phase_colors)
        self._swap_axes = bool(swap_axes)
        self._labelled = None

    # -- producer wiring ----------------------------------------------------
    def phase_grid_needs(self) -> list:
        return [self._need]

    def receive_phase_grid(self, mapping: dict) -> None:
        self._labelled = mapping.get(self._need)

    # -- compute ------------------------------------------------------------
    def compute_data(self, shared_ressources: dict) -> JsonDict:
        import ase.io

        if self._labelled is None:
            raise RuntimeError(
                "PhaseGridPanel declared a phase-grid need but nothing was "
                "delivered -- run resolve_phase_grid() over the dashboard "
                "before rendering."
            )
        frames = ase.io.read(str(self._labelled), index=":")
        rows = []
        for at in frames:
            if SPACEGROUP_KEY not in at.info:
                raise ValueError(
                    f"{self._labelled} is missing '{SPACEGROUP_KEY}'. Did the "
                    "spacegroup processor run?"
                )
            info = at.info
            rows.append(
                {
                    "T": float(info["grid_T"]),
                    "P": float(info["grid_P"]),
                    "replica": int(info["replica"]),
                    "iter": int(info["iter"]),
                    "share": float(info.get("grid_weight_share", np.nan)),
                    "V": float(info.get("relax_volume", np.nan)),
                    "H": float(info.get("relax_enthalpy", np.nan)),
                    "converged": bool(info.get("is_converged", False)),
                    "sg": int(info[SPACEGROUP_KEY]),
                    "symbol": str(info.get(SPACEGROUP_SYMBOL_KEY, "unknown")),
                }
            )
        return JsonDict(
            rows=rows,
            model=Path(self._need.model).name,
            symprec=self._need.symprec,
            swap_axes=self._swap_axes,
        )

    # -- render -------------------------------------------------------------
    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(
            1, 2, width_ratios=[1.3, 1], wspace=0.3
        )
        self.ax = fig.add_subplot(self.gs[0, 0])
        self.ax_v = fig.add_subplot(self.gs[0, 1])

    def plot_data(self):
        from matplotlib.lines import Line2D

        rows = self.data["rows"]
        if not rows:
            self.ax.text(
                0.5,
                0.5,
                "no samples",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            return
        name_pins = {}
        for r in rows:
            key = next(
                (k for k in (r["sg"], r["symbol"]) if k in self._pins), None
            )
            if key is not None:
                marker, color = self._pins[key]
                name_pins[phase_name(r["sg"], r["symbol"])] = (
                    r["sg"],
                    marker,
                    color,
                )
        labels, order = phase_labels(
            [r["sg"] for r in rows],
            [r["symbol"] for r in rows],
            self._max_phases,
            pinned={number for number, _, _ in name_pins.values()},
        )
        style = phase_styles(order, self._pins, name_pins)

        P = np.array([r["P"] for r in rows])
        T = np.array([r["T"] for r in rows])
        V = np.array([r["V"] for r in rows])
        conv = np.array([r["converged"] for r in rows])
        labels = np.array(labels)
        swap = self.data["swap_axes"]
        for name in order:
            marker, color = style[name]
            for ok in (True, False):
                sel = (labels == name) & (conv == ok)
                if not sel.any():
                    continue
                face = color if ok else "none"
                x, y = (P[sel], T[sel]) if swap else (T[sel], P[sel])
                self.ax.scatter(
                    x,
                    y,
                    marker=marker,
                    s=46,
                    facecolors=face,
                    edgecolors=color if not ok else "white",
                    linewidths=0.9,
                    zorder=3,
                )
                self.ax_v.scatter(
                    P[sel],
                    V[sel],
                    marker=marker,
                    s=26,
                    facecolors=face,
                    edgecolors=color if not ok else "white",
                    linewidths=0.6,
                    zorder=3,
                )
        self.ax.set_xlabel("$P$" if swap else "$T$")
        self.ax.set_ylabel("$T$" if swap else "$P$")
        self.ax.set_title(
            f"relaxed phase (spglib {self.data['symprec']:g} Å)",
            fontsize="small",
        )
        self.ax_v.set_xlabel("$P$")
        self.ax_v.set_ylabel(r"relaxed $V$ [Å$^3$/atom]")
        self.ax_v.set_title(f"model: {self.data['model']}", fontsize="small")
        self.ax_v.grid(True, alpha=0.3)

        handles = [
            Line2D(
                [],
                [],
                ls="",
                marker=style[n][0],
                markersize=6,
                markerfacecolor=style[n][1],
                markeredgecolor="white",
                label=n,
            )
            for n in order
        ]
        if not conv.all():
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="",
                    marker="o",
                    markersize=6,
                    markerfacecolor="none",
                    markeredgecolor="#52514e",
                    label="not converged",
                )
            )
        self.ax.legend(
            handles=handles,
            fontsize="x-small",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            frameon=False,
        )


class PhaseGridSpec(PanelSpec):
    """Phase of the most likely NS sample on a (T, P) grid. Opt-in.

    ``grid_temperatures`` sets the grid explicitly; otherwise ``n_grid_T``
    temperatures are spread evenly (both ends included) over the run-global range
    (``temperatures``, inherited). ``model`` is a torch ``.model`` file (or a
    NeuralIL ``.pkl``); unset, the iteration's model is used -- for MACE its
    torch original, for NeuralIL the recorded ``.pkl`` itself --
    ``model_source`` picks the one that drove this iteration's sampler
    (``sampling``, default: the phases the NS run actually sampled) or this
    iteration's freshly trained one (``trained``). ``initial_model`` is the
    model for an iteration whose sampling model is not recorded -- iteration
    0, seeded by an out-of-band foundation model. ``config`` defaults to the
    ``ns_config.yaml`` beside the NS output dir. ``replica_stride`` thins the pressure ladder.
    ``backend`` defaults to ``neuralil`` for a ``.pkl`` model, else ``mace``.
    A model torch-sim cannot run (NeuralIL, a mace-jax bundle) is relaxed
    through its jaxrens backend with the NS config's
    ``backend.supercell_trafo`` / ``max_neighbors_list`` /
    ``max_neighbors_offset``, which the same-named spec fields override.
    ``phase_colors`` pins a style per space group so runs stay comparable,
    keyed by number or spglib symbol, valued by a colour or ``{color,
    marker}``, e.g. ``{216: "#2a78d6", F-43m: ..., 186: {color: "#eb6834",
    marker: s}}``; pinned phases always get their own legend entry.
    """

    type: str = "phase_grid"
    model: str | None = None
    model_source: str = "sampling"
    initial_model: str | None = None
    config: str | None = None
    grid_temperatures: list | None = None
    n_grid_T: int = 7
    replica_stride: int = 1
    backend: str = "mace"
    device: str = "cuda"
    fmax: float = 0.02
    steps: int = 1000
    stages: list = ["positions", "cell"]
    symprec: float = 0.1
    supercell_trafo: list | None = None
    max_neighbors_list: list | None = None
    max_neighbors_offset: int | None = None
    max_phases: int = 4
    phase_colors: dict | None = None
    swap_axes: bool | None = None
    k_B: float | None = None
    temperatures: list | None = None


def _ns_config_for(result_dir):
    for cand in (
        Path(result_dir).parent / "ns_config.yaml",
        Path(result_dir) / "ns_config.yaml",
    ):
        if cand.is_file():
            return cand
    return None


def _grid_temperatures(spec):
    if spec.grid_temperatures:
        return [float(t) for t in spec.grid_temperatures]
    if not spec.temperatures:
        return None
    lo, hi = float(min(spec.temperatures)), float(max(spec.temperatures))
    return [float(t) for t in np.linspace(lo, hi, spec.n_grid_T)]


def _model_for(spec, ctx):
    """The model file to relax with: an explicit one, a recorded NeuralIL
    ``.pkl`` as-is, or the torch original of a recorded MACE bundle."""
    if spec.model:
        return Path(spec.model)
    from jaxrens_al.dashboard.run_layout import torch_model_for

    if spec.model_source not in ("sampling", "trained"):
        raise ValueError(
            f"phase_grid: model_source must be 'sampling' or 'trained', got "
            f"{spec.model_source!r}"
        )
    if spec.model_source == "sampling" and ctx.sampling_model_file is None:
        # iteration 0: the seed model is not in AL_progress.json
        return Path(spec.initial_model) if spec.initial_model else None
    recorded = (
        ctx.sampling_model_file
        if spec.model_source == "sampling"
        else ctx.model_file
    )
    if recorded is not None and Path(recorded).suffix == ".pkl":
        return Path(recorded)  # NeuralIL: relaxed by its jaxrens backend
    return torch_model_for(ctx.calchome, recorded)


def _is_torch_model(model, backend) -> bool:
    """Mirrors ``run.phase_grid.relaxer_for`` (kept off the run side)."""
    return backend == "mace" and Path(model).suffix == ".model"


def _backend_for(spec, model) -> str:
    if "backend" in spec.model_fields_set:
        return spec.backend
    return "neuralil" if Path(model).suffix == ".pkl" else spec.backend


def _jax_backend_settings(spec, config) -> dict:
    """``supercell_trafo`` + neighbour ladder: the spec's, else the NS
    config's ``backend`` block (the settings the run sampled with)."""
    block = {}
    if config is not None:
        import yaml

        with open(config) as fh:
            block = (yaml.safe_load(fh) or {}).get("backend") or {}
    offset = spec.max_neighbors_offset
    if offset is None:
        offset = block.get("max_neighbors_offset", 4)
    return dict(
        supercell_trafo=spec.supercell_trafo or block.get("supercell_trafo"),
        max_neighbors_list=(
            spec.max_neighbors_list or block.get("max_neighbors_list")
        ),
        max_neighbors_offset=int(offset),
    )


def _count_replicas(result_dir, prefix):
    return len(
        list(Path(result_dir).glob(f"{prefix}.run[0-9][0-9].traj.extxyz"))
    )


def _build_phase_grid(spec, ctx, c):
    if not ctx.ns_dirs or ctx.is_summary:
        return None
    parse_phase_colors(spec.phase_colors)  # fail on a typo before any job
    result_dir = Path(ctx.ns_dirs[0])
    model = _model_for(spec, ctx)
    temperatures = _grid_temperatures(spec)
    config = Path(spec.config) if spec.config else _ns_config_for(result_dir)
    missing = [
        what
        for what, val in (
            (
                (
                    "model: this iteration's sampling model is not recorded "
                    "(iteration 0) -- set 'initial_model' (or 'model')"
                    if spec.model_source == "sampling"
                    and ctx.sampling_model_file is None
                    else f"torch model: no Model.model beside the "
                    f"{spec.model_source} bundle -- set 'model'"
                ),
                model,
            ),
            ("grid temperatures", temperatures),
            ("the NS config (pressures)", config),
        )
        if val is None
    ]
    if missing:
        logger.info(
            "phase_grid: skipped for %s -- no %s",
            result_dir,
            "; no ".join(missing),
        )
        return None
    backend = _backend_for(spec, model)
    jax_settings = (
        {}
        if _is_torch_model(model, backend)
        else _jax_backend_settings(spec, config)
    )
    replicas = None
    if spec.replica_stride > 1:
        replicas = list(
            range(
                0,
                _count_replicas(result_dir, ctx.ns_prefix),
                spec.replica_stride,
            )
        )
    return PhaseGridPanel(
        result_dir,
        model_file=model,
        temperatures=temperatures,
        config=config,
        prefix=ctx.ns_prefix,
        replicas=replicas,
        backend=backend,
        device=spec.device,
        fmax=spec.fmax,
        steps=spec.steps,
        stages=spec.stages,
        symprec=spec.symprec,
        k_B=spec.k_B if spec.k_B is not None else KB_EV,
        **jax_settings,
        max_phases=spec.max_phases,
        phase_colors=spec.phase_colors,
        swap_axes=bool(spec.swap_axes),
        **c,
    )


register_panel("phase_grid", PhaseGridSpec, _build_phase_grid)


__all__ = [
    "PhaseGridConsumer",
    "PhaseGridNeed",
    "PhaseGridPanel",
    "PhaseGridSpec",
    "hm_tex",
    "parse_phase_colors",
    "phase_labels",
    "phase_name",
    "phase_styles",
]
