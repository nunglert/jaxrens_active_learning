"""Project panels: Steinhardt order parameters, computed once and reused.

The (expensive, remote) Steinhardt computation lives in the run-side
:class:`jaxrens_al.processors.steinhardt.SteinhardtProcessor` (a
``remote_handler`` trajectory-processing step), run via
``remote_handler.processing.RemoteTrajectoryProcessor``; it writes an annotated
trajectory (per-atom ``q<l>`` / ``w<l>`` arrays + per-frame ``q<l>_mean`` /
``w<l>_mean`` in ``atoms.info``). These panels are pure **readers** of that
artifact (ase + numpy, no cluster, no ``remote_handler`` import).

Each panel takes its data from ``annotated_file=`` -- an already-annotated
extxyz. Run the processor once and hand the *same* output file to any number of
panels: that is how one computation is shared (the sharing is explicit, in the
file, not hidden behind a producer). ``remote_handler`` never enters here.

Panels:
  * :class:`SteinhardtDistributionPanel` -- per-atom ``q_l`` / ``w_l``
    distribution (joint scatter for two l-values, else per-l histograms);
  * :class:`SteinhardtHeatmapPanel` -- expectation ``<q_l>`` (or ``<w_l>``) as
    a heatmap over a 2D grid of two per-config keys (e.g. volume vs energy).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from alviz import Entry, JsonDict
from alviz.panels import HeatmapData, HeatmapPanel

_PARAMS = ("q", "w")


def _read_frames(annotated: Path):
    import ase.io

    return ase.io.read(str(annotated), index=":")


def _require_array(at, key, annotated):
    if key not in at.arrays:
        raise ValueError(
            f"Annotated trajectory {annotated} is missing per-atom array "
            f"'{key}'. Did the Steinhardt processor write it?"
        )
    return np.asarray(at.arrays[key], float)


class _SteinhardtSource:
    """Shared source-resolution for the reader panels (mix in before Entry)."""

    def _init_source(self, annotated_file, l_values, param):
        if annotated_file is None:
            raise ValueError("Steinhardt panel needs annotated_file=.")
        self._annotated_file = annotated_file
        self._l_values = [int(l) for l in l_values]
        if not self._l_values:
            raise ValueError("Steinhardt panel needs at least one l value.")
        if param not in _PARAMS:
            raise ValueError(f"param must be one of {_PARAMS}; got {param!r}.")
        self._param = param

    def _annotated(self) -> Path:
        return Path(self._annotated_file)


class SteinhardtDistributionPanel(_SteinhardtSource, Entry):
    """Per-atom ``q_l`` / ``w_l`` distribution of a set of configs.

    With exactly two ``l_values`` (e.g. ``(4, 6)``) draws the classic joint
    ``X_{l0}``-vs-``X_{l1}`` scatter coloured by species (``X`` = ``param``);
    otherwise one histogram per ``l``.

    Args:
        annotated_file: An extxyz already annotated by the Steinhardt
            processor (``processors.steinhardt.SteinhardtProcessor``).
        l_values: Steinhardt degrees present in the annotation (default 4, 6).
        param: ``"q"`` (default) or ``"w"``.
    """

    name = "steinhardt"

    def __init__(
        self,
        *args,
        annotated_file: Optional[os.PathLike] = None,
        l_values: Sequence[int] = (4, 6),
        param: str = "q",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_source(annotated_file, l_values, param)
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
        annotated_file: annotated extxyz to read (see module doc).
        l: Single Steinhardt degree to map (default 6).
        param: ``"q"`` (default) or ``"w"``.
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
        l: int = 6,
        param: str = "q",
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
        self._init_source(annotated_file, [l], param)
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
