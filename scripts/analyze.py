"""Analysis for the jaxmd_si16 NPT pressure-sweep run.

Loads the multi-run output via ``MonitorCollection.from_multi_run_directory``
(one Monitor per replica), tags each with its pressure, and emits the
standard ``postprocess.plotting`` overlays (energy trace, cumulative log Z,
log Z(T), Cp(T), G(T)) plus a summary table.

Notes
-----
The NS run uses ``EnsembleBackend(pressure=P)`` per replica, so the energy
stored in ``pop.energy`` (and the ``.energies`` file) is the NPT enthalpy
``H = U + P*V``.  ``from_multi_run_directory`` correspondingly leaves
``dead_volumes`` / ``live_volumes`` as ``None`` so the thermodynamics
functions treat the energy column as the full thermal variable — yielding
``log Z_NPT`` / ``Cp`` / Gibbs ``G`` directly.  The bare cell volume
column from each ``.energies`` log is stashed on each Monitor as
``volume_trace`` for separate ``<V>(T)`` analysis.

Run::

    /home/nico.unglert/miniconda3/envs/jaxrens/bin/python \\
        experiments/jaxmd_si16_2/analyze.py
"""

from __future__ import annotations

import os
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jaxrens.postprocess import Monitor, MonitorCollection
from jaxrens.postprocess.plotting import (
    plot_max_neighbors,
    plot_re_acceptance_stacked,
)

# HERE = Path(__file__).parent
HERE = Path(os.getcwd())
OUT_DIR = HERE / "output"
FIG_DIR = HERE / "figures"
PREFIX = "mace_sn16"


# ---------------------------------------------------------------------------
# Build a MonitorCollection from the multi-run output
# ---------------------------------------------------------------------------


def load_pressure_sweep(out_dir: Path | str = OUT_DIR) -> MonitorCollection:
    """Build one Monitor per pressure replica.

    Thin wrapper around ``MonitorCollection.from_multi_run_directory``.
    With ``config.yaml`` sitting next to ``output/`` the helper resolves
    the file prefix, pressure-sweep labels, and per-replica
    ``pressure_gpa`` metadata itself; we just hand it the run directory.
    """
    return MonitorCollection.from_multi_run_directory(out_dir)


# Boltzmann constant in eV/K — the jax-md Tersoff backend stores energies
# in eV, so we need this factor to scan T in Kelvin (β = 1/(k_B*T)).
K_BOLTZMANN_EV_PER_K = 8.617333262e-5


# ---------------------------------------------------------------------------
# Volume vs temperature (NPT-specific observable)
# ---------------------------------------------------------------------------


def volume_vs_temperature(
    monitor: Monitor,
    T: np.ndarray,
    k_B: float = K_BOLTZMANN_EV_PER_K,
) -> np.ndarray:
    """Compute <V>(T) using ``Monitor.expectation`` over the cell-volume series.

    The volume column from the energy log is one entry per dead point, so
    the live-tail entries are filled with the mean dead volume (matches the
    convention in :py:meth:`Monitor.expectation`).
    """
    return monitor.expectation(monitor.volume_trace, T, k_B=k_B)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _decorate(ax, title: str) -> None:
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.3)


def make_plots(coll: MonitorCollection, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Si melting via Tersoff sits around 2-3 kK; bracket comfortably.
    T = np.linspace(100.0, 2500.0, 200)
    k_B = K_BOLTZMANN_EV_PER_K  # energies are in eV; scan T in Kelvin.

    fig, ax = plt.subplots(figsize=(8, 5))
    coll.plot_energy_trace(ax=ax)
    _decorate(ax, "Dead-point NPT enthalpy vs iteration")
    ax.set_ylabel("H = U + PV  [eV]")
    fig.tight_layout()
    fig.savefig(fig_dir / "energy_trace.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    coll.plot_log_evidence_trace(ax=ax)
    _decorate(ax, "Cumulative log Z (prior-mass weighted)")
    fig.tight_layout()
    fig.savefig(fig_dir / "log_evidence_trace.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    coll.plot_heat_capacity(T, ax=ax, k_B=k_B)
    _decorate(ax, "Heat capacity Cp(T) per pressure")
    ax.set_xlabel("T  [K]")
    fig.tight_layout()
    fig.savefig(fig_dir / "heat_capacity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    coll.plot_partition_function(T, ax=ax, k_B=k_B)
    _decorate(ax, "log Z_NPT(T) per pressure")
    ax.set_xlabel("T  [K]")
    fig.tight_layout()
    fig.savefig(fig_dir / "log_partition_function.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    coll.plot_free_energy(T, ax=ax, k_B=k_B)
    _decorate(ax, "Gibbs free energy G(T) per pressure")
    ax.set_xlabel("T  [K]")
    ax.set_ylabel("G [eV]")
    fig.tight_layout()
    fig.savefig(fig_dir / "gibbs_free_energy.png", dpi=150)
    plt.close(fig)

    # Volume vs T (NPT-specific, sourced from the volume column of .energies).
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for i, m in enumerate(coll):
        V_T = volume_vs_temperature(m, T, k_B=k_B)
        ax.plot(T, V_T, color=cmap(i / max(1, len(coll) - 1)), label=m.label)
    ax.set_xlabel("T  [K]")
    ax.set_ylabel("<V>(T)  [Å³]")
    _decorate(ax, "Mean cell volume vs temperature")
    fig.tight_layout()
    fig.savefig(fig_dir / "volume_vs_T.png", dpi=150)
    plt.close(fig)

    # P-T heatmaps: Cp, <V>, and log Z over the pressure sweep.
    for quantity, fname, title in [
        ("heat_capacity", "heatmap_Cp.png", "Cp(T, P) heatmap"),
        ("volume", "heatmap_V.png", "<V>(T, P) heatmap"),
        ("log_Z", "heatmap_logZ.png", "log Z_NPT(T, P) heatmap"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        coll.plot_heatmap(T, quantity, ax=ax, k_B=k_B)
        ax.set_title(title)
        ax.set_xlabel("P  [GPa]")
        ax.set_ylabel("T  [K]")
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    # Adaptation trace: per-move step size and acceptance rate vs iteration,
    # as mean ± std across replicas.
    if coll.adaptation_trace is not None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        coll.plot_step_sizes(ax=axes[0], per_run=False)
        _decorate(axes[0], "Step size — mean ± std across replicas")
        axes[0].set_yscale("log")
        coll.plot_acceptance_rates(ax=axes[1], per_run=False)
        _decorate(axes[1], "Acceptance rate — mean ± std across replicas")
        fig.tight_layout()
        fig.savefig(fig_dir / "adaptation.png", dpi=150)
        plt.close(fig)

    # Inter-replica swap acceptance — vertically stacked per pair so curves
    # don't overlap.  Pair labels read off the collection labels.
    re_path = OUT_DIR / f"{PREFIX}.re_stats.h5"
    if re_path.exists():
        from jaxrens.io.re_stats_log import RELogger

        re_log = RELogger.read(re_path)
        labels = [m.label for m in coll]
        if len(labels) == re_log.n_pairs + 1:
            pair_labels = [
                f"{labels[p]} ↔ {labels[p + 1]}" for p in range(re_log.n_pairs)
            ]
        else:
            pair_labels = None
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * re_log.n_pairs + 2)))
        plot_re_acceptance_stacked(re_log, ax=ax, pair_labels=pair_labels)
        ax.set_title("Inter-replica swap acceptance — per-pair stacked")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "re_acceptance_stacked.png", dpi=150)
        plt.close(fig)

    # Max-neighbor diagnostics — bucket headroom over time.  Percentile view
    # is the at-a-glance "is the bucket about to overflow"; heatmap shows
    # the full per-walker distribution.
    mn_path = OUT_DIR / f"{PREFIX}.max_neighbors.h5"
    if mn_path.exists():
        from jaxrens.io.max_neighbors_log import MaxNeighborsLogger

        mn_log = MaxNeighborsLogger.read(mn_path)
        n_runs = mn_log.n_runs
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        plot_max_neighbors(mn_log, ax=axes[0], kind="percentiles", run=0)
        axes[0].set_title(
            f"max_neighbors percentiles (run 0 of {n_runs}) vs bucket"
        )
        axes[0].grid(alpha=0.3)
        plot_max_neighbors(mn_log, ax=axes[1], kind="heatmap", run=0)
        axes[1].set_title("max_neighbors per-walker distribution (run 0)")
        fig.tight_layout()
        fig.savefig(fig_dir / "max_neighbors.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tabular summary
# ---------------------------------------------------------------------------


def print_summary(coll: MonitorCollection) -> None:
    print()
    hdr = f"{'replica':>16}  {'n_dead':>8}  {'log_Z':>12}  {'<V>(100 K)':>12}"
    print(hdr)
    print("-" * len(hdr))
    T_low = np.array([100.0])  # K
    for m in coll:
        v_lowT = float(volume_vs_temperature(m, T_low)[0])
        print(
            f"{m.label:>16}  {m.n_dead:>8d}  {m.log_evidence:>12.4f}  "
            f"{v_lowT:>12.2f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    coll = load_pressure_sweep()
    print_summary(coll)
    make_plots(coll, FIG_DIR)
    print(f"\nFigures written to {FIG_DIR}/")
