"""Render a dashboard for a *single* jaxrens NS run (no AL_progress.json).

The ``jaxrens-al-dashboard`` CLI is built for a finished *active-learning*
run: it reads ``AL_progress.json`` and shows cross-iteration progress
(dataset growth, parity, sampled-vs-DFT energies, EOS). When a loop only got
as far as the iteration-0 NS sampling -- e.g. it died pulling output back --
there is no ``AL_progress.json`` and no AL progress to show, so the CLI bails.

This script reuses the *same* project panels (``jaxrens_al.panels``) the AL
dashboard uses, wired directly onto one multi-run NS output directory (the
``<prefix>.runNN.energies`` layout) -- so there is no panel logic here, only
wiring. Run it in an env with jaxrens installed, e.g.::

    ~/miniconda3/envs/jaxrens_al/bin/python scripts/ns_dashboard.py \
        experiments/run4/calchome/phase1_mace/0/ns_sampler/results/output \
        --prefix mace_sn16 -o experiments/run4/ns_dashboard.pdf

Add ``--steinhardt-rcut R`` to include the NS ``<q_l>(T)`` heatmap. The
per-replica trajectories are annotated by the real ``remote_handler``
processing pipeline (``SteinhardtProcessor``); ``--cluster`` picks where it
runs -- ``local`` (default, scheduler-free on this machine) or a ClusterConfig
JSON for a dedicated/existing postprocessing cluster. Same code path either
way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from alviz import Dashboard

from jaxrens_al.panels import (
    AdaptationAcceptancePanel,
    NSEnergyTracePanel,
    NSSampledEnergyPanel,
    NSThermodynamicsHeatmap,
    RESwapAcceptancePanel,
    SteinhardtExpectationHeatmap,
    StepSizePanel,
)

# MACE/jax energies are in eV, so scan temperature in Kelvin. (Pass
# --k-B 1.0 with a reduced --tmax for a reduced-unit run instead.)
K_BOLTZMANN_EV_PER_K = 8.617333262e-5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ns_dashboard",
        description="Dashboard for a single multi-run jaxrens NS output dir "
        "(the directory holding <prefix>.runNN.energies).",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="NS output directory (holds <prefix>.runNN.energies, "
        "<prefix>.re_stats.h5, <prefix>.adaptation.h5, "
        "<prefix>.runNN.traj.extxyz).",
    )
    parser.add_argument(
        "--prefix",
        default="ns",
        help="jaxrens out_file_prefix of the run files (default: ns).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PDF (default: <output_dir>/ns_dashboard.pdf).",
    )
    parser.add_argument(
        "--datapath",
        default=None,
        help="Where panels cache computed JSON "
        "(default: <output_dir>/ns_dashboard_data).",
    )
    parser.add_argument(
        "--k-B",
        type=float,
        default=K_BOLTZMANN_EV_PER_K,
        help="Boltzmann constant in the energy units of the .energies log "
        "(default: 8.617e-5 eV/K; use 1.0 for reduced units).",
    )
    parser.add_argument(
        "--tmin", type=float, default=100.0, help="Min temperature."
    )
    parser.add_argument(
        "--tmax", type=float, default=3000.0, help="Max temperature."
    )
    parser.add_argument(
        "--n-T", type=int, default=60, help="Number of temperature points."
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Step-size / acceptance panels: one line per replica instead "
        "of the mean-across-replicas band.",
    )
    parser.add_argument(
        "--steinhardt-rcut",
        type=float,
        default=None,
        help="Enable the NS Steinhardt <q_l>(T) heatmap: neighbour cutoff in "
        "Angstrom. Annotates the dead-point pool locally via jaxrens, cached. "
        "Omit to skip the panel.",
    )
    parser.add_argument(
        "--steinhardt-l",
        type=int,
        default=6,
        help="Steinhardt degree for the <q_l>(T) heatmap (default: 6).",
    )
    parser.add_argument(
        "--steinhardt-rcut-min",
        type=float,
        default=0.0,
        help="Lower Steinhardt neighbour cutoff (default: 0).",
    )
    parser.add_argument(
        "--steinhardt-shard-size",
        type=int,
        default=2000,
        help="Configs per shard when annotating via the processing machinery "
        "(default: 2000).",
    )
    parser.add_argument(
        "--cluster",
        default="local",
        help="Where to run the Steinhardt annotation: 'local' (default, "
        "scheduler-free on this machine) or the name/path of a remote_handler "
        "ClusterConfig JSON (absolute, or resolved under "
        "REMOTE_HANDLER_CONFIG_DIR).",
    )
    parser.add_argument(
        "--swap-pt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Thermodynamics heatmap: pressure on x-axis, temperature on "
        "y-axis (default; use --no-swap-pt for T on x, P on y).",
    )
    parser.add_argument("--force", action="store_true", help="Recompute.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    out_dir = Path(args.output_dir)
    if not out_dir.is_dir():
        print(f"Not a directory: {out_dir}", file=sys.stderr)
        return 1

    energies = sorted(out_dir.glob(f"{args.prefix}.run*.energies"))
    if not energies:
        print(
            f"No {args.prefix}.run*.energies under {out_dir} -- wrong "
            f"--prefix or not a multi-run NS output dir?",
            file=sys.stderr,
        )
        return 1
    trajs = sorted(out_dir.glob(f"{args.prefix}.run*.traj.extxyz"))
    re_stats = out_dir / f"{args.prefix}.re_stats.h5"
    adaptation = out_dir / f"{args.prefix}.adaptation.h5"

    output = Path(args.output) if args.output else out_dir / "ns_dashboard.pdf"
    datapath = (
        Path(args.datapath) if args.datapath else out_dir / "ns_dashboard_data"
    )
    datapath.mkdir(parents=True, exist_ok=True)
    common = dict(datapath=datapath, force_recompute=args.force)

    T = np.linspace(args.tmin, args.tmax, args.n_T)
    entries = [
        # Energy descent per replica -- the at-a-glance "are they all moving".
        NSEnergyTracePanel(
            result_dirs=[out_dir], prefix=args.prefix, **common
        ),
        # Thermodynamics from the dead+live energies, one row per replica.
        NSThermodynamicsHeatmap(
            result_dirs=[out_dir],
            quantity="heat_capacity",
            temperatures=T,
            k_B=args.k_B,
            prefix=args.prefix,
            swap_axes=args.swap_pt,
            **common,
        ),
    ]
    if adaptation.exists():
        entries += [
            StepSizePanel(
                adaptation_file=adaptation, per_run=args.per_run, **common
            ),
            AdaptationAcceptancePanel(
                adaptation_file=adaptation, per_run=args.per_run, **common
            ),
        ]
    if re_stats.exists():
        entries.append(RESwapAcceptancePanel(re_stats_file=re_stats, **common))
    if trajs:
        entries.append(
            NSSampledEnergyPanel(
                config_files=trajs, iteration_index=0, **common
            )
        )
    # NS Steinhardt <q_l>(T) heatmap: reweights the per-dead-point q_l onto the
    # temperature grid with each replica's Monitor (like heat capacity). The
    # panel discovers its own per-replica trajectories and declares a
    # processing need on each; the resolver annotates them below.
    if args.steinhardt_rcut is not None and trajs:
        entries.append(
            SteinhardtExpectationHeatmap(
                result_dirs=[out_dir],
                prefix=args.prefix,
                r_cut=args.steinhardt_rcut,
                r_cut_min=args.steinhardt_rcut_min,
                l=args.steinhardt_l,
                param="q",
                temperatures=T,
                k_B=args.k_B,
                swap_axes=args.swap_pt,
                **common,
            )
        )

    # Run the real processing machinery over any panel that needs it (annotate
    # trajectories), locally via a scheduler-free executor -- one step over the
    # whole dashboard. Skipped (and remote_handler unimported) when nothing
    # declares a need.
    if any(
        p.processing_needs() for p in entries if hasattr(p, "processing_needs")
    ):
        from jaxrens_al.processing_resolver import (
            execution_context,
            resolve_processing,
        )

        print(
            f"Annotating via the processing machinery (cluster="
            f"{args.cluster!r}); may be slow."
        )
        resolve_processing(
            entries,
            shard_size=args.steinhardt_shard_size,
            num_parallel=1,
            **execution_context(args.cluster, datapath / "processing"),
        )

    board = Dashboard.from_entries(
        entries, ncols=2, width_per_panel=5.0, height_per_panel=4.0
    )
    board.render(
        suptitle=f"NS run: {out_dir.name} "
        f"({len(energies)} replicas, prefix {args.prefix!r})"
    )
    board.savefig(output)
    print(f"Wrote {output} ({len(entries)} panel(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
