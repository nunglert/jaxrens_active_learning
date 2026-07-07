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

from jaxrens_al.cli_args import (
    add_output_args,
    add_processing_args,
    add_steinhardt_args,
    add_thermo_args,
    steinhardt_params,
)
from jaxrens_al.sections import ns_section

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
    add_output_args(parser)
    # Kelvin defaults: MACE/jax energies are in eV (see K_BOLTZMANN_EV_PER_K).
    add_thermo_args(
        parser,
        k_B=K_BOLTZMANN_EV_PER_K,
        tmin=100.0,
        tmax=3000.0,
        n_T=60,
        swap_default=True,
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Step-size / acceptance panels: one line per replica instead "
        "of the mean-across-replicas band.",
    )
    add_steinhardt_args(parser)
    add_processing_args(parser)
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

    output = Path(args.output) if args.output else out_dir / "ns_dashboard.pdf"
    datapath = (
        Path(args.datapath) if args.datapath else out_dir / "ns_dashboard_data"
    )
    datapath.mkdir(parents=True, exist_ok=True)
    common = dict(datapath=datapath, force_recompute=args.force)

    T = np.linspace(args.tmin, args.tmax, args.n_T)
    entries = ns_section(
        out_dir,
        prefix=args.prefix,
        k_B=args.k_B,
        temperatures=T,
        swap_axes=args.swap_pt,
        per_run=args.per_run,
        steinhardt=steinhardt_params(args),
        **common,
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
            shard_size=args.shard_size,
            num_parallel=args.num_parallel,
            n_processors=args.n_processors,
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
