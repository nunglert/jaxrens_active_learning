"""Render a dashboard for a *single* set of DFT-labelled configs.

The ``jaxrens-al-dashboard`` CLI is built for a finished *active-learning*
run: it reads ``AL_progress.json`` and renders cross-iteration progress. When
a loop dies after the DFT step but before recording a completed iteration --
``AL_progress.json`` then shows that iteration as unfinished, so the CLI skips
it -- the labelled configs are still on disk and worth looking at on their own.

This script wires the *same* project / alviz panels the AL dashboard uses
directly onto one extxyz of DFT-labelled configs (``REF_energy`` /
``REF_forces``, the keys ``remote_handler``'s evaluators write), so there is no
panel logic here, only wiring. Point it at the evaluator output, e.g.::

    ~/miniconda3/envs/jaxrens_al/bin/python scripts/dft_dashboard.py \
        experiments/run6/.../evaluator_output/evaluator_it0_0.extxyz

Pass several files to pool them into one set. The ``ns_energy`` key (the
sampler's foundation-model energy, if present) is plotted against the DFT
reference as a parity panel -- "how far off was the sampler's surrogate".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alviz import Dashboard

from jaxrens_al.cli_args import add_output_args, add_ref_key_args
from jaxrens_al.sections import dft_section


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dft_dashboard",
        description="Dashboard for a single set of DFT-labelled configs "
        "(extxyz with REF_energy / REF_forces).",
    )
    parser.add_argument(
        "config_files",
        nargs="+",
        help="One or more DFT-labelled extxyz files (REF_energy / "
        "REF_forces). Several are pooled into one set.",
    )
    add_output_args(parser)
    add_ref_key_args(parser)
    parser.add_argument(
        "--pred-energy-key",
        default="ns_energy",
        help="info key of a surrogate energy to plot against the reference "
        "(default: ns_energy; the sampler's foundation-model energy).",
    )
    parser.add_argument(
        "--no-parity",
        action="store_true",
        help="Skip the surrogate-vs-DFT parity / error panels even if "
        "--pred-energy-key is present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    files = [Path(f) for f in args.config_files]
    missing = [f for f in files if not f.is_file()]
    if missing:
        print(
            "No such file(s): " + ", ".join(str(f) for f in missing),
            file=sys.stderr,
        )
        return 1

    output = (
        Path(args.output)
        if args.output
        else files[0].parent / "dft_dashboard.pdf"
    )
    datapath = (
        Path(args.datapath)
        if args.datapath
        else output.parent / "dft_dashboard_data"
    )
    datapath.mkdir(parents=True, exist_ok=True)
    common = dict(datapath=datapath, force_recompute=args.force)

    entries = dft_section(
        files,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        pred_energy_key=None if args.no_parity else args.pred_energy_key,
        **common,
    )

    board = Dashboard.from_entries(
        entries, ncols=2, width_per_panel=5.0, height_per_panel=4.0
    )
    n_label = " + ".join(f.name for f in files)
    board.render(suptitle=f"DFT-labelled configs: {n_label}")
    board.savefig(output)
    print(f"Wrote {output} ({len(entries)} panel(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
