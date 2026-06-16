"""Standalone dashboard CLI for a remote_handler active-learning run.

Plotting is decoupled from running the loop (mirroring the legacy
``jaxns/scripts/utility/al_dashboard.py``): the AL loop only produces the
``calchome`` artifacts; this command turns them into a PDF afterwards, as many
times as you like. Run it as the installed console script::

    jaxrens-al-dashboard path/to/calchome
    jaxrens-al-dashboard path/to/calchome -o summary.pdf --force

or via the thin wrapper ``python scripts/al_dashboard.py path/to/calchome``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jaxrens-al-dashboard",
        description="Build an alviz dashboard PDF for a remote_handler "
        "active-learning run (a 'calchome' directory holding "
        "AL_progress.json).",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path of the AL run (the calchome directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: <path>/dashboard.pdf).",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        help="Where panels cache computed JSON (default: <path>/dashboard).",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=4.0,
        help="Width per panel in inches (default: 4.0).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=4.0,
        help="Height per panel in inches (default: 4.0).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replot even if the output PDF already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble and report the panel layout without "
        "computing/plotting or writing a PDF.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    args = _build_parser().parse_args(argv)

    calchome = Path(args.path)
    if not calchome.exists():
        logger.error(f"Path does not exist: {calchome}")
        return 1
    if not (calchome / "AL_progress.json").exists():
        logger.error(
            f"No AL_progress.json under {calchome} -- is this a calchome "
            f"directory produced by a remote_handler AL run?"
        )
        return 1

    output = Path(args.output) if args.output else calchome / "dashboard.pdf"

    if output.exists() and not args.force and not args.dry_run:
        logger.info(
            f"{output} already exists; skipping (pass --force to replot)."
        )
        return 0

    # Import here so --help / arg errors don't pay the alviz import cost.
    import matplotlib.pyplot as plt
    from alviz import merge_pdfs

    from jaxrens_al.dashboard import build_iteration_dashboard
    from jaxrens_al.run_layout import (
        evaluated_configs_per_iteration,
        load_progress,
        sampled_configs_per_iteration,
    )

    progress = load_progress(calchome)
    n_iter = len(progress)
    eval_per_iter = evaluated_configs_per_iteration(calchome, progress)
    samp_per_iter = sampled_configs_per_iteration(calchome, progress)

    datapath = Path(args.datapath) if args.datapath else calchome / "dashboard"
    plots_dir = calchome / "plots"

    def _page(index, *, compute):
        dp = datapath / str(index)
        dp.mkdir(parents=True, exist_ok=True)
        return build_iteration_dashboard(
            calchome,
            index,
            datapath=dp,
            eval_files=eval_per_iter[index],
            sampled_files=samp_per_iter[index],
            width_per_panel=args.width,
            height_per_panel=args.height,
            compute=compute,
            force_recompute=args.force,
        )

    if args.dry_run:
        logger.info(f"Dry run: {n_iter} AL iteration(s).")
        for i in range(n_iter):
            board = _page(i, compute=False)
            entries = getattr(board, "_entries", []) if board else []
            names = (
                ", ".join(getattr(e, "name", str(e)) for e in entries)
                or "(no panels)"
            )
            logger.info(f"  iter {i:<3} {len(entries)} panel(s): {names}")
        return 0

    # One PDF page per iteration, then stitch them into the combined output.
    plots_dir.mkdir(parents=True, exist_ok=True)
    page_files = []
    for i in range(n_iter):
        board = _page(i, compute=True)
        if board is None:
            logger.info(f"iter {i}: no panels with data; skipping page.")
            continue
        page = plots_dir / f"Iteration{i}.pdf"
        board.savefig(page)
        plt.close(board._fig)
        page_files.append(page)
        logger.info(f"Wrote {page}")

    if not page_files:
        logger.error("No iteration produced a page; nothing to write.")
        return 1

    merge_pdfs(page_files, output)
    logger.info(f"Wrote combined {output} ({len(page_files)} page(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
