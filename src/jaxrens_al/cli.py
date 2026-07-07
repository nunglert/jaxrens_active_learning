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
import tempfile
from pathlib import Path

from jaxrens_al.cli_args import (
    add_config_arg,
    add_output_args,
    add_processing_args,
    add_steinhardt_args,
    add_thermo_args,
    apply_config,
    dump_config,
    steinhardt_params,
)

logger = logging.getLogger(__name__)


def _run_init_config(argv: list[str] | None) -> int:
    """``init-config <flavor>`` -- write a starter config for a parser flavor.

    Builds the chosen flavor's parser (the same one its render path uses) and
    dumps every option's default as a commented YAML template via
    :func:`jaxrens_al.cli_args.dump_config` -- to ``-o PATH`` or stdout.
    """
    p = argparse.ArgumentParser(
        prog="jaxrens-al-dashboard init-config",
        description="Write a starter dashboard config (all default flag "
        "values, commented) for a dashboard flavor. Redirect or -o it to "
        "dashboard.yaml beside the run/artifact; CLI flags still override it.",
    )
    p.add_argument(
        "flavor",
        choices=("al", "ns", "dft", "training"),
        help="Which dashboard's options to scaffold: 'al' (a full AL run) or a "
        "section (ns/dft/training).",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the template here (default: stdout).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = p.parse_args(argv)

    if args.flavor == "al":
        parser = _build_parser()
        title = "jaxrens-al-dashboard"
    else:
        from jaxrens_al.section_dashboards import (
            build_dft_parser,
            build_ns_parser,
            build_training_parser,
        )

        builder = {
            "ns": build_ns_parser,
            "dft": build_dft_parser,
            "training": build_training_parser,
        }[args.flavor]
        parser = builder(prog=f"jaxrens-al-dashboard {args.flavor}")
        title = f"jaxrens-al-dashboard {args.flavor}"

    text = dump_config(parser, title=title)
    if args.output:
        out = Path(args.output)
        if out.exists() and not args.force:
            logger.error(f"{out} already exists; pass --force to overwrite.")
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        logger.info(f"Wrote {out}")
    else:
        print(text, end="")
    return 0


def _clear_file_logging() -> None:
    """Drop any dashboard log-file handler a previous ``main`` call added.

    ``main`` may run several times in one process (tests, a batch driver);
    stripping our own marked handler up front keeps them from accumulating and
    stops an earlier run's handler catching a later run's logs.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_jaxrens_dashboard_log", False):
            root.removeHandler(h)
            h.close()


def _setup_file_logging(log_file, calchome):
    """Tee INFO logs to ``log_file``; return the resolved path.

    ``True`` (a config ``log-file: true``) or a bare ``--log-file`` means
    ``<calchome>/dashboard.log``; a relative path resolves under the calchome.
    If the target already exists it is left untouched and a timestamped sibling
    (``dashboard_YYYY-MM-DD_HH-MM-SS.log``, like the loop logs) is written
    instead, so a re-render never clobbers an earlier run's log.
    """
    if log_file is True:
        log_file = "dashboard.log"
    path = Path(log_file)
    if not path.is_absolute():
        path = Path(calchome) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        import datetime

        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    handler = logging.FileHandler(path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler._jaxrens_dashboard_log = True
    logging.getLogger().addHandler(handler)
    # Ensure INFO records actually reach the handler: basicConfig above is a
    # no-op when the root already has handlers (e.g. under pytest), leaving the
    # root at WARNING. Bump the loggers whose output belongs in the run log.
    for name in ("jaxrens_al", "alviz"):
        lg = logging.getLogger(name)
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
    return path


# First-token subcommands that render a single raw artifact set instead of a
# whole AL run. Dispatched in main() before the AL parser so bare
# ``jaxrens-al-dashboard <calchome>`` stays the (backward-compatible) default.
_SECTION_COMMANDS = ("ns", "dft", "training")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jaxrens-al-dashboard",
        description="Build an alviz dashboard PDF for a remote_handler "
        "active-learning run (a 'calchome' directory holding "
        "AL_progress.json).",
        epilog="Section subcommands render a single raw artifact set (no "
        "AL_progress.json needed):\n"
        "  jaxrens-al-dashboard ns       <ns_output_dir>\n"
        "  jaxrens-al-dashboard dft      <labelled.extxyz ...>\n"
        "  jaxrens-al-dashboard training <results_dir>\n"
        "Run 'jaxrens-al-dashboard <section> --help' for each.\n"
        "Scaffold a config:\n"
        "  jaxrens-al-dashboard init-config <al|ns|dft|training> [-o PATH]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path of the AL run (the calchome directory).",
    )
    add_config_arg(parser)
    parser.add_argument(
        "--log-file",
        nargs="?",
        const="dashboard.log",
        default=None,
        metavar="PATH",
        help="Also write logs to a file (a relative path resolves under the "
        "calchome). Bare --log-file uses <calchome>/dashboard.log. Config "
        "key: log-file.",
    )
    add_output_args(
        parser, force_help="Replot even if the output PDF already exists."
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
        "--dry-run",
        action="store_true",
        help="Assemble and report the panel layout without "
        "computing/plotting or writing a PDF.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="PANEL",
        help="Render only these panels (by name), e.g. --only "
        "sampled_energy_volume neuralil_parity.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        metavar="PANEL",
        help="Skip these panels (by name).",
    )
    parser.add_argument(
        "--eos-reference",
        nargs="+",
        default=None,
        metavar="EXTXYZ",
        help="Reference volume-scan file(s) for the equation-of-state panel, "
        "one per phase; each iteration's model is evaluated on them.",
    )
    parser.add_argument(
        "--eos-backend",
        type=str,
        default="neuralil",
        help="jaxrens backend used to load the model for the EOS panel "
        "(default: neuralil).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="jaxrens out_file_prefix of the NS run files (default: "
        "auto-detected from the first iteration's <prefix>.run*.energies, "
        "e.g. 'mace_sn16'; falls back to 'ns').",
    )
    add_thermo_args(parser)  # AL defaults: reduced units, T on x
    parser.add_argument(
        "--sections",
        action="store_true",
        help="Also emit, per iteration, the standalone NS / DFT / training "
        "'section' pages (the same panels scripts/{ns,dft,training}_dashboard "
        "render) built from each iteration's raw working dir -- including "
        "iterations AL_progress.json never recorded (a crashed loop).",
    )
    add_steinhardt_args(parser)
    add_processing_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if argv is None:
        argv = sys.argv[1:]
    # Section subcommands (ns/dft/training) render a single raw artifact set;
    # dispatch to jaxrens_al.section_dashboards. Bare ``<calchome>`` (no known
    # subcommand as the first token) falls through to the AL run flow below.
    if argv and argv[0] in _SECTION_COMMANDS:
        from jaxrens_al.section_dashboards import SECTION_COMMANDS

        return SECTION_COMMANDS[argv[0]](argv[1:])
    # ``init-config <flavor>`` scaffolds a starter dashboard config (all default
    # flag values, commented) for the chosen flavor's parser.
    if argv and argv[0] == "init-config":
        return _run_init_config(argv[1:])
    _clear_file_logging()  # drop any log handler a previous run left attached
    parser = _build_parser()
    args = parser.parse_args(argv)

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

    # A dashboard.* config beside the run (or --config) supplies flag defaults;
    # CLI flags still win. Applied before the output/skip logic so a config can
    # set --output / --force too.
    try:
        args = apply_config(parser, argv, args, logger)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error(str(exc))
        return 1

    output = Path(args.output) if args.output else calchome / "dashboard.pdf"

    if output.exists() and not args.force and not args.dry_run:
        logger.info(
            f"{output} already exists; skipping (pass --force to replot)."
        )
        return 0

    # Tee logs to a file once we know we'll do real work (after the skip check,
    # so a no-op run doesn't truncate an existing log). Captures the dry-run
    # report and the full render.
    if args.log_file:
        logger.info(
            f"Logging to {_setup_file_logging(args.log_file, calchome)}"
        )

    # Import here so --help / arg errors don't pay the alviz import cost.
    import matplotlib.pyplot as plt
    import numpy as np
    from alviz import Dashboard, merge_pdfs

    from jaxrens_al.dashboard import build_iteration_dashboard
    from jaxrens_al.run_layout import (
        detect_ns_prefix,
        evaluated_configs_per_iteration,
        iteration_dir,
        iteration_dirs,
        load_progress,
        ns_result_dir,
        sampled_configs_per_iteration,
        training_results_dir,
    )
    from jaxrens_al.sections import dft_section, ns_section, training_section

    progress = load_progress(calchome)
    n_iter = len(progress)
    eval_per_iter = evaluated_configs_per_iteration(calchome, progress)
    samp_per_iter = sampled_configs_per_iteration(calchome, progress)

    datapath = Path(args.datapath) if args.datapath else calchome / "dashboard"
    plots_dir = calchome / "plots"

    # jaxrens out_file_prefix: honour --prefix, else sniff it off the first
    # iteration's NS energies (a MACE run writes 'mace_sn16', not 'ns').
    it_dirs = iteration_dirs(calchome)
    ns_prefix = args.prefix
    if ns_prefix is None:
        ns_prefix = next(
            (p for p in (detect_ns_prefix(d) for d in it_dirs) if p), "ns"
        )
        logger.info(f"NS out_file_prefix: {ns_prefix!r} (auto-detected).")

    T = np.linspace(args.tmin, args.tmax, args.n_T)
    steinhardt = steinhardt_params(args)

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
            only=args.only,
            exclude=args.exclude,
            eos_reference=args.eos_reference,
            eos_backend=args.eos_backend,
            ns_prefix=ns_prefix,
            ns_swap_axes=args.swap_pt,
            ns_k_B=args.k_B,
            ns_temperatures=T,
            steinhardt=steinhardt,
        )

    def _section_board(entries):
        """Wrap raw-dir section entries in a 2-col board (or None if empty)."""
        if not entries:
            return None
        return Dashboard.from_entries(
            entries,
            ncols=2,
            width_per_panel=args.width,
            height_per_panel=args.height,
            only=args.only,
            exclude=args.exclude,
        )

    def _section_pages(index):
        """The NS / training / DFT section boards for one iteration on disk.

        Built from that iteration's raw working dir via the shared section
        builders -- so an iteration AL_progress.json never recorded (a crashed
        loop) is still rendered. Yields ``(stem, board, title)`` tuples; the
        caller chooses where each page is written (section pages are ephemeral
        -- they are merged into the combined PDF but not kept in ``plots/``,
        since those panels already live on the global dashboard).
        """
        it_dir = iteration_dir(calchome, index)
        base = datapath / str(index)

        def _sec(kind, entries):
            board = _section_board(entries)
            if board is not None:
                yield (
                    f"Iteration{index}_{kind}",
                    board,
                    f"AL iteration {index} -- {kind}",
                )

        # Per-iteration prefix (a MACE run writes 'mace_sn16'); fall back to
        # the run-level one sniffed above.
        prefix = detect_ns_prefix(it_dir, default=ns_prefix)
        out = ns_result_dir(it_dir, prefix=prefix)
        if out is not None:
            dp = base / "ns_section"
            dp.mkdir(parents=True, exist_ok=True)
            yield from _sec(
                "NS",
                ns_section(
                    out,
                    prefix=prefix,
                    k_B=args.k_B,
                    temperatures=T,
                    swap_axes=args.swap_pt,
                    steinhardt=steinhardt,
                    datapath=dp,
                    force_recompute=args.force,
                ),
            )

        results = training_results_dir(it_dir)
        if results is not None:
            dp = base / "training_section"
            dp.mkdir(parents=True, exist_ok=True)
            yield from _sec(
                "Training",
                training_section(
                    results,
                    backend=args.eos_backend,
                    datapath=dp,
                    force_recompute=args.force,
                ),
            )

        eval_files = eval_per_iter[index] if index < len(eval_per_iter) else []
        if eval_files:
            dp = base / "dft_section"
            dp.mkdir(parents=True, exist_ok=True)
            yield from _sec(
                "DFT",
                dft_section(
                    eval_files, datapath=dp, force_recompute=args.force
                ),
            )

    if args.dry_run:
        logger.info(f"Dry run: {n_iter} AL iteration(s).")
        for i in range(n_iter):
            board = _page(i, compute=False)
            entries = board.entries if board else []
            names = (
                ", ".join(getattr(e, "name", str(e)) for e in entries)
                or "(no panels)"
            )
            logger.info(f"  iter {i:<3} {len(entries)} panel(s): {names}")
        if args.sections:
            for it_dir in iteration_dirs(calchome):
                i = int(it_dir.name)
                for _stem, board, title in _section_pages(i):
                    logger.info(
                        f"  section {title}: {len(board.entries)} panel(s)"
                    )
        return 0

    # Build every page (no compute yet), so a single processing pass can
    # annotate all trajectories the panels need before anything renders. AL
    # iteration pages persist under plots/; section pages render into a temp
    # dir -- they are merged into the combined PDF but not kept (their panels
    # already live on the per-iteration AL pages).
    plots_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as sec_tmp:
        sec_dir = Path(sec_tmp)
        pages = []  # (page_path, board, suptitle)
        for i in range(n_iter):
            board = _page(i, compute=False)
            if board is None:
                logger.info(f"iter {i}: no panels with data; skipping page.")
                continue
            pages.append(
                (plots_dir / f"Iteration{i}.pdf", board, f"AL iteration {i}")
            )
        if args.sections:
            for it_dir in iteration_dirs(calchome):
                for stem, board, title in _section_pages(int(it_dir.name)):
                    pages.append((sec_dir / f"{stem}.pdf", board, title))

        if not pages:
            logger.error("No iteration produced a page; nothing to write.")
            return 1

        # One processing job over every panel that declares a need (e.g. the
        # Steinhardt heatmaps), then render. Skipped -- and remote_handler
        # unimported -- when nothing needs annotation.
        all_entries = [e for _, board, _ in pages for e in board.entries]
        if any(
            hasattr(e, "processing_needs") and e.processing_needs()
            for e in all_entries
        ):
            from jaxrens_al.processing_resolver import (
                execution_context,
                resolve_processing,
            )

            logger.info(
                f"Annotating trajectories via the processing machinery "
                f"(cluster={args.cluster!r}); may be slow."
            )
            resolve_processing(
                all_entries,
                shard_size=args.shard_size,
                num_parallel=args.num_parallel,
                n_processors=args.n_processors,
                **execution_context(args.cluster, datapath / "processing"),
            )

        page_files = []
        for page, board, suptitle in pages:
            board.render(suptitle=suptitle)
            board.savefig(page)
            plt.close(board._fig)
            page_files.append(page)
            if page.parent == plots_dir:
                logger.info(f"Wrote {page}")

        merge_pdfs(page_files, output)
    logger.info(f"Wrote combined {output} ({len(page_files)} page(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
