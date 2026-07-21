"""Render logic for the standalone *section* dashboards (NS / DFT / training).

One definition per section, two callers:

* the ``jaxrens-al dashboard`` CLI dispatches its ``ns`` / ``dft`` /
  ``training`` subcommands here (see :func:`SECTION_COMMANDS`);
* the thin ``scripts/{ns,dft,training}_dashboard.py`` checkout wrappers call the
  same ``run_*`` entry points.

So a section dashboard has a single ``build_*_parser`` (its argparse, built on
the shared :mod:`jaxrens_al.cli_args` groups) and a single ``run_*`` (parse ->
apply a ``dashboard.*`` config -> build the section -> render), whether invoked
through the installed CLI or a checkout script -- mirroring how
:mod:`jaxrens_al.dashboard.sections` unifies the *panel* wiring those same commands feed.

Each command auto-discovers a ``dashboard.{yaml,yml,toml,json}`` beside its
input (or takes ``--config PATH``); CLI flags still win. Pure dashboard-side
(ase + alviz); heavy imports (numpy, alviz, the section builders) are deferred
into the ``run_*`` bodies so ``--help`` and arg errors stay cheap.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from jaxrens_al.cli_args import (
    add_config_arg,
    add_output_args,
    add_processing_args,
    add_ref_key_args,
    add_thermo_args,
    apply_config,
)

logger = logging.getLogger(__name__)

# MACE/jax energies are in eV, so the NS section scans temperature in Kelvin.
# (Pass --k-B 1.0 with a reduced --tmax for a reduced-unit run instead.)
K_BOLTZMANN_EV_PER_K = 8.617333262e-5


def _resolve_section_needs(entries, args, datapath) -> None:
    """Run the cluster jobs a section's panels declare, before rendering.

    Trajectory annotation (Steinhardt) and trained-model evaluation (EOS /
    model-parity, e.g. a GPU-only MACE-jax bundle) each run on ``--cluster``;
    skipped -- and ``remote_handler`` unimported -- when nothing declares them.
    """
    proc = [
        p
        for p in entries
        if hasattr(p, "processing_needs") and p.processing_needs()
    ]
    eval_ = [
        p
        for p in entries
        if hasattr(p, "model_eval_needs") and p.model_eval_needs()
    ]
    phase = [
        p for p in entries if hasattr(p, "phase_needs") and p.phase_needs()
    ]
    if not (proc or eval_ or phase):
        return

    from jaxrens_al.run.processing_resolver import (
        execution_context,
        resolve_model_evaluations,
        resolve_phase_labeling,
        resolve_processing,
    )

    knobs = dict(
        shard_size=args.shard_size,
        num_parallel=args.num_parallel,
        n_processors=args.n_processors,
    )
    print(
        f"Running cluster jobs for the dashboard (cluster="
        f"{args.cluster!r}); may be slow."
    )
    if proc:
        resolve_processing(
            entries,
            **knobs,
            **execution_context(args.cluster, datapath / "processing"),
        )
    if eval_:
        resolve_model_evaluations(
            entries,
            **knobs,
            **execution_context(args.cluster, datapath / "model_eval"),
        )
    if phase:
        from jaxrens_al.cli_args import relax_shard_size

        resolve_phase_labeling(
            entries,
            **{**knobs, "shard_size": relax_shard_size(args)},
            batch_size=args.relax_batch_size,
            **execution_context(args.cluster, datapath / "phase_label"),
        )


def _configure_logging() -> None:
    """Ensure INFO records show when a ``run_*`` is called directly.

    A no-op when the root logger already has handlers (e.g. under the CLI,
    which calls ``basicConfig`` first, or pytest).
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# ==========================================================================
# NS -- a single multi-run jaxrens NS output dir.
# ==========================================================================
def build_ns_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog or "ns_dashboard",
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
        default=None,
        help="jaxrens out_file_prefix of the run files (default: "
        "auto-detected from <prefix>.run*.energies in the output dir, e.g. "
        "'mace_sn16'; falls back to 'ns').",
    )
    add_config_arg(parser)
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
    add_processing_args(parser)
    return parser


def run_ns(argv: list[str] | None = None, prog: str | None = None) -> int:
    _configure_logging()
    parser = build_ns_parser(prog)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    if not out_dir.is_dir():
        print(f"Not a directory: {out_dir}", file=sys.stderr)
        return 1

    try:
        args = apply_config(
            parser,
            argv,
            args,
            logger,
            # The PDF defaults one dir above the NS output dir, so a config
            # dropped beside *either* the input or the output is discovered.
            run_dir=[out_dir, out_dir.parent],
            positional="output_dir",
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error(str(exc))
        return 1

    # Prefix: honour --prefix / config, else sniff it off the output dir's
    # <prefix>.run*.energies (a MACE run wrote 'mace_sn16', not 'ns').
    if args.prefix is None:
        from jaxrens_al.dashboard.run_layout import ns_prefix_from_dir

        args.prefix = ns_prefix_from_dir(out_dir, default="ns")
        logger.info(f"NS out_file_prefix: {args.prefix!r} (auto-detected).")

    energies = sorted(out_dir.glob(f"{args.prefix}.run*.energies"))
    if not energies:
        print(
            f"No {args.prefix}.run*.energies under {out_dir} -- wrong "
            f"--prefix or not a multi-run NS output dir?",
            file=sys.stderr,
        )
        return 1

    import numpy as np
    from alviz import Dashboard

    from jaxrens_al.dashboard.panel_specs import parse_panel_config
    from jaxrens_al.dashboard.sections import ns_section

    # Default the PDF (and its data cache) one dir *above* the NS output dir,
    # so the artifact lands beside the run rather than buried in the deep
    # ``.../ns_sampler/results/output`` leaf it reads from.
    output = (
        Path(args.output)
        if args.output
        else out_dir.parent / "ns_dashboard.pdf"
    )
    datapath = (
        Path(args.datapath)
        if args.datapath
        else out_dir.parent / "ns_dashboard_data"
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
        panel_config=parse_panel_config(
            getattr(args, "panel_config_raw", None)
        ),
        **common,
    )

    _resolve_section_needs(entries, args, datapath)

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


# ==========================================================================
# DFT -- a single set of DFT-labelled configs.
# ==========================================================================
def build_dft_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog or "dft_dashboard",
        description="Dashboard for a single set of DFT-labelled configs "
        "(extxyz with REF_energy / REF_forces).",
    )
    parser.add_argument(
        "config_files",
        nargs="+",
        help="One or more DFT-labelled extxyz files (REF_energy / "
        "REF_forces). Several are pooled into one set.",
    )
    add_config_arg(parser)
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


def run_dft(argv: list[str] | None = None, prog: str | None = None) -> int:
    _configure_logging()
    parser = build_dft_parser(prog)
    args = parser.parse_args(argv)

    files = [Path(f) for f in args.config_files]
    missing = [f for f in files if not f.is_file()]
    if missing:
        print(
            "No such file(s): " + ", ".join(str(f) for f in missing),
            file=sys.stderr,
        )
        return 1

    try:
        args = apply_config(
            parser,
            argv,
            args,
            logger,
            run_dir=files[0].parent,
            positional="config_files",
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error(str(exc))
        return 1

    from alviz import Dashboard

    from jaxrens_al.dashboard.panel_specs import parse_panel_config
    from jaxrens_al.dashboard.sections import dft_section

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
        panel_config=parse_panel_config(
            getattr(args, "panel_config_raw", None)
        ),
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


# ==========================================================================
# Training -- a single trainer 'results' directory (MACE or NeuralIL).
# ==========================================================================
def build_training_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog or "training_dashboard",
        description="Dashboard for a single training run (a trainer's "
        "'results' directory). Auto-detects MACE vs NeuralIL.",
    )
    parser.add_argument(
        "results_dir",
        help="Training results directory (searched recursively for "
        "*_train.txt / error_*.txt / *.npz / db_train.extxyz).",
    )
    add_config_arg(parser)
    add_output_args(parser)
    add_ref_key_args(parser)
    add_processing_args(parser)
    parser.add_argument(
        "--model",
        default=None,
        help="Trained-model bundle to re-score for the MACE parity panel "
        "(a mace-jax .bundle dir or *.pkl; default: auto-detect under "
        "results_dir, then a sibling trainer_output/).",
    )
    parser.add_argument(
        "--valid-set",
        default=None,
        help="Labelled extxyz to re-score for the model parity "
        "(default: auto-detect db_val*.extxyz, else db_train.extxyz).",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="jaxrens backend to load --model with (default: 'mace' for a "
        "MACE run, else 'neuralil').",
    )
    parser.add_argument(
        "--remote-model-eval",
        action="store_true",
        default=False,
        help="Evaluate the model on --cluster instead of in-process "
        "(required for a GPU-only MACE-jax bundle).",
    )
    parser.add_argument(
        "--no-forces",
        action="store_true",
        help="Model parity: energies only (skip the forces subplot).",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=50,
        help="Static neighbor-buffer size for the backend call (default: 50).",
    )
    parser.add_argument(
        "--no-parity",
        action="store_true",
        help="Skip the trained-model parity panel.",
    )
    return parser


def run_training(
    argv: list[str] | None = None, prog: str | None = None
) -> int:
    _configure_logging()
    parser = build_training_parser(prog)
    args = parser.parse_args(argv)

    results = Path(args.results_dir)
    if not results.is_dir():
        print(f"Not a directory: {results}", file=sys.stderr)
        return 1

    try:
        args = apply_config(
            parser,
            argv,
            args,
            logger,
            run_dir=results,
            positional="results_dir",
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error(str(exc))
        return 1

    from alviz import Dashboard

    from jaxrens_al.dashboard.panel_specs import parse_panel_config
    from jaxrens_al.dashboard.sections import training_section

    output = (
        Path(args.output)
        if args.output
        else results / "training_dashboard.pdf"
    )
    datapath = (
        Path(args.datapath)
        if args.datapath
        else results / "training_dashboard_data"
    )
    datapath.mkdir(parents=True, exist_ok=True)
    common = dict(datapath=datapath, force_recompute=args.force)

    entries = training_section(
        results,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        model=args.model,
        valid_set=args.valid_set,
        backend=args.backend,
        remote_model_eval=args.remote_model_eval,
        show_forces=not args.no_forces,
        max_neighbors=args.max_neighbors,
        include_parity=not args.no_parity,
        panel_config=parse_panel_config(
            getattr(args, "panel_config_raw", None)
        ),
        **common,
    )
    if not entries:
        print(
            f"No training logs under {results} -- expected *_train.txt "
            f"(MACE) or error_*.txt / *.npz (NeuralIL).",
            file=sys.stderr,
        )
        return 1
    kind = "MACE" if sorted(results.rglob("*_train.txt")) else "NeuralIL"

    _resolve_section_needs(entries, args, datapath)

    board = Dashboard.from_entries(
        entries, ncols=2, width_per_panel=5.0, height_per_panel=4.0
    )
    board.render(suptitle=f"{kind} training: {results}")
    board.savefig(output)
    print(f"Wrote {output} ({kind}, {len(entries)} panel(s)).")
    return 0


# The ``jaxrens-al dashboard <section>`` subcommands, dispatched from
# jaxrens_al.cli:main. Each closes over its display prog so ``<section> --help``
# reads ``jaxrens-al dashboard <section>``.
SECTION_COMMANDS = {
    "ns": lambda argv: run_ns(argv, prog="jaxrens-al dashboard ns"),
    "dft": lambda argv: run_dft(argv, prog="jaxrens-al dashboard dft"),
    "training": lambda argv: run_training(
        argv, prog="jaxrens-al dashboard training"
    ),
}
