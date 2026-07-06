"""Render a dashboard for a *single* training run (no AL_progress.json).

The ``jaxrens-al-dashboard`` CLI only renders iterations that
``AL_progress.json`` records as *finished*. When a loop trains a model but then
dies before completing the iteration (e.g. the readback fails), that training
is invisible to the AL dashboard even though its logs and model are on disk.

This script wires the *same* project panels the AL dashboard uses directly onto
one training ``results`` directory, so there is no panel logic here, only
wiring. It auto-detects the trainer by what is on disk (recursively):

* **MACE** (``*_train.txt`` JSON-lines log) -> :class:`MaceTrainingCurve`.
* **NeuralIL** (``error_*.txt`` logs and ``save_data`` ``.npz`` bundles)
  -> :class:`NeuralILTrainingCurve` + :class:`NeuralILParityPanel`.

Parity of the *trained model* against the DFT reference is shown too:

* **NeuralIL** reads the trainer's ``.npz`` bundle directly
  (:class:`NeuralILParityPanel`).
* **MACE** writes no such bundle, so :class:`ModelParityPanel` *re-scores* a
  labelled set (the validation split ``db_val.extxyz``, else ``db_train``) by
  loading the trained model through the jaxrens backend -- this is a real
  predicted-vs-reference parity, not the sampler-surrogate parity that
  ``dft_dashboard`` shows.

The model bundle and validation set are auto-detected under ``results_dir``;
override with ``--model`` / ``--valid-set`` / ``--backend``. If a
``db_train.extxyz`` training set is present, its energy/volume spread and
descriptor diversity are shown as well. Point at the training results dir,
e.g.::

    ~/miniconda3/envs/jaxrens_al/bin/python scripts/training_dashboard.py \
        experiments/run6/calchome/phase1_mace/0/train/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alviz import Dashboard

from jaxrens_al.cli_args import add_output_args, add_ref_key_args
from jaxrens_al.sections import training_section


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training_dashboard",
        description="Dashboard for a single training run (a trainer's "
        "'results' directory). Auto-detects MACE vs NeuralIL.",
    )
    parser.add_argument(
        "results_dir",
        help="Training results directory (searched recursively for "
        "*_train.txt / error_*.txt / *.npz / db_train.extxyz).",
    )
    add_output_args(parser)
    add_ref_key_args(parser)
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    results = Path(args.results_dir)
    if not results.is_dir():
        print(f"Not a directory: {results}", file=sys.stderr)
        return 1

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
        show_forces=not args.no_forces,
        max_neighbors=args.max_neighbors,
        include_parity=not args.no_parity,
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

    board = Dashboard.from_entries(
        entries, ncols=2, width_per_panel=5.0, height_per_panel=4.0
    )
    board.render(suptitle=f"{kind} training: {results}")
    board.savefig(output)
    print(f"Wrote {output} ({kind}, {len(entries)} panel(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
