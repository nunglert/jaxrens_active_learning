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
from alviz.panels import ExtxyzSampledEnergyVolume

from jaxrens_al.panels import (
    AtomicEnvsPanel,
    MaceTrainingCurve,
    ModelParityPanel,
    NeuralILParityPanel,
    NeuralILTrainingCurve,
)


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
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PDF (default: <results_dir>/training_dashboard.pdf).",
    )
    parser.add_argument(
        "--datapath",
        default=None,
        help="Where panels cache computed JSON "
        "(default: <results_dir>/training_dashboard_data).",
    )
    parser.add_argument(
        "--energy-key",
        default="REF_energy",
        help="info key of the reference (DFT) energy (default: REF_energy).",
    )
    parser.add_argument(
        "--forces-key",
        default="REF_forces",
        help="array key of the reference (DFT) forces (default: REF_forces).",
    )
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
    parser.add_argument("--force", action="store_true", help="Recompute.")
    return parser


# jax-loadable model artifacts, best match first: a mace-jax ``.bundle`` dir
# (``config.json`` + ``params.msgpack``), a ``*-jax.pkl`` / ``*.pkl`` bundle.
_MODEL_PATTERNS = ("*.bundle", "*-jax.pkl", "*.pkl")


def _find_model(results: Path, override: str | None):
    """Locate the trained-model bundle to re-score (or None).

    An explicit ``--model`` wins (a ``.bundle`` is a directory, so we accept
    any existing path). Otherwise search the results tree first, then a sibling
    ``trainer_output/`` -- the staged handler writes the trained model there,
    not under ``<i>/train/results`` -- found by walking up the parents.
    """
    if override:
        p = Path(override)
        return p if p.exists() else None

    def _search(root: Path):
        for pattern in _MODEL_PATTERNS:
            hits = sorted(root.rglob(pattern))
            if hits:
                return hits[0]
        return None

    found = _search(results)
    if found is not None:
        return found
    for ancestor in results.parents:
        trainer_output = ancestor / "trainer_output"
        if trainer_output.is_dir():
            found = _search(trainer_output)
            if found is not None:
                return found
    return None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    results = Path(args.results_dir)
    if not results.is_dir():
        print(f"Not a directory: {results}", file=sys.stderr)
        return 1

    mace_logs = sorted(results.rglob("*_train.txt"))
    error_logs = sorted(results.rglob("error_*.txt"))
    npz_files = sorted(results.rglob("*.npz"))
    train_sets = sorted(results.rglob("db_train.extxyz"))
    # Validation split to re-score for the model parity: prefer db_val*, then
    # the training set. MACE's split_db writes db_val.extxyz.
    valid_sets = (
        sorted(results.rglob("db_val.extxyz"))
        or sorted(results.rglob("db_valid.extxyz"))
        or train_sets
    )

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

    entries = []
    # ``has_parity`` tracks whether a trained-model parity panel is already
    # covered (NeuralIL reads it from .npz); MACE has none, so we re-score.
    has_parity = False
    if mace_logs:
        kind, default_backend = "MACE", "mace"
        entries.append(MaceTrainingCurve(train_logs=mace_logs, **common))
    elif error_logs or npz_files:
        kind, default_backend = "NeuralIL", "neuralil"
        if error_logs:
            entries.append(
                NeuralILTrainingCurve(error_files=error_logs, **common)
            )
        if npz_files:
            entries.append(NeuralILParityPanel(npz_files=npz_files, **common))
            has_parity = True
    else:
        print(
            f"No training logs under {results} -- expected *_train.txt "
            f"(MACE) or error_*.txt / *.npz (NeuralIL).",
            file=sys.stderr,
        )
        return 1

    # Trained-model parity by re-scoring a labelled set (mainly MACE, which
    # writes no .npz bundle). Auto-detect the model bundle and backend.
    if not has_parity and not args.no_parity and valid_sets:
        model_file = _find_model(results, args.model)
        if model_file is None:
            print(
                f"No model bundle found under {results} for the parity panel "
                "(pass --model, or --no-parity to skip).",
                file=sys.stderr,
            )
        else:
            entries.append(
                ModelParityPanel(
                    config_files=valid_sets,
                    model_file=model_file,
                    backend_type=args.backend or default_backend,
                    energy_ref_key=args.energy_key,
                    forces_ref_key=args.forces_key,
                    show_forces=not args.no_forces,
                    max_neighbors=args.max_neighbors,
                    **common,
                )
            )

    if train_sets:
        # Energy/volume spread + descriptor diversity of what it trained on.
        entries += [
            ExtxyzSampledEnergyVolume(
                config_files=train_sets,
                energy_key=args.energy_key,
                forces_key=args.forces_key,
                **common,
            ),
            AtomicEnvsPanel(config_files=train_sets, **common),
        ]

    board = Dashboard.from_entries(
        entries, ncols=2, width_per_panel=5.0, height_per_panel=4.0
    )
    board.render(suptitle=f"{kind} training: {results}")
    board.savefig(output)
    print(f"Wrote {output} ({kind}, {len(entries)} panel(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
