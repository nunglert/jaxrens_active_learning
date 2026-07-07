"""Run a ground-truth (DFT) evaluator over standalone config sets.

The AL loop labels configurations *inside* the orchestrator; sometimes you want
the **same** DFT level applied to a fixed, hand-built set instead -- e.g. an
equation-of-state volume scan or a Materials-Project-derived reference database
for the dashboard's EOS / atomic-environment panels (see
``examples/reference_db/build_reference_db.py``). This is that one-shot path.

It drives a ``remote_handler`` ``GroundTruthEvaluator`` (``VASPEvaluator`` /
``ORCAEvaluator``) purely through its public ``update`` / ``get_evaluated_
configs`` contract, so this module imports **no** ``remote_handler`` (the
evaluator is duck-typed and passed in already configured -- reuse the exact one
from your ``run_al.py`` and the DFT level matches by construction). Only ``ase``
is needed here, and lazily.

Note a ``VASPEvaluator`` rebuilds its outputs from ``vasprun.xml``, so custom
``atoms.info`` on the inputs (``vol_scaling``, ``mp_id``, ...) does **not**
survive labelling. Group configs into separate sets *before* evaluating when a
downstream reader needs them kept apart (the EOS panel wants one file per
phase); geometry-derived quantities like the cell volume do survive.
"""

from __future__ import annotations

from pathlib import Path


def evaluate_config_sets(evaluator, sets, workdir) -> dict[str, Path]:
    """Label several named config sets with one ground-truth evaluator.

    For each ``label -> [ase.Atoms]`` entry, writes the configs to an
    ``unlabelled.extxyz`` under ``workdir/<label>/``, runs the evaluator over
    it (``update`` then ``get_evaluated_configs``), and collects the resulting
    labelled path. Sets share the one evaluator (so one DFT level), and their
    per-label working dirs keep the outputs from colliding.

    Args:
        evaluator: A configured ``remote_handler`` ``GroundTruthEvaluator``
            (duck-typed: ``update(calcdir, evaluator_input)`` +
            ``get_evaluated_configs() -> path``). Reuse the one from your AL
            loop so the DFT settings are identical.
        sets: ``{label: sequence-of-ase.Atoms}``. Empty sets are skipped.
        workdir: Directory the per-label calc dirs are created under.

    Returns:
        ``{label: Path}`` mapping each non-empty set to its labelled extxyz
        (carrying ``REF_energy`` / ``REF_forces``), in input order.
    """
    import ase.io

    workdir = Path(workdir)
    labelled: dict[str, Path] = {}
    for label, configs in sets.items():
        configs = list(configs)
        if not configs:
            continue
        calcdir = workdir / label
        calcdir.mkdir(parents=True, exist_ok=True)
        unlabelled = calcdir / "unlabelled.extxyz"
        ase.io.write(str(unlabelled), configs)
        evaluator.update(calcdir=calcdir, evaluator_input=unlabelled)
        labelled[label] = Path(evaluator.get_evaluated_configs())
    return labelled


def concatenate_configs(paths, output) -> int:
    """Merge several labelled extxyz files into one ``output`` extxyz.

    Used to fold the per-phase EOS scans plus a rattled pool into a single
    reference database for the atomic-environment / energy-volume panels.
    Missing/empty inputs are skipped. Returns the number of frames written.
    """
    import ase.io

    frames = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            frames.extend(ase.io.read(str(p), index=":"))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(output), frames)
    return len(frames)
