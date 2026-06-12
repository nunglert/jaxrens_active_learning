"""Read a remote_handler active-learning run's on-disk artifacts.

This is the project's *bridge*. It understands remote_handler's run layout
(the ``calchome`` tree + ``AL_progress.json``) and turns it into plain file
paths that alviz panels consume.

It reads *artifacts only* and deliberately does **not** import
remote_handler: the dashboard side is coupled to the orchestrator's output
*format*, not its code. ``AL_progress.json`` is the source of truth -- the
recorded output paths are used rather than hard-coded directory names.

Layout (from remote_handler's ActiveLearningHandler)::

    calchome/
    ├── 0/  1/  ...                 # per-iteration actor working dirs
    ├── evaluator_output/           # DFT-labelled configs (REF_energy/REF_forces)
    ├── sampler_output/             # candidate configs
    ├── trainer_output/             # trained models
    └── AL_progress.json            # [ {steps, sampler_output,
                                    #     evaluator_output, trainer_output}, ... ]
"""

import json
from pathlib import Path


def load_progress(calchome) -> list:
    """Load ``AL_progress.json`` as a per-iteration list of dicts."""
    calchome = Path(calchome)
    progress_file = calchome / "AL_progress.json"
    if not progress_file.exists():
        raise FileNotFoundError(
            f"No AL_progress.json found in {calchome}. Is this a "
            "remote_handler run directory?"
        )
    with open(progress_file, "r") as f:
        progress = json.load(f)
    return _as_iteration_list(progress)


def _as_iteration_list(progress) -> list:
    """Normalize to an ordered list (tolerate a dict keyed by iteration)."""
    if isinstance(progress, dict):
        return [progress[k] for k in sorted(progress, key=lambda x: int(x))]
    return list(progress)


def _collect(calchome, progress, key) -> list:
    calchome = Path(calchome)
    paths = []
    for iteration in progress:
        for rel in iteration.get(key, []) or []:
            paths.append((calchome / rel).resolve())
    return paths


def evaluated_configs(calchome, progress=None) -> list:
    """Absolute paths to all DFT-evaluated config files, in order."""
    if progress is None:
        progress = load_progress(calchome)
    return _collect(calchome, progress, "evaluator_output")


def sampled_configs(calchome, progress=None) -> list:
    """Absolute paths to all sampled (candidate) config files, in order."""
    if progress is None:
        progress = load_progress(calchome)
    return _collect(calchome, progress, "sampler_output")


def trained_models(calchome, progress=None) -> list:
    """Absolute paths to all trained-model files, in order."""
    if progress is None:
        progress = load_progress(calchome)
    return _collect(calchome, progress, "trainer_output")


def evaluated_configs_per_iteration(calchome, progress=None) -> list:
    """List (per iteration) of the evaluated-config paths for that iteration."""
    calchome = Path(calchome)
    if progress is None:
        progress = load_progress(calchome)
    out = []
    for iteration in progress:
        rels = iteration.get("evaluator_output", []) or []
        out.append([(calchome / rel).resolve() for rel in rels])
    return out
