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


def sampled_configs_per_iteration(calchome, progress=None) -> list:
    """List (per iteration) of the sampled-config paths for that iteration."""
    calchome = Path(calchome)
    if progress is None:
        progress = load_progress(calchome)
    out = []
    for iteration in progress:
        rels = iteration.get("sampler_output", []) or []
        out.append([(calchome / rel).resolve() for rel in rels])
    return out


# ---------------------------------------------------------------------------
# Per-iteration *working* dirs.
#
# AL_progress.json records only the actors' final output files. The richer
# artifacts some panels need -- the jaxrens NS run (``.energies`` /
# ``.final.checkpoint.h5``) and the NeuralIL training results (parity
# ``.npz`` / ``error_*.txt``) -- live inside each iteration's working tree,
# whose layout is a remote_handler convention:
#
#     calchome/<i>/ns_sampler/results/   (JaxrensNSSampler)
#     calchome/<i>/training/results/     (Neuralil/Mace trainers)
#
# These helpers locate them by that convention and glob defensively (missing
# dirs -> empty), so a panel simply gets nothing to plot when an actor didn't
# run or produced no such files (e.g. stub mode).
# ---------------------------------------------------------------------------


def iteration_dirs(calchome) -> list:
    """Numeric per-iteration working dirs (``calchome/0``, ``1``, ...)."""
    calchome = Path(calchome)
    dirs = [p for p in calchome.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def ns_result_dirs(calchome, prefix: str = "ns") -> list:
    """Per-iteration dirs that contain a jaxrens ``<prefix>.energies`` file.

    Returns the *containing* directory of each energies file (what
    ``jaxrens.postprocess.Monitor.from_directory`` expects), newest last.
    """
    out = []
    for it_dir in iteration_dirs(calchome):
        found = ns_result_dir(it_dir, prefix=prefix)
        if found is not None:
            out.append(found)
    return out


def training_npz_files(calchome) -> list:
    """All NeuralIL parity ``.npz`` files across iterations (in order).

    The NeuralIL trainer writes ``save_data`` npz bundles
    (``e_obs``/``e_pred``/``f_obs``/``f_pred``) under each iteration's
    ``training/results`` tree.
    """
    out = []
    for it_dir in iteration_dirs(calchome):
        out.extend(training_npz_for_iteration(it_dir))
    return out


def training_error_files(calchome) -> list:
    """All NeuralIL ``error_*.txt`` training logs across iterations."""
    out = []
    for it_dir in iteration_dirs(calchome):
        out.extend(training_error_for_iteration(it_dir))
    return out


# ---------------------------------------------------------------------------
# Single-iteration accessors.
#
# The multipage dashboard builds one board per AL iteration, so it needs the
# artifacts of *one* iteration's working dir (``calchome/<i>``) rather than the
# across-iterations collections above. These take an iteration dir and glob
# defensively (missing -> None/empty), so a panel simply gets nothing when the
# actor didn't run for that iteration.
# ---------------------------------------------------------------------------


def iteration_dir(calchome, index: int):
    """Working dir of a single AL iteration (``calchome/<index>``)."""
    return Path(calchome) / str(index)


def ns_result_dir(it_dir, prefix: str = "ns"):
    """Dir holding this iteration's jaxrens ``<prefix>.energies``, or None.

    Returns the *containing* directory (what
    ``jaxrens.postprocess.Monitor.from_directory`` expects), or ``None`` when
    the NS run for this iteration produced no energies file.
    """
    base = Path(it_dir) / "ns_sampler" / "results"
    if not base.exists():
        return None
    hits = sorted(base.rglob(f"{prefix}.energies"))
    return hits[0].parent if hits else None


def training_npz_for_iteration(it_dir) -> list:
    """This iteration's NeuralIL parity ``.npz`` bundles (possibly empty)."""
    base = Path(it_dir) / "training" / "results"
    return sorted(base.rglob("*.npz")) if base.exists() else []


def training_error_for_iteration(it_dir) -> list:
    """This iteration's NeuralIL ``error_*.txt`` logs (possibly empty)."""
    base = Path(it_dir) / "training" / "results"
    return sorted(base.rglob("error_*.txt")) if base.exists() else []
