"""Read a remote_handler active-learning run's on-disk artifacts.

This is the project's *bridge*. It understands remote_handler's run layout
(the ``calchome`` tree + ``AL_progress.json``) and turns it into plain file
paths that alviz panels consume.

It reads *artifacts only* and deliberately does **not** import
remote_handler: the dashboard side is coupled to the orchestrator's output
*format*, not its code. ``AL_progress.json`` is the source of truth -- the
recorded output paths are used rather than hard-coded directory names.

Two on-disk schemas are supported and normalized to one internal shape (an
ordered per-iteration list of ``{<stage>_output: [rel, ...]}`` dicts):

* **classic** (``ActiveLearningHandler``) -- a list of per-iteration dicts::

      [ {steps, sampler_output, evaluator_output, trainer_output}, ... ]

* **staged** (``StagedActiveLearningHandler``: sampler -> selector ->
  evaluator -> trainer) -- one object whose ``outputs`` is keyed by stage then
  iteration::

      { stages, entry_index, exit_index, num_iterations,
        outputs: { <stage>: { <iter>: [rel, ...] } } }

  Stage names (``sampler`` / ``selector`` / ``evaluator`` / ``trainer``) map to
  the classic ``<stage>_output`` keys, so every accessor below is schema-blind.

Layout::

    calchome/
    ├── 0/  1/  ...                 # per-iteration actor working dirs
    ├── sampler_output/             # full NS dead-point pool (pure producer)
    ├── selector_output/            # candidate configs the selector chose
    ├── evaluator_output/           # DFT-labelled configs (REF_energy/REF_forces)
    ├── trainer_output/             # trained models
    └── AL_progress.json            # classic list OR staged outputs (above)
"""

import json
from pathlib import Path

# Staged-schema stage names -> the classic per-iteration ``<stage>_output`` key
# the rest of this module (and the panels) already understand.
_STAGE_TO_OUTPUT_KEY = {
    "sampler": "sampler_output",
    "selector": "selector_output",
    "evaluator": "evaluator_output",
    "trainer": "trainer_output",
}


def load_progress(calchome) -> list:
    """Load ``AL_progress.json`` as a per-iteration list of dicts.

    Accepts either the classic or the staged schema (see the module docstring)
    and returns the same normalized per-iteration list either way.
    """
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
    """Normalize any supported schema to an ordered per-iteration list."""
    if isinstance(progress, dict):
        if "outputs" in progress:  # staged schema
            return _normalize_staged(progress)
        # A bare dict keyed by iteration index (a tolerated classic variant).
        return [progress[k] for k in sorted(progress, key=lambda x: int(x))]
    return list(progress)


def _normalize_staged(progress) -> list:
    """Turn the staged ``outputs`` map into a per-iteration list of dicts.

    ``outputs`` is ``{stage: {iter: [rel, ...]}}``; this pivots it to one dict
    per iteration keyed by the classic ``<stage>_output`` names, covering
    iterations ``0..max`` that any stage recorded (partial/in-flight runs stop
    at the last completed iteration rather than padding to ``num_iterations``).
    """
    outputs = progress.get("outputs", {}) or {}
    max_iter = -1
    for stage_map in outputs.values():
        for it in stage_map or {}:
            max_iter = max(max_iter, int(it))

    result = []
    for i in range(max_iter + 1):
        entry = {}
        for stage, key in _STAGE_TO_OUTPUT_KEY.items():
            stage_map = outputs.get(stage, {}) or {}
            entry[key] = stage_map.get(str(i), stage_map.get(i, [])) or []
        result.append(entry)
    return result


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


def selected_configs(calchome, progress=None) -> list:
    """Absolute paths to all selector-chosen candidate files, in order.

    The selector stage (staged schema only) thins the sampler's full NS pool
    down to the configs actually sent for evaluation; empty for a classic run.
    """
    if progress is None:
        progress = load_progress(calchome)
    return _collect(calchome, progress, "selector_output")


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


def selected_configs_per_iteration(calchome, progress=None) -> list:
    """List (per iteration) of the selector-chosen paths for that iteration."""
    calchome = Path(calchome)
    if progress is None:
        progress = load_progress(calchome)
    out = []
    for iteration in progress:
        rels = iteration.get("selector_output", []) or []
        out.append([(calchome / rel).resolve() for rel in rels])
    return out


def trained_models_per_iteration(calchome, progress=None) -> list:
    """List (per iteration) of the recorded trained-model paths."""
    calchome = Path(calchome)
    if progress is None:
        progress = load_progress(calchome)
    out = []
    for iteration in progress:
        rels = iteration.get("trainer_output", []) or []
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


def training_mace_logs(calchome) -> list:
    """All MACE ``*_train.txt`` JSON-lines logs across iterations (in order).

    A MACE(-finetuning) trainer writes one per training run; the
    ``MaceTrainingCurve`` panel parses the ``mode == "eval"`` rows out of them.
    """
    out = []
    for it_dir in iteration_dirs(calchome):
        out.extend(training_mace_for_iteration(it_dir))
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


# The trainer actor's working-dir name is a remote_handler convention that has
# not been stable: the staged handler names it ``train``, an older/other layout
# ``training``. Resolve whichever is on disk so every training accessor (and
# the dashboard's per-iteration training panels) works across both.
_TRAINING_DIR_NAMES = ("train", "training")


def training_results_dir(it_dir):
    """This iteration's trainer ``results`` dir (``train``/``training``), or
    None. Prefers whichever exists on disk."""
    for name in _TRAINING_DIR_NAMES:
        base = Path(it_dir) / name / "results"
        if base.exists():
            return base
    return None


def torch_model_for(calchome, model):
    """The torch MACE ``Model.model`` behind a recorded trained model, or None.

    The MACE trainer trains in torch and converts to a mace-jax bundle; the
    recorded ``trainer_output/trainer_it<k>_<j>.bundle`` is the jax side, and
    the torch original stays in iteration ``k``'s trainer tree at
    ``<k>/train/results/calcdirs/<j>/models/Model.model``. Torch-based tools
    (torch-sim relaxations) need that file. A path that already is a ``.model``
    file is returned unchanged.
    """
    import re

    if model is None:
        return None
    model = Path(model)
    if model.suffix == ".model" and model.is_file():
        return model
    match = re.search(r"trainer_it(\d+)_(\d+)\.bundle$", model.name)
    if match is None:
        return None
    k, j = match.groups()
    base = training_results_dir(iteration_dir(calchome, int(k)))
    if base is None:
        return None
    candidate = base / "calcdirs" / j / "models" / "Model.model"
    return candidate if candidate.is_file() else None


def neuralil_full_ensemble_for(model):
    """The every-member ensemble file behind a recorded NeuralIL trainer output.

    With ``save_reduced`` the trainer writes ``model_<k>.pkl`` (the best
    ``save_reduced`` members) and ``full_model_<k>.pkl`` (all ``n_ensemble``
    members) into iteration ``k``'s trainer ``results`` dir. The staged handler
    records only the first, copied to ``<calchome>/trainer_output/
    trainer_it<k>_<j>.pkl``, so the full file is found from that recorded path
    through iteration ``k``'s trainer tree (as :func:`torch_model_for` does for
    MACE). Returns None for any other path, or when the full file is not on
    disk (e.g. a run trained without ``save_reduced``).
    """
    import re

    if model is None:
        return None
    model = Path(model)
    match = re.fullmatch(r"trainer_it(\d+)_\d+\.pkl", model.name)
    if match is None:
        return None
    k = int(match.group(1))
    base = training_results_dir(iteration_dir(model.parent.parent, k))
    if base is None:
        return None
    candidate = base / f"full_model_{k}.pkl"
    return candidate if candidate.is_file() else None


def ns_prefix_from_dir(results_dir, default=None):
    """Infer the jaxrens ``out_file_prefix`` from an NS output directory.

    Reads it off the first ``<prefix>.run*.energies`` (or bare
    ``<prefix>.energies``) found under ``results_dir`` (searched recursively) --
    so callers need not know a MACE run wrote ``mace_sn16`` rather than the
    ``ns`` default. Returns ``default`` when no energies file is present. This
    takes the directory that *directly* holds the energies (what the standalone
    NS dashboard points at); :func:`detect_ns_prefix` wraps it for the AL
    ``ns_sampler/results`` iteration layout.
    """
    base = Path(results_dir)
    if base.exists():
        hits = sorted(base.rglob("*.run*.energies")) or sorted(
            base.rglob("*.energies")
        )
        for f in hits:
            marker = ".run" if ".run" in f.name else ".energies"
            prefix = f.name[: f.name.rfind(marker)]
            if prefix:
                return prefix
    return default


def detect_ns_prefix(it_dir, default=None):
    """Infer the jaxrens ``out_file_prefix`` from an AL iteration's NS output.

    Sniffs the energies under ``<it_dir>/ns_sampler/results`` (the AL working-
    dir convention); returns ``default`` when none is present.
    """
    return ns_prefix_from_dir(Path(it_dir) / "ns_sampler" / "results", default)


def ns_result_dir(it_dir, prefix: str = "ns"):
    """Dir holding this iteration's jaxrens NS energies, or None.

    Returns the *containing* directory (what
    ``MonitorCollection.from_multi_run_directory`` expects), or ``None`` when
    the NS run for this iteration produced no energies file. Matches both the
    multi-run layout (``<prefix>.runNN.energies``) and a bare single-run
    ``<prefix>.energies`` (the one-replica edge case).
    """
    base = Path(it_dir) / "ns_sampler" / "results"
    if not base.exists():
        return None
    hits = sorted(base.rglob(f"{prefix}.run*.energies")) or sorted(
        base.rglob(f"{prefix}.energies")
    )
    return hits[0].parent if hits else None


def adaptation_file(it_dir, prefix: str = "ns"):
    """This iteration's jaxrens ``<prefix>.adaptation.h5`` log, or None."""
    base = Path(it_dir) / "ns_sampler" / "results"
    if not base.exists():
        return None
    hits = sorted(base.rglob(f"{prefix}.adaptation.h5"))
    return hits[0] if hits else None


def acc_rates_file(it_dir, prefix: str = "ns"):
    """This iteration's jaxrens ``<prefix>.acc_rates.h5`` log, or None."""
    base = Path(it_dir) / "ns_sampler" / "results"
    if not base.exists():
        return None
    hits = sorted(base.rglob(f"{prefix}.acc_rates.h5"))
    return hits[0] if hits else None


def re_stats_file(it_dir, prefix: str = "ns"):
    """This iteration's jaxrens ``<prefix>.re_stats.h5`` swap log, or None."""
    base = Path(it_dir) / "ns_sampler" / "results"
    if not base.exists():
        return None
    hits = sorted(base.rglob(f"{prefix}.re_stats.h5"))
    return hits[0] if hits else None


def quests_pca_json(it_dir):
    """This iteration's QUESTS PCA map (``quests_pca.json``), or None.

    Written by a ``QUESTSSelector`` (via ``JaxrensNSSampler``): per-candidate
    PCA coords + a ``selected`` flag. Join to :func:`resampled_provenance_json`
    on ``index`` for the ``(run_id, iter)`` of each selected config.
    """
    p = Path(it_dir) / "ns_sampler" / "results" / "quests_pca.json"
    return p if p.exists() else None


def resampled_provenance_json(it_dir):
    """This iteration's resampling provenance (``resampled_provenance.json``).

    Lists the selected dead points as ``{index, run_id, iter}``. Returns the
    path, or None if the sampler did not write one (e.g. stub runs).
    """
    p = Path(it_dir) / "ns_sampler" / "results" / "resampled_provenance.json"
    return p if p.exists() else None


def training_npz_for_iteration(it_dir) -> list:
    """This iteration's NeuralIL parity ``.npz`` bundles (possibly empty)."""
    base = training_results_dir(it_dir)
    return sorted(base.rglob("*.npz")) if base else []


# jax-loadable model artifacts, best match first: a mace-jax ``.bundle`` dir,
# a ``-jax.pkl`` bundle, then a plain neuralil ``.pkl``.
_MODEL_PATTERNS = ("*.bundle", "*-jax.pkl", "*.pkl")


def _search_model_tree(root):
    """First model artifact under ``root`` (:data:`_MODEL_PATTERNS`), else None."""
    for pattern in _MODEL_PATTERNS:
        hits = sorted(Path(root).rglob(pattern))
        if hits:
            return hits[0]
    return None


def trained_model_for_iteration(it_dir):
    """This iteration's trained-model bundle under the trainer ``results`` dir,
    or None. Used by the EOS / model-parity panels to evaluate the model.
    Returns the first match (:data:`_MODEL_PATTERNS` order) if several exist.
    """
    base = training_results_dir(it_dir)
    return _search_model_tree(base) if base is not None else None


def find_model(results, override=None):
    """Locate the trained-model bundle to re-score under a raw ``results`` dir.

    An explicit ``override`` wins (a ``.bundle`` is a directory, so any existing
    path is accepted). Otherwise search the results tree, then a sibling
    ``trainer_output/`` -- the staged handler writes the model there, not under
    ``<i>/training/results`` -- found by walking up the parents. The raw-dir
    counterpart to :func:`model_for_iteration` (which instead resolves the
    recorded model from ``AL_progress.json``); both share
    :data:`_MODEL_PATTERNS`.
    """
    if override:
        p = Path(override)
        return p if p.exists() else None
    results = Path(results)
    found = _search_model_tree(results)
    if found is not None:
        return found
    for ancestor in results.parents:
        trainer_output = ancestor / "trainer_output"
        if trainer_output.is_dir():
            found = _search_model_tree(trainer_output)
            if found is not None:
                return found
    return None


def model_for_iteration(calchome, index, it_dir=None, progress=None):
    """The model the EOS panel evaluates for iteration ``index``, or None.

    A NeuralIL run writes a ``.pkl`` under the iteration's ``train/results``
    (found by :func:`trained_model_for_iteration`); a MACE-jax run's model is
    the recorded trainer output -- a bundle *directory* under
    ``trainer_output/`` that lives outside the iteration dir, so it is resolved
    from ``AL_progress.json``. Prefer the local ``.pkl`` (unchanged behaviour),
    else fall back to the recorded trainer output for this iteration.
    """
    if it_dir is None:
        it_dir = iteration_dir(calchome, index)
    pkl = trained_model_for_iteration(it_dir)
    if pkl is not None:
        return pkl
    recorded = trained_models_per_iteration(calchome, progress=progress)
    if 0 <= index < len(recorded) and recorded[index]:
        return recorded[index][0]
    return None


def sampling_model_for_iteration(calchome, index, progress=None):
    """The model that DROVE the NS sampler at iteration ``index``, or None.

    In the staged ring the sampler consumes the *previous* iteration's trainer
    output -- the model it injects into ``backend.checkpoint_path`` to generate
    this iteration's pool. So the sampling model is
    ``trained_models_per_iteration[index - 1]``; the sampler runs jaxrens with a
    single member and takes the **last** one (``model_file[-1]``, see
    :class:`~jaxrens_al.run.sampler.JaxrensNSSampler`), so we return that member.

    Iteration 0 is seeded by an out-of-band foundation model that
    ``AL_progress.json`` does not record, so it returns ``None`` (no re-scorable
    sampling model on disk) -- the caller's panel then simply drops out.
    """
    if index <= 0:
        return None
    recorded = trained_models_per_iteration(calchome, progress=progress)
    prev = index - 1
    if 0 <= prev < len(recorded) and recorded[prev]:
        return recorded[prev][-1]
    return None


def training_error_for_iteration(it_dir) -> list:
    """This iteration's NeuralIL ``error_*.txt`` logs (possibly empty)."""
    base = training_results_dir(it_dir)
    return sorted(base.rglob("error_*.txt")) if base else []


def training_mace_for_iteration(it_dir) -> list:
    """This iteration's MACE ``*_train.txt`` logs (possibly empty)."""
    base = training_results_dir(it_dir)
    return sorted(base.rglob("*_train.txt")) if base else []
