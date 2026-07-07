"""Raw-directory dashboard *sections* -- wiring the standalone scripts share.

``scripts/{ns,dft,training}_dashboard.py`` each render a *single* on-disk
artifact set (an NS output dir, one set of DFT-labelled configs, a training
``results`` dir) with the same project/alviz panels the AL dashboard uses --
so a crashed loop, whose iteration ``AL_progress.json`` never recorded, is
still inspectable. That per-artifact panel wiring lived duplicated in each
script; it lives here now, once, as three ``*_section`` builders returning a
``list`` of alviz entries.

Two consumers share them:

* the standalone scripts, now thin arg-parsing wrappers over these builders;
* the global ``jaxrens-al-dashboard`` CLI's ``--sections`` mode, which points
  them at each iteration's raw working dir (``calchome/<i>/...``) so the
  combined report includes the same per-artifact detail -- including iterations
  ``AL_progress.json`` skipped.

These are **dashboard-side**: ase + alviz only, no ``remote_handler`` import.
The Steinhardt heatmap a section may emit is a pure reader that merely
*declares* a processing need; running that need is the caller's job (via
:func:`jaxrens_al.processing_resolver.resolve_processing`), off this import
path.
"""

from __future__ import annotations

from pathlib import Path

from alviz.panels import ExtxyzSampledEnergyVolume

from jaxrens_al.keys import (
    REF_ENERGY_KEY,
    REF_FORCES_KEY,
    SURROGATE_ENERGY_KEY,
)
from jaxrens_al.panels import (
    AtomicEnvsPanel,
    MaceTrainingCurve,
    ModelParityPanel,
    NeuralILParityPanel,
    NeuralILTrainingCurve,
)

# jax-loadable model artifacts, best match first: a mace-jax ``.bundle`` dir
# (``config.json`` + ``params.msgpack``), a ``*-jax.pkl`` / ``*.pkl`` bundle.
_MODEL_PATTERNS = ("*.bundle", "*-jax.pkl", "*.pkl")


# ---------------------------------------------------------------------------
# NS section (from scripts/ns_dashboard.py).
# ---------------------------------------------------------------------------
def ns_section(
    out_dir,
    *,
    prefix: str = "ns",
    k_B: float = 1.0,
    temperatures=None,
    swap_axes: bool = True,
    per_run: bool = False,
    panel_config: dict | None = None,
    **common,
) -> list:
    """Panels describing one multi-run jaxrens NS output directory.

    ``out_dir`` holds ``<prefix>.runNN.energies`` (+ optional ``.re_stats.h5``,
    ``.adaptation.h5``, ``.runNN.traj.extxyz``). Each panel is wired only when
    its file(s) are present, so a partial run yields a smaller section.

    A Steinhardt ``<q_l>(T)`` heatmap is added via ``panel_config`` (an ``add``
    entry), not a dedicated argument; the panel only *declares* a processing
    need -- the caller runs it via ``resolve_processing`` before rendering.

    ``panel_config`` (``{"disable", "add", "panels"}`` of
    :class:`~alviz.PanelSpec`) layers a config over the curated NS set -- e.g.
    an ``add`` of several ``SteinhardtSpec`` at different ``l``/``r_cut`` (each
    gets a distinct, param-keyed name so their caches/slots never collide).
    Built through the same registry that backs the AL page, so an NS panel has
    one definition regardless of which dashboard renders it.
    """
    out_dir = Path(out_dir)
    energies = sorted(out_dir.glob(f"{prefix}.run*.energies"))
    if not energies:
        return []
    trajs = sorted(out_dir.glob(f"{prefix}.run*.traj.extxyz"))
    re_stats = out_dir / f"{prefix}.re_stats.h5"
    adaptation = out_dir / f"{prefix}.adaptation.h5"

    # A raw NS output dir is just a single-iteration RunContext
    # (ns_dirs=[out_dir]) whose "sampled" configs are the per-replica
    # trajectories -- so the shared builders wire the section exactly as they
    # wire the AL page.
    from alviz import build_entries, resolve_panel_list

    from jaxrens_al.dashboard import RunContext
    from jaxrens_al.panel_specs import apply_ctx_globals, ns_default_specs

    ctx = RunContext(
        calchome=out_dir,
        datapath=Path(common["datapath"]),
        iteration=0,  # single "iteration" -> is_summary is False
        ns_dirs=[out_dir],
        ns_prefix=prefix,
        ns_k_B=k_B,
        ns_temperatures=temperatures,
        ns_swap_axes=swap_axes,
        sampled_files=trajs,
        adaptation=adaptation if adaptation.exists() else None,
        re_stats=re_stats if re_stats.exists() else None,
        force_recompute=common.get("force_recompute", False),
    )
    cfg = panel_config or {}
    specs = resolve_panel_list(
        ns_default_specs(per_run=per_run),
        panels=cfg.get("panels"),
        add=cfg.get("add"),
        disable=cfg.get("disable"),
    )
    specs = apply_ctx_globals(specs, ctx)
    return build_entries(
        specs,
        ctx,
        datapath=ctx.datapath,
        force_recompute=ctx.force_recompute,
    )


# ---------------------------------------------------------------------------
# DFT section (from scripts/dft_dashboard.py).
# ---------------------------------------------------------------------------
def dft_section(
    files,
    *,
    energy_key: str = REF_ENERGY_KEY,
    forces_key: str = REF_FORCES_KEY,
    pred_energy_key: str | None = SURROGATE_ENERGY_KEY,
    panel_config: dict | None = None,
    **common,
) -> list:
    """Panels describing one set of DFT-labelled configs.

    ``files`` carry ``energy_key`` / ``forces_key`` (the keys remote_handler's
    evaluators write). When ``pred_energy_key`` is present in the configs (the
    sampler's surrogate energy, preserved through the evaluator), the
    surrogate-vs-reference parity/error panels are added too -- "how far off
    was the sampler's model". Pass ``pred_energy_key=None`` to skip that probe.

    Built through the shared panel registry (like ``ns_section``): the labelled
    configs are a single-iteration ``RunContext`` and ``panel_config``
    (``disable``/``add``/``panels``) layers on top.
    """
    files = [Path(f) for f in files]
    if not files:
        return []

    from alviz import build_entries, resolve_panel_list

    from jaxrens_al.dashboard import RunContext
    from jaxrens_al.panel_specs import apply_ctx_globals, dft_default_specs

    ctx = RunContext(
        calchome=files[0].parent,
        datapath=Path(common["datapath"]),
        iteration=0,  # single "iteration" -> is_summary is False
        eval_files=files,  # sampled_energy_volume / parity / energy_error
        sampled_files=files,  # atomic_envs reads sampled_files
        energy_key=energy_key,
        forces_key=forces_key,
        # None -> the parity/error builders self-gate off (no surrogate probe).
        surrogate_energy_key=pred_energy_key,
        force_recompute=common.get("force_recompute", False),
    )
    cfg = panel_config or {}
    specs = resolve_panel_list(
        dft_default_specs(),
        panels=cfg.get("panels"),
        add=cfg.get("add"),
        disable=cfg.get("disable"),
    )
    specs = apply_ctx_globals(specs, ctx)
    return build_entries(
        specs,
        ctx,
        datapath=ctx.datapath,
        force_recompute=ctx.force_recompute,
    )


# ---------------------------------------------------------------------------
# Training section (from scripts/training_dashboard.py).
# ---------------------------------------------------------------------------
def find_model(results, override=None):
    """Locate the trained-model bundle to re-score (or None).

    An explicit ``override`` wins (a ``.bundle`` is a directory, so any
    existing path is accepted). Otherwise search the results tree, then a
    sibling ``trainer_output/`` -- the staged handler writes the model there,
    not under ``<i>/training/results`` -- found by walking up the parents.
    """
    if override:
        p = Path(override)
        return p if p.exists() else None

    results = Path(results)

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


def training_section(
    results_dir,
    *,
    energy_key: str = REF_ENERGY_KEY,
    forces_key: str = REF_FORCES_KEY,
    model=None,
    valid_set=None,
    backend: str | None = None,
    show_forces: bool = True,
    max_neighbors: int = 50,
    include_parity: bool = True,
    **common,
) -> list:
    """Panels describing one training ``results`` directory.

    Auto-detects the trainer by what is on disk (searched recursively):

    * **MACE** (``*_train.txt`` JSON-lines log) -> :class:`MaceTrainingCurve`,
      plus a re-scored :class:`ModelParityPanel` (MACE writes no such bundle).
    * **NeuralIL** (``error_*.txt`` logs / ``.npz`` bundles) ->
      :class:`NeuralILTrainingCurve` + :class:`NeuralILParityPanel`.

    If a ``db_train.extxyz`` is present, its energy/volume spread and
    descriptor diversity are added. Returns ``[]`` when no logs are found (the
    caller decides whether that is an error).
    """
    results = Path(results_dir)
    mace_logs = sorted(results.rglob("*_train.txt"))
    error_logs = sorted(results.rglob("error_*.txt"))
    npz_files = sorted(results.rglob("*.npz"))
    train_sets = sorted(results.rglob("db_train.extxyz"))
    # Validation split to re-score for the model parity: prefer db_val*, then
    # the training set. MACE's split_db writes db_val.extxyz.
    valid_sets = (
        [Path(valid_set)]
        if valid_set
        else (
            sorted(results.rglob("db_val.extxyz"))
            or sorted(results.rglob("db_valid.extxyz"))
            or train_sets
        )
    )

    entries: list = []
    # ``has_parity`` tracks whether a trained-model parity panel is already
    # covered (NeuralIL reads it from .npz); MACE has none, so we re-score.
    has_parity = False
    default_backend = "neuralil"
    if mace_logs:
        default_backend = "mace"
        entries.append(MaceTrainingCurve(train_logs=mace_logs, **common))
    elif error_logs or npz_files:
        if error_logs:
            entries.append(
                NeuralILTrainingCurve(error_files=error_logs, **common)
            )
        if npz_files:
            entries.append(NeuralILParityPanel(npz_files=npz_files, **common))
            has_parity = True
    else:
        return []

    # Trained-model parity by re-scoring a labelled set (mainly MACE, which
    # writes no .npz bundle). Auto-detect the model bundle and backend.
    if not has_parity and include_parity and valid_sets:
        model_file = find_model(results, model)
        if model_file is not None:
            entries.append(
                ModelParityPanel(
                    config_files=valid_sets,
                    model_file=model_file,
                    backend_type=backend or default_backend,
                    energy_ref_key=energy_key,
                    forces_ref_key=forces_key,
                    show_forces=show_forces,
                    max_neighbors=max_neighbors,
                    **common,
                )
            )

    if train_sets:
        # Energy/volume spread + descriptor diversity of what it trained on.
        entries += [
            ExtxyzSampledEnergyVolume(
                config_files=train_sets,
                energy_key=energy_key,
                forces_key=forces_key,
                **common,
            ),
            AtomicEnvsPanel(config_files=train_sets, **common),
        ]
    return entries
