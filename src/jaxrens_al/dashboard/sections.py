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
* the global ``jaxrens-al dashboard`` CLI's ``--sections`` mode, which points
  them at each iteration's raw working dir (``calchome/<i>/...``) so the
  combined report includes the same per-artifact detail -- including iterations
  ``AL_progress.json`` skipped.

These are **dashboard-side**: ase + alviz only, no ``remote_handler`` import.
The Steinhardt heatmap a section may emit is a pure reader that merely
*declares* a processing need; running that need is the caller's job (via
:func:`jaxrens_al.run.processing_resolver.resolve_processing`), off this import
path.
"""

from __future__ import annotations

from pathlib import Path

# Raw-dir model discovery lives in run_layout (the filesystem-reading layer);
# re-exported here for the training section + backwards compatibility.
from jaxrens_al.dashboard.run_layout import find_model  # noqa: F401
from jaxrens_al.keys import (
    REF_ENERGY_KEY,
    REF_FORCES_KEY,
    SURROGATE_ENERGY_KEY,
)


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
    from jaxrens_al.dashboard import RunContext, assemble
    from jaxrens_al.dashboard.panel_specs import ns_default_specs

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
    return assemble(
        ctx, specs=ns_default_specs(per_run=per_run), panel_config=panel_config
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

    from jaxrens_al.dashboard import RunContext, assemble
    from jaxrens_al.dashboard.panel_specs import dft_default_specs

    ctx = RunContext(
        calchome=files[0].parent,
        datapath=Path(common["datapath"]),
        iteration=0,  # single "iteration" -> is_summary is False
        eval_files=files,  # every dft panel reads eval_files (via source=)
        energy_key=energy_key,
        forces_key=forces_key,
        # None -> the parity/error builders self-gate off (no surrogate probe).
        surrogate_energy_key=pred_energy_key,
        force_recompute=common.get("force_recompute", False),
    )
    return assemble(ctx, specs=dft_default_specs(), panel_config=panel_config)


# ---------------------------------------------------------------------------
# Training section (from scripts/training_dashboard.py).
# ---------------------------------------------------------------------------
def training_section(
    results_dir,
    *,
    energy_key: str = REF_ENERGY_KEY,
    forces_key: str = REF_FORCES_KEY,
    model=None,
    valid_set=None,
    backend: str | None = None,
    remote_model_eval: bool = False,
    show_forces: bool = True,
    max_neighbors: int = 50,
    include_parity: bool = True,
    panel_config: dict | None = None,
    **common,
) -> list:
    """Panels describing one training ``results`` directory.

    Auto-detects the trainer by what is on disk (searched recursively):

    * **MACE** (``*_train.txt`` JSON-lines log) -> a MACE training curve, plus
      a re-scored model-parity panel (MACE writes no parity bundle).
    * **NeuralIL** (``error_*.txt`` logs / ``.npz`` bundles) -> a NeuralIL
      training curve + parity read from the ``.npz``.

    If a ``db_train.extxyz`` is present, its energy/volume spread and descriptor
    diversity are added. Returns ``[]`` when no logs are found (the caller
    decides whether that is an error).

    Built through the shared panel registry (like ``ns_section`` /
    ``dft_section``): the discovered artifacts become a single-iteration
    ``RunContext`` and ``panel_config`` (``disable``/``add``/``panels``) layers
    on top. The two config-set views bind their input by name -- ``model_parity``
    re-scores the validation split (``valid_sets``), the spread/diversity views
    read ``train_sets`` -- so a curated panel has one definition regardless of
    which config set it reads.
    """
    results = Path(results_dir)
    mace_logs = sorted(results.rglob("*_train.txt"))
    error_logs = sorted(results.rglob("error_*.txt"))
    npz_files = sorted(results.rglob("*.npz"))
    train_sets = sorted(results.rglob("db_train.extxyz"))
    if not (mace_logs or error_logs or npz_files):
        return []
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
    # Model re-score is for the no-.npz case (mainly MACE); NeuralIL's .npz
    # already carries real parity, so model_parity self-gates off then.
    model_file = None if npz_files else find_model(results, model)

    from jaxrens_al.dashboard import RunContext, assemble, detect_backend
    from jaxrens_al.dashboard.panel_specs import training_default_specs

    ctx = RunContext(
        calchome=results,
        datapath=Path(common["datapath"]),
        iteration=0,  # single "iteration" -> is_summary is False
        mace_logs=mace_logs,
        error_files=error_logs,
        npz_files=npz_files,
        train_sets=train_sets,
        valid_sets=valid_sets,
        model_file=model_file,
        eos_backend=backend or detect_backend(mace_logs, model_file),
        remote_model_eval=remote_model_eval,
        energy_key=energy_key,
        forces_key=forces_key,
        force_recompute=common.get("force_recompute", False),
    )
    # ``include_parity=False`` folds into the panel_config as a disable, so the
    # shared assemble handles it like any other config-driven drop.
    cfg = dict(panel_config or {})
    disable = list(cfg.get("disable") or [])
    if not include_parity and "model_parity" not in disable:
        disable.append("model_parity")
    cfg["disable"] = disable
    return assemble(
        ctx,
        specs=training_default_specs(
            show_forces=show_forces, max_neighbors=max_neighbors
        ),
        panel_config=cfg,
    )
