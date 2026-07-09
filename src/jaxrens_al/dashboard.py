"""Assemble an alviz dashboard for a remote_handler active-learning run.

This is the heart of the composition root: it knows *where* remote_handler /
jaxrens put a run's artifacts (via :mod:`jaxrens_al.run_layout`) and feeds
those paths into panels -- generic ``Extxyz*`` readers from ``alviz.panels``
plus this project's own panels (``jaxrens_al.panels``). Neither alviz nor
remote_handler knows about the other; this module is the only bridge.

The same logic serves two views:

* the cross-iteration **summary** (:func:`build_dashboard`) -- one page whose
  panels aggregate over the whole run;
* one **per-iteration** page (:func:`build_iteration_dashboard`) fed only that
  iteration's artifacts -- the CLI loops it into a multipage report.

Both resolve a :class:`RunContext` (the one seam where "where the files live"
meets "what the panels need"), then call the single :func:`assemble` builder.
A panel is included **only when its inputs exist** in that context, so a
fresh/stub run yields a small board and a full run a richer one; the grid
auto-sizes to the panels that have data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from alviz import Dashboard

# Config keys live in one place -- jaxrens_al.keys.
from jaxrens_al.keys import (
    REF_ENERGY_KEY,
    REF_FORCES_KEY,
    SURROGATE_ENERGY_KEY,
)
from jaxrens_al.run_layout import (
    acc_rates_file,
    adaptation_file,
    evaluated_configs,
    evaluated_configs_per_iteration,
    iteration_dir,
    load_progress,
    model_for_iteration,
    ns_result_dir,
    ns_result_dirs,
    quests_pca_json,
    re_stats_file,
    resampled_provenance_json,
    sampled_configs,
    sampled_configs_per_iteration,
    training_error_files,
    training_error_for_iteration,
    training_mace_for_iteration,
    training_mace_logs,
    training_npz_files,
    training_npz_for_iteration,
)


# ---------------------------------------------------------------------------
# Context: the resolved artifacts a board is built from. Filled two ways --
# whole-run (summary) or one iteration -- so a single assembler serves both.
# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    """Artifacts + settings one board is assembled from.

    ``iteration is None`` marks the cross-iteration summary; otherwise it is
    the index of the single AL iteration this page covers. The file lists are
    whatever that view exposes (the whole run, or just this iteration), so the
    same :func:`assemble` reads them uniformly.
    """

    calchome: Path
    datapath: Path
    eval_files: list = field(default_factory=list)
    sampled_files: list = field(default_factory=list)
    npz_files: list = field(default_factory=list)
    error_files: list = field(default_factory=list)
    mace_logs: list = field(default_factory=list)
    # Trainer config sets: the set a model_parity re-scores (valid_sets) and
    # the set whose spread/diversity is shown (train_sets). A panel spec names
    # which of these (or eval_files / sampled_files) it reads via its ``source``
    # field, so one panel type serves several config sets across consumers.
    train_sets: list = field(default_factory=list)
    valid_sets: list = field(default_factory=list)
    ns_dirs: list = field(default_factory=list)
    acc_rates: Optional[Path] = None
    adaptation: Optional[Path] = None
    re_stats: Optional[Path] = None
    quests_pca: Optional[Path] = None
    resampled_provenance: Optional[Path] = None
    eos_reference: Optional[list] = None  # phase reference files
    model_file: Optional[Path] = None
    eos_backend: str = "neuralil"
    # Evaluate trained models (EOS / model-parity) on the cluster instead of
    # in-process -- required for a GPU-only MACE-jax bundle.
    remote_model_eval: bool = False
    energy_key: str = REF_ENERGY_KEY
    forces_key: str = REF_FORCES_KEY
    surrogate_energy_key: str = SURROGATE_ENERGY_KEY
    iteration: Optional[int] = None
    ns_prefix: str = "ns"
    ns_k_B: float = 1.0
    ns_swap_axes: bool = False
    ns_temperatures: Optional[list] = None
    force_recompute: bool = False

    @property
    def is_summary(self) -> bool:
        return self.iteration is None

    @property
    def common(self) -> dict:
        """Kwargs every panel shares (cache dir + recompute flag)."""
        return dict(
            datapath=self.datapath, force_recompute=self.force_recompute
        )


def _coerce_paths(value):
    """A single path or a list of them -> ``list[Path]`` (``None`` stays None).

    Guards the EOS ``eos_reference`` input: a scalar string (a user pointing at
    one file or one directory in the config) must not be iterated character by
    character. Directory *expansion* to per-phase files happens later, in the
    EOS panel, the one place both this global and a per-panel ``references=``
    converge.
    """
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        value = [value]
    return [Path(p) for p in value]


def detect_backend(mace_logs=None, model_file=None) -> str:
    """Infer the jaxrens backend from the trainer's on-disk artifacts.

    MACE writes ``*_train.txt`` logs and a ``.bundle`` / ``-jax.pkl`` model;
    NeuralIL writes ``error_*.txt`` / ``.npz`` and a plain ``.pkl``. Mirrors the
    section path's ``"mace" if mace_logs else "neuralil"``
    (:func:`jaxrens_al.sections.training_section`) and additionally sniffs the
    model artifact, so a MACE bundle is recognised even when its logs weren't
    kept. The AL-page context builders use this so the backend need not be set
    by hand -- an explicit ``eos_backend`` still overrides it.
    """
    if mace_logs:
        return "mace"
    if model_file is not None:
        name = Path(model_file).name
        if name.endswith(".bundle") or name.endswith("-jax.pkl"):
            return "mace"
    return "neuralil"


def summary_context(
    calchome,
    datapath,
    *,
    eos_backend: Optional[str] = None,
    ns_prefix: str = "ns",
    ns_k_B: float = 1.0,
    ns_swap_axes: bool = False,
    force_recompute: bool = False,
) -> RunContext:
    """Whole-run context: every iteration's artifacts, aggregated.

    ``eos_backend`` defaults to auto-detection from the trainer artifacts
    (:func:`detect_backend`); pass a value to force it.
    """
    calchome = Path(calchome)
    progress = load_progress(calchome)
    mace_logs = training_mace_logs(calchome)
    return RunContext(
        calchome=calchome,
        datapath=Path(datapath),
        eval_files=evaluated_configs(calchome, progress),
        sampled_files=sampled_configs(calchome, progress),
        npz_files=training_npz_files(calchome),
        error_files=training_error_files(calchome),
        mace_logs=mace_logs,
        ns_dirs=ns_result_dirs(calchome, prefix=ns_prefix),
        eos_backend=eos_backend or detect_backend(mace_logs),
        iteration=None,
        ns_prefix=ns_prefix,
        ns_k_B=ns_k_B,
        ns_swap_axes=ns_swap_axes,
        force_recompute=force_recompute,
    )


def iteration_context(
    calchome,
    datapath,
    index,
    *,
    eval_files=None,
    sampled_files=None,
    eos_reference=None,
    eos_backend: Optional[str] = None,
    remote_model_eval: bool = False,
    ns_prefix: str = "ns",
    ns_k_B: float = 1.0,
    ns_swap_axes: bool = False,
    ns_temperatures=None,
    force_recompute: bool = False,
) -> RunContext:
    """Single-iteration context (``calchome/<index>``'s artifacts).

    ``eval_files`` / ``sampled_files`` may be passed in (the CLI already has
    them per iteration) to avoid re-reading ``AL_progress.json`` each page.

    ``eos_backend`` defaults to auto-detection from this iteration's trainer
    artifacts (:func:`detect_backend` -- MACE logs / a ``.bundle`` model ->
    ``mace``, else ``neuralil``); pass a value to force it.
    """
    calchome = Path(calchome)
    it_dir = iteration_dir(calchome, index)
    if eval_files is None:
        eval_files = evaluated_configs_per_iteration(calchome)[index]
    if sampled_files is None:
        sampled_files = sampled_configs_per_iteration(calchome)[index]
    ns_dir = ns_result_dir(it_dir, prefix=ns_prefix)
    mace_logs = training_mace_for_iteration(it_dir)
    model_file = model_for_iteration(calchome, index, it_dir=it_dir)
    return RunContext(
        calchome=calchome,
        datapath=Path(datapath),
        eval_files=eval_files,
        sampled_files=sampled_files,
        npz_files=training_npz_for_iteration(it_dir),
        error_files=training_error_for_iteration(it_dir),
        mace_logs=mace_logs,
        ns_dirs=[ns_dir] if ns_dir is not None else [],
        acc_rates=acc_rates_file(it_dir, prefix=ns_prefix),
        adaptation=adaptation_file(it_dir, prefix=ns_prefix),
        re_stats=re_stats_file(it_dir, prefix=ns_prefix),
        quests_pca=quests_pca_json(it_dir),
        resampled_provenance=resampled_provenance_json(it_dir),
        eos_reference=_coerce_paths(eos_reference),
        model_file=model_file,
        eos_backend=eos_backend or detect_backend(mace_logs, model_file),
        remote_model_eval=remote_model_eval,
        iteration=index,
        ns_prefix=ns_prefix,
        ns_k_B=ns_k_B,
        ns_swap_axes=ns_swap_axes,
        ns_temperatures=ns_temperatures,
        force_recompute=force_recompute,
    )


# ---------------------------------------------------------------------------
# Assemble + select: the one builder for both views.
# ---------------------------------------------------------------------------
def assemble(ctx: RunContext, *, panel_config=None) -> list:
    """Panels whose inputs exist in ``ctx`` (summary or single iteration).

    Builds the curated default spec list (:func:`panel_specs.default_specs`,
    in historical render order) and lets a ``panel_config`` layer on top --
    ``{"disable": [...], "add": [PanelSpec...], "panels": [PanelSpec...]}`` (see
    ``alviz.resolve_panel_list``). A Steinhardt heatmap is just an ``add`` entry
    (there is no CLI flag). Each spec's builder self-gates on its inputs, so
    absent-input / wrong-view panels drop out and the grid auto-sizes -- exactly
    as the old hand-wired ``assemble`` did.
    """
    from alviz import build_entries, resolve_panel_list

    from jaxrens_al.panel_specs import apply_ctx_globals, default_specs

    cfg = panel_config or {}
    add = list(cfg.get("add") or [])
    specs = resolve_panel_list(
        default_specs(ctx),
        panels=cfg.get("panels"),
        add=add,
        disable=cfg.get("disable"),
    )
    # Each spec inherits unset global settings (swap_axes/k_B/temperatures)
    # from the context; explicit per-panel values survive.
    specs = apply_ctx_globals(specs, ctx)
    return build_entries(
        specs,
        ctx,
        datapath=ctx.datapath,
        force_recompute=ctx.force_recompute,
    )


def build_dashboard(
    calchome,
    *,
    datapath=None,
    ncols: int = 2,
    width_per_panel: float = 4.0,
    height_per_panel: float = 4.0,
    ns_prefix: str = "ns",
    ns_k_B: float = 1.0,
    ns_swap_axes: bool = False,
    panel_config=None,
    compute: bool = True,
    only=None,
    exclude=None,
    force_recompute: bool = False,
) -> Dashboard:
    """Build the cross-iteration summary board for a run.

    Args:
        calchome: Path to a run directory (holds ``AL_progress.json``).
        datapath: Where panels cache their computed JSON. Defaults to
            ``calchome/dashboard``.
        ncols: Number of columns; rows auto-size to the available panels.
        width_per_panel, height_per_panel: Panel sizing.
        ns_prefix: jaxrens ``out_file_prefix`` of the NS run files.
        ns_k_B: Boltzmann constant for the thermodynamics panel, in the energy
            units of the NS energies (1.0 reduced; 8.617e-5 eV with T in K).
        compute: If True, run the compute/plot lifecycle before returning.
        only / exclude: Restrict the rendered panels by name.
        force_recompute: Recompute panel data even if a JSON cache exists.

    Returns:
        The assembled :class:`alviz.Dashboard`.
    """
    calchome = Path(calchome)
    datapath = Path(datapath) if datapath else calchome / "dashboard"
    datapath.mkdir(parents=True, exist_ok=True)

    ctx = summary_context(
        calchome,
        datapath,
        ns_prefix=ns_prefix,
        ns_k_B=ns_k_B,
        ns_swap_axes=ns_swap_axes,
        force_recompute=force_recompute,
    )
    board = Dashboard.from_entries(
        assemble(ctx, panel_config=panel_config),
        ncols=ncols,
        width_per_panel=width_per_panel,
        height_per_panel=height_per_panel,
        only=only,
        exclude=exclude,
    )
    if compute:
        board.render()
    return board


def build_iteration_dashboard(
    calchome,
    index,
    *,
    datapath,
    eval_files=None,
    sampled_files=None,
    eos_reference=None,
    eos_backend: Optional[str] = None,
    remote_model_eval: bool = False,
    ncols: int = 3,
    width_per_panel: float = 4.0,
    height_per_panel: float = 4.0,
    ns_prefix: str = "ns",
    ns_k_B: float = 1.0,
    ns_swap_axes: bool = False,
    ns_temperatures=None,
    panel_config=None,
    compute: bool = True,
    only=None,
    exclude=None,
    force_recompute: bool = False,
):
    """Build one dashboard page for a single AL iteration.

    Args:
        calchome: Path to the run directory.
        index: AL iteration index (its working dir is ``calchome/<index>``).
        datapath: Where this iteration's panels cache their JSON (must exist;
            use a per-iteration subdir to avoid clobbering other pages).
        eval_files / sampled_files: This iteration's evaluated/sampled config
            files. If None, they are looked up from ``AL_progress.json``.
        only / exclude: Restrict the rendered panels by name.
        force_recompute: Recompute panel data even if a JSON cache exists.

    Returns:
        The assembled :class:`alviz.Dashboard`, or ``None`` if this iteration
        has no panels with data.
    """
    ctx = iteration_context(
        calchome,
        Path(datapath),
        index,
        eval_files=eval_files,
        sampled_files=sampled_files,
        eos_reference=eos_reference,
        eos_backend=eos_backend,
        remote_model_eval=remote_model_eval,
        ns_prefix=ns_prefix,
        ns_k_B=ns_k_B,
        ns_swap_axes=ns_swap_axes,
        ns_temperatures=ns_temperatures,
        force_recompute=force_recompute,
    )
    board = Dashboard.from_entries(
        assemble(ctx, panel_config=panel_config),
        ncols=ncols,
        width_per_panel=width_per_panel,
        height_per_panel=height_per_panel,
        only=only,
        exclude=exclude,
    )
    if not board.entries:
        return None
    if compute:
        board.render(suptitle=f"AL iteration {index}")
    return board
