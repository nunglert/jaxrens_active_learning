"""Run-side resolver: annotate a dashboard's trajectories in one job.

Producer-backed panels declare *needs* (plain
:class:`~jaxrens_al.dashboard.panels.processing.ProcessingNeed` data -- what annotation
their input trajectory requires). :func:`resolve_processing` collects those
across a whole dashboard, deduplicates them into a minimal set of processor
steps, runs a **single** ``remote_handler`` trajectory-processing job (all
steps over all distinct trajectories -- the handler shards them; no manual
concatenation), and hands each panel back its annotated path via
``receive_annotations``.

This keeps the "annotate the trajectory" orchestration in one place instead of
wiring a producer into every panel, and keeps the panels themselves pure
dashboard-side readers. It is invoked explicitly by a run-side dashboard
script -- a plain (already-annotated) render never calls it.

New processor kinds slot in by registering a builder with
:func:`register_processor_kind` -- ``steinhardt`` is registered here as the
first consumer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jaxrens_al.run.resolver_base import (
    KindRegistry,
    collect_needs,
    deliver,
    stable_tag,
)

logger = logging.getLogger(__name__)

# kind (registry name) -> callable(params: dict) -> remote_handler Processor.
_REGISTRY = KindRegistry("processor")


def local_execution_context(workdir) -> dict:
    """Cluster context for running the processing pipeline **locally**.

    Returns the ``cluster_config``/``remote_fs``/``remote_savepath``/
    ``local_savepath`` kwargs for :func:`resolve_processing`, wired to
    ``remote_handler``'s scheduler-free ``direct`` executor +
    ``LocalFileSystem`` -- so the real ``driver.py`` + processor steps run on
    this machine (no SLURM, no SSH). The conda env for the driver is from the
    interpreter (``sys.prefix``). Use it in a run-side dashboard script; on a
    real cluster pass a proper ``ClusterConfig``/``remote_fs`` instead.
    """
    import sys

    from remote_handler.core import file_handler
    from remote_handler.core.cluster_handler import ClusterConfig

    prefix = Path(sys.prefix)
    conda_env = prefix.name
    conda_source = (
        str(prefix.parents[1]) if prefix.parent.name == "envs" else ""
    )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cluster = ClusterConfig(
        remote_machine="local",
        partition="",
        conda_env=conda_env,
        conda_source=conda_source,
        savepath=str(workdir),
        template="local.sh.j2",
        executor="direct",
    )
    return {
        "cluster_config": cluster,
        "remote_fs": file_handler.LocalFileSystem(),
        "remote_savepath": str(workdir),
        "local_savepath": str(workdir),
    }


def execution_context(cluster, workdir) -> dict:
    """Cluster context for :func:`resolve_processing`, selected by name.

    ``cluster`` is one of: ``"local"`` (or ``None``) for scheduler-free local
    execution (:func:`local_execution_context`); the name/path of a
    ``remote_handler`` ``ClusterConfig`` JSON -- an absolute path, or a name
    resolved under ``REMOTE_HANDLER_CONFIG_DIR`` -- for a configured cluster
    (SLURM, or a remote machine whose config sets ``executor="direct"``); or an
    already-built ``ClusterConfig`` object (e.g. from ``ClusterConfig.from_name``
    in a run script). That is how a caller picks local, a dedicated
    postprocessing cluster, or an existing one, with a single argument. Results
    are mirrored into ``workdir``.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if cluster in (None, "local"):
        return local_execution_context(workdir)

    from remote_handler.core import file_handler
    from remote_handler.core.cluster_handler import ClusterConfig

    cfg = (
        cluster
        if isinstance(cluster, ClusterConfig)
        else ClusterConfig.from_file(cluster)
    )
    return {
        "cluster_config": cfg,
        "remote_fs": file_handler.RemoteFileSystem(cfg.remote_machine),
        "remote_savepath": str(cfg.savepath),
        "local_savepath": str(workdir),
    }


def _up_to_date(input_file, output_file) -> bool:
    """True if ``output_file`` exists and is no older than ``input_file``.

    The mtime guard makes reuse re-run when a trajectory grew (e.g. new NS
    frames) while a stale annotation from an earlier render still sits on disk;
    if the input can't be stat'd, fall back to "reuse if the output exists".
    """
    output_file = Path(output_file)
    if not output_file.exists():
        return False
    try:
        return output_file.stat().st_mtime >= Path(input_file).stat().st_mtime
    except OSError:
        return True


def register_processor_kind(kind: str, builder) -> None:
    """Register how ``ProcessingNeed``\\ s of ``kind`` become Processor step(s).

    ``builder(param_dicts: list[dict]) -> Processor | list[Processor]`` --
    receives *all* the distinct ``params_dict``\\ s for this kind across the
    dashboard, so it can **merge** them into as few steps as possible. That is
    required, not just an optimization: ``remote_handler`` resolves a processor
    by its (fixed) ``name`` on the cluster and forbids two steps of the same
    name in one pipeline, so a kind that would otherwise emit N same-named steps
    must fold them into one (e.g. Steinhardt unions its ``l_values``). Called
    run-side (may import ``remote_handler`` / the upstream).
    """
    _REGISTRY.register(kind, builder)


def _steps_from_needs(needs):
    """Distinct needs -> Processor steps, each kind's needs built together."""
    by_kind, seen = {}, set()
    for need in needs:
        key = (need.kind, need.params)
        if key in seen:
            continue
        seen.add(key)
        if need.kind not in _REGISTRY:
            _REGISTRY.builder(
                need.kind
            )  # raise the uniform unknown-kind error
        by_kind.setdefault(need.kind, []).append(need.params_dict)

    steps = []
    for kind, param_dicts in by_kind.items():
        built = _REGISTRY.builder(kind)(param_dicts)
        steps.extend(built if isinstance(built, list) else [built])
    return steps


def resolve_processing(
    panels,
    *,
    cluster_config,
    remote_fs,
    remote_savepath,
    shard_size: int = 512,
    n_processors: int = 1,
    num_parallel: int = 8,
    loop_time: float = 30.0,
    local_savepath=None,
    reuse: bool = True,
):
    """Annotate the trajectories a dashboard's panels need, in one job.

    Collects every panel's :meth:`processing_needs`, deduplicates them into a
    minimal pipeline, runs one ``RemoteTrajectoryProcessor`` over all distinct
    input trajectories, and delivers ``{input -> annotated}`` back to each
    panel via ``receive_annotations``.

    Args:
        panels: The dashboard's panels (those without needs are ignored).
        cluster_config, remote_fs, remote_savepath: cluster context.
        shard_size, n_processors, num_parallel, loop_time: array-job knobs.
        local_savepath: optional local mirror dir for the handler.
        reuse: Annotate only trajectories whose output is missing or stale
            (input newer than a prior annotation); skip the ones already done.

    Returns:
        The ``{input_file -> annotated_file}`` mapping (empty if no needs).
    """
    consumers, needs = collect_needs(panels, "processing_needs")
    if not needs:
        return {}

    steps = _steps_from_needs(needs)
    inputs = sorted({i for n in needs for i in n.inputs})

    # A tag keyed by the step-set, so a different pipeline writes different
    # output files (no stale reuse) and an identical one reuses.
    tag = stable_tag("proc", sorted({(n.kind, n.params) for n in needs}))

    from remote_handler.core.atoms_shard import front_tag

    mapping = {f: Path(front_tag(f, tag)) for f in inputs}
    # Annotate only the trajectories whose output is missing or stale -- adding
    # one iteration re-annotates just that trajectory, not the whole set (the
    # handler shards whatever it is given; already-done inputs are left alone).
    todo = [f for f in inputs if not (reuse and _up_to_date(f, mapping[f]))]
    if todo:
        from remote_handler.processing import RemoteTrajectoryProcessor

        logger.info(
            "resolve_processing: %d step(s) over %d of %d trajectory file(s) "
            "(%d reused)",
            len(steps),
            len(todo),
            len(inputs),
            len(inputs) - len(todo),
        )
        RemoteTrajectoryProcessor(
            traj_files=list(todo),  # handler shards them -- no manual concat
            cluster_config=cluster_config,
            remote_fs=remote_fs,
            n_processors=n_processors,
            shard_size=shard_size,
            savepath=str(remote_savepath),
            num_parallel=num_parallel,
            processors=steps,
            tag=tag,
            local_savepath=local_savepath,
            jobname="AL_PROCESSING",
            loop_time=loop_time,
            # every=1 (take every frame) -- also dodges a combine_trajs crash
            # when the slice step is None.
            every=1,
        ).run()
    else:
        logger.info(
            "resolve_processing: all %d trajectory file(s) already annotated; "
            "skipping job",
            len(inputs),
        )

    delivered = {f: mapping[f] for f in inputs}
    deliver(consumers, "receive_annotations", delivered)
    return delivered


def _submit_pending(pending) -> None:
    """Submit every ``(key, RemoteTrajectoryProcessor)`` in ``pending``.

    ``RemoteTrajectoryProcessor.run()`` bundles allocate+submit+**wait**
    +collect into one blocking call, so a caller looping ``.run()`` over
    several independent jobs (e.g. one per distinct model) only puts job
    *N+1* on the SLURM queue once job *N* has entirely finished. Splitting
    off ``allocate_data()`` + ``submit(only_start=True)`` -- the same split
    ``ArrayScriptTask.detach()`` uses for crash-resume -- submits every job
    up front instead, so independent jobs queue and run concurrently. Pair
    with :func:`_collect_pending`.
    """
    for _, proc in pending:
        proc.allocate_data()
        proc.submit(only_start=True)


def _collect_pending(pending) -> None:
    """Wait for + harvest every job started by :func:`_submit_pending`."""
    for _, proc in pending:
        proc.executor.wait_until_finished()
        proc.collect_and_harvest()


def resolve_model_evaluations(
    panels,
    *,
    cluster_config,
    remote_fs,
    remote_savepath,
    shard_size: int = 512,
    n_processors: int = 1,
    num_parallel: int = 8,
    loop_time: float = 30.0,
    local_savepath=None,
    reuse: bool = True,
):
    """Score a dashboard's configs with their trained models, on the cluster.

    Collects every panel's :meth:`model_eval_needs`
    (:class:`~jaxrens_al.dashboard.panels.model_eval.ModelEvalNeed`) and runs **one job per
    distinct model** (a ``model_eval`` processor over that model's config files),
    so a GPU-only MACE-jax bundle is evaluated on a GPU node rather than in the
    dashboard process. Predictions land under per-model keys, delivered back via
    ``receive_model_eval``.

    One job per model (not one merged job) both dodges the pipeline's
    same-processor-name rule and avoids scoring every model on every other
    model's configs. Every model's job is submitted (:func:`_submit_pending`)
    before any of them is awaited (:func:`_collect_pending`), so independent
    models' jobs queue on the cluster concurrently instead of one only
    reaching the queue once the previous model's job has fully finished. Args
    mirror :func:`resolve_processing`.

    Returns:
        A ``{model -> {input_file -> annotated_file}}`` mapping (empty if no
        needs) -- keyed by model so panels sharing an input file (the EOS
        reference scans) each read their own model's annotation.
    """
    from jaxrens_al.dashboard.panels.model_eval import model_eval_keys

    consumers, needs = collect_needs(panels, "model_eval_needs")
    if not needs:
        return {}

    from remote_handler.core.atoms_shard import front_tag

    # Group by model identity (+ the eval knobs that change the annotation), so
    # each distinct trained model is loaded and run once over its config union.
    groups: dict = {}
    for n in needs:
        key = (
            n.model,
            n.backend,
            n.want_forces,
            n.max_neighbors,
            n.min_image_cutoff,
            n.batch_size,
            n.max_neighbors_list,
            n.max_neighbors_offset,
        )
        groups.setdefault(key, set()).update(n.inputs)

    group_files: dict = {}  # key -> (mapping, inputs)
    pending = []  # (key, RemoteTrajectoryProcessor) awaiting collection
    for key, input_set in groups.items():
        (
            model,
            backend,
            want_forces,
            max_nb,
            cutoff,
            batch_size,
            ladder,
            offset,
        ) = key
        inputs = sorted(input_set)
        energy_key, forces_key = model_eval_keys(model)
        # The bucketing knobs join the tag only when set, so a static-buffer
        # need keeps its pre-existing tag (and on-disk annotations are reused).
        tag_key = (model, backend, want_forces, max_nb, cutoff)
        if batch_size is not None or ladder:
            tag_key += (batch_size, ladder, offset)
        tag = stable_tag("meval", tag_key)
        mapping = {f: Path(front_tag(f, tag)) for f in inputs}
        group_files[key] = (mapping, inputs)
        todo = [
            f for f in inputs if not (reuse and _up_to_date(f, mapping[f]))
        ]
        if todo:
            from remote_handler.processing import RemoteTrajectoryProcessor

            from jaxrens_al.run.processors.model_eval import ModelEvalProcessor

            logger.info(
                "resolve_model_evaluations: model %s over %d of %d file(s) "
                "(%d reused)",
                Path(model).name,
                len(todo),
                len(inputs),
                len(inputs) - len(todo),
            )
            proc = RemoteTrajectoryProcessor(
                traj_files=list(todo),
                cluster_config=cluster_config,
                remote_fs=remote_fs,
                n_processors=n_processors,
                shard_size=shard_size,
                savepath=str(remote_savepath),
                num_parallel=num_parallel,
                processors=[
                    ModelEvalProcessor(
                        model,
                        backend=backend,
                        energy_key=energy_key,
                        forces_key=forces_key,
                        want_forces=want_forces,
                        max_neighbors=max_nb,
                        min_image_cutoff=cutoff,
                        batch_size=batch_size,
                        max_neighbors_list=list(ladder),
                        max_neighbors_offset=offset,
                    )
                ],
                tag=tag,
                local_savepath=local_savepath,
                jobname="AL_MODEL_EVAL",
                loop_time=loop_time,
                every=1,
            )
            pending.append((key, proc))

    _submit_pending(pending)
    _collect_pending(pending)

    # Keyed by model, not input: distinct iterations' models annotate the
    # *same* (shared) reference files into differently-tagged outputs, so a
    # flat {input -> annotated} map would let the last model overwrite the
    # rest. ``{model -> {input -> annotated}}`` keeps each panel's own.
    delivered: dict = {}
    for (model, *_rest), (mapping, inputs) in group_files.items():
        delivered.setdefault(model, {}).update({f: mapping[f] for f in inputs})

    deliver(consumers, "receive_model_eval", delivered)
    return delivered


def resolve_phase_labeling(
    panels,
    *,
    cluster_config,
    remote_fs,
    remote_savepath,
    shard_size: int = 512,
    n_processors: int = 1,
    num_parallel: int = 8,
    loop_time: float = 30.0,
    batch_size: int = 32,
    local_savepath=None,
    reuse: bool = True,
):
    """Relax + space-group-label a dashboard's NS pools, on the cluster.

    Collects every panel's :meth:`phase_needs`
    (:class:`~jaxrens_al.dashboard.panels.phase.PhaseNeed`) and runs **one job per
    distinct model** -- an ordered ``[relax, spacegroup]`` pipeline over that
    model's trajectories. Relaxation *mutates* structures, so this is its own
    job, isolated from the (annotation-only) Steinhardt / model-eval passes.
    Every distinct model's job is submitted (:func:`_submit_pending`) before
    any of them is awaited (:func:`_collect_pending`), so independent models'
    jobs queue on the cluster concurrently instead of serially. Delivers
    ``{model -> {input -> labelled_file}}`` back via ``receive_phase_labels``.
    Args mirror :func:`resolve_processing`.
    """
    consumers, needs = collect_needs(panels, "phase_needs")
    if not needs:
        return {}

    from remote_handler.core.atoms_shard import front_tag

    groups: dict = {}
    for n in needs:
        key = (
            n.model,
            n.backend,
            n.method,
            n.fmax,
            n.steps,
            n.symprec,
            n.max_neighbors,
        )
        groups.setdefault(key, set()).update(n.inputs)

    group_files: dict = {}  # key -> (mapping, inputs)
    pending = []  # (key, RemoteTrajectoryProcessor) awaiting collection
    for key, input_set in groups.items():
        model, backend, method, fmax, steps, symprec, max_nb = key
        inputs = sorted(input_set)
        tag = stable_tag("phase", key)
        mapping = {f: Path(front_tag(f, tag)) for f in inputs}
        group_files[key] = (mapping, inputs)
        todo = [
            f for f in inputs if not (reuse and _up_to_date(f, mapping[f]))
        ]
        if todo:
            from remote_handler.processing import RemoteTrajectoryProcessor

            from jaxrens_al.run.processors.relax import RelaxProcessor
            from jaxrens_al.run.processors.spacegroup import (
                SpaceGroupProcessor,
            )

            logger.info(
                "resolve_phase_labeling: model %s over %d of %d file(s) "
                "(%d reused)",
                Path(model).name,
                len(todo),
                len(inputs),
                len(inputs) - len(todo),
            )
            proc = RemoteTrajectoryProcessor(
                traj_files=list(todo),
                cluster_config=cluster_config,
                remote_fs=remote_fs,
                n_processors=n_processors,
                shard_size=shard_size,
                savepath=str(remote_savepath),
                num_parallel=num_parallel,
                processors=[
                    RelaxProcessor(
                        model,
                        backend=backend,
                        method=method,
                        fmax=fmax,
                        steps=steps,
                        batch_size=batch_size,
                        max_neighbors=max_nb,
                    ),
                    SpaceGroupProcessor(symprec=symprec),
                ],
                tag=tag,
                local_savepath=local_savepath,
                jobname="AL_PHASE_LABEL",
                loop_time=loop_time,
                every=1,
            )
            pending.append((key, proc))

    _submit_pending(pending)
    _collect_pending(pending)

    delivered: dict = {}
    for (model, *_rest), (mapping, inputs) in group_files.items():
        delivered.setdefault(model, {}).update({f: mapping[f] for f in inputs})

    deliver(consumers, "receive_phase_labels", delivered)
    return delivered


def _build_steinhardt(param_dicts: list):
    """One ``SteinhardtProcessor`` per ``(r_cut, r_cut_min)``, l values unioned.

    Several Steinhardt panels at different ``l`` (same cutoff) share a single
    processor computing all their ``l`` in one pass -- the processor already
    takes an ``l_values`` list, and two same-named steps can't coexist. Distinct
    ``r_cut`` groups need separate processors; that would still collide on the
    ``steinhardt`` name (and the ``q<l>`` output keys carry no cutoff), so it is
    rejected with a clear message rather than the cryptic pipeline error.
    """
    from remote_handler.processing.processors import SteinhardtProcessor

    groups: dict = {}
    for p in param_dicts:
        key = (p["r_cut"], p.get("r_cut_min", 0.0))
        groups.setdefault(key, set()).update(p["l_values"])
    if len(groups) > 1:
        raise ValueError(
            "Steinhardt panels must share one (r_cut, r_cut_min); got "
            f"{sorted(groups)}. The 'steinhardt' processor writes cutoff-free "
            "q<l> keys, so multiple cutoffs can't be annotated in one pass -- "
            "use a single r_cut across the Steinhardt add: entries."
        )
    (r_cut, r_cut_min), ls = next(iter(groups.items()))
    return SteinhardtProcessor(
        r_cut=r_cut, r_cut_min=r_cut_min, l_values=sorted(ls)
    )


register_processor_kind("steinhardt", _build_steinhardt)


def _build_rdf(param_dicts: list):
    """One ``RDFProcessor`` for every RDF panel -- they must agree on the grid.

    The processor writes cutoff-free ``r_rdf`` / ``rdf_total`` keys and two
    steps of the same name can't coexist in one pipeline, so a second
    ``(cutoff, nbins)`` cannot be annotated in the same pass. Panels differing
    only in their *reduction* share one need (the need carries just the grid),
    so this is only hit when two ``add:`` entries really ask for different
    grids.
    """
    from remote_handler.processing.processors import RDFProcessor

    grids = {(p["cutoff"], p["nbins"]) for p in param_dicts}
    if len(grids) > 1:
        raise ValueError(
            f"RDF panels must share one (cutoff, nbins); got {sorted(grids)}. "
            "The 'rdf' processor writes cutoff-free r_rdf/rdf_total keys, so "
            "only one grid can be annotated per pass -- use a single cutoff / "
            "nbins across the rdf_expectation add: entries."
        )
    cutoff, nbins = next(iter(grids))
    return RDFProcessor(cutoff=cutoff, nbins=int(nbins))


register_processor_kind("rdf", _build_rdf)


__all__ = [
    "resolve_processing",
    "resolve_model_evaluations",
    "resolve_phase_labeling",
    "register_processor_kind",
    "execution_context",
    "local_execution_context",
]
