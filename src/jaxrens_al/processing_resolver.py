"""Run-side resolver: annotate a dashboard's trajectories in one job.

Producer-backed panels declare *needs* (plain
:class:`~jaxrens_al.panels.processing.ProcessingNeed` data -- what annotation
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

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# kind (registry name) -> callable(params: dict) -> remote_handler Processor.
_KIND_BUILDERS: dict = {}


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

    ``cluster`` is either ``"local"`` (or ``None``) for scheduler-free local
    execution (:func:`local_execution_context`), or the name/path of a
    ``remote_handler`` ``ClusterConfig`` JSON -- an absolute path, or a name
    resolved under ``REMOTE_HANDLER_CONFIG_DIR`` -- for a configured cluster
    (SLURM, or a remote machine whose config sets ``executor="direct"``). That
    is how a caller picks local, a dedicated postprocessing cluster, or an
    existing one, with a single flag. Results are mirrored into ``workdir``.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if cluster in (None, "local"):
        return local_execution_context(workdir)

    from remote_handler.core import file_handler
    from remote_handler.core.cluster_handler import ClusterConfig

    cfg = ClusterConfig.from_file(cluster)
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
    _KIND_BUILDERS[kind] = builder


def _steps_from_needs(needs):
    """Distinct needs -> Processor steps, each kind's needs built together."""
    by_kind, seen = {}, set()
    for need in needs:
        key = (need.kind, need.params)
        if key in seen:
            continue
        seen.add(key)
        if need.kind not in _KIND_BUILDERS:
            raise ValueError(
                f"No processor builder registered for kind {need.kind!r}; "
                f"known: {sorted(_KIND_BUILDERS)}."
            )
        by_kind.setdefault(need.kind, []).append(need.params_dict)

    steps = []
    for kind, param_dicts in by_kind.items():
        built = _KIND_BUILDERS[kind](param_dicts)
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
    consumers = [p for p in panels if hasattr(p, "processing_needs")]
    needs = [n for p in consumers for n in p.processing_needs()]
    if not needs:
        return {}

    steps = _steps_from_needs(needs)
    inputs = sorted({i for n in needs for i in n.inputs})

    # A tag keyed by the step-set, so a different pipeline writes different
    # output files (no stale reuse) and an identical one reuses.
    step_key = sorted({(n.kind, n.params) for n in needs})
    tag = "proc" + hashlib.sha1(repr(step_key).encode()).hexdigest()[:10]

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
    for panel in consumers:
        panel.receive_annotations(delivered)
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
    from jaxrens_al.processors.steinhardt import SteinhardtProcessor

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


__all__ = [
    "resolve_processing",
    "register_processor_kind",
    "execution_context",
    "local_execution_context",
]
