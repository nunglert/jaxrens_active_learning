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


def register_processor_kind(kind: str, builder) -> None:
    """Register how a :class:`ProcessingNeed` of ``kind`` becomes a Processor.

    ``builder(params: dict) -> Processor`` -- ``params`` is the need's
    ``params_dict``. Called run-side (may import ``remote_handler`` / the
    upstream); this is the extension point for future producer-backed panels.
    """
    _KIND_BUILDERS[kind] = builder


def _steps_from_needs(needs):
    """Distinct (kind, params) needs -> deduplicated Processor steps."""
    steps, seen = [], set()
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
        steps.append(_KIND_BUILDERS[need.kind](need.params_dict))
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
        reuse: Skip the job when every expected annotated file already exists.

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
    if not (reuse and all(out.exists() for out in mapping.values())):
        from remote_handler.processing import RemoteTrajectoryProcessor

        logger.info(
            "resolve_processing: %d step(s) over %d trajectory file(s)",
            len(steps),
            len(inputs),
        )
        RemoteTrajectoryProcessor(
            traj_files=list(inputs),  # handler shards them -- no manual concat
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

    delivered = {f: mapping[f] for f in inputs}
    for panel in consumers:
        panel.receive_annotations(delivered)
    return delivered


def _build_steinhardt(params: dict):
    from jaxrens_al.processors.steinhardt import SteinhardtProcessor

    return SteinhardtProcessor(
        r_cut=params["r_cut"],
        l_values=params["l_values"],
        r_cut_min=params.get("r_cut_min", 0.0),
    )


register_processor_kind("steinhardt", _build_steinhardt)


__all__ = [
    "resolve_processing",
    "register_processor_kind",
    "execution_context",
    "local_execution_context",
]
