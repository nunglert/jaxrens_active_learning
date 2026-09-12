"""The jaxrens nested-sampling Sampler for this project.

In the layered architecture, concrete samplers belong to the *project*, not to
the orchestrator: ``remote_handler`` owns the ``Sampler`` *interface* and the
generic remote-execution machinery (``SingleScriptTask``, ``ClusterConfig``,
the SLURM templating), while the application picks and wires the actual
sampling method.

``JaxrensNSSampler`` is a **native** implementation of that interface for the
``jaxrens`` nested sampler (https://github.com/nunglert/jaxrens). It does *not*
delegate to ``remote_handler``'s own ``NSTrajectorySampler`` -- that class
targets the deprecated ``jaxnest`` package (and references a ``body_ns.sh.j2``
template that no longer ships) and is on its way out. Instead this sampler:

    1. takes a base ``jaxrens`` YAML config -- the user owns the NS settings
       (moves, ``n_live``, termination, ...);
    2. per AL iteration, injects the freshly trained model into
       ``backend.checkpoint_path`` and points ``output`` at the iteration's
       results dir, then ships the config and runs ``jaxrens run`` on the
       cluster via a ``SingleScriptTask`` (body template
       ``body_jaxrens.sh.j2``, shipped by :func:`jaxrens_al.template_dir`);
    3. reads the resulting ``<prefix>.traj.extxyz`` trajectories back as
       ``ase.Atoms`` and emits **all** dead points, tagged with their replica
       ``run_id``.

This sampler is a **pure producer**: it does no candidate selection or
subsampling. Reducing the dead-point pool to the configs worth labelling is a
*selector* concern, modelled as its own stage in
``remote_handler.workflows.StagedActiveLearningHandler`` (a
``SelectorActor`` wrapping a ``ConfigSelector``). Use ``UniformSelector`` for
plain uniform thinning, ``QUESTSSelector`` for descriptor diversity/novelty, or
``UncertaintySelector`` for committee variance -- inserted between this sampler
and the evaluator via ``from_actors(selector=...)``. Keeping selection out of
the sampler is what lets the same NS run feed any selection strategy unchanged.

Requires the ``[run]`` extra (``pip install ".[run]"`` -> remote_handler +
jaxrens). ``jaxrens`` itself only runs on the cluster; the host side needs just
``remote_handler`` + ``ase`` + ``pyyaml``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Union

import ase.io
import numpy as np
import yaml
from ase import Atoms
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import DetachableTaskMixin, SingleScriptTask
from remote_handler.exceptions import FileSystemError, WorkflowError
from remote_handler.workflows.active_learning_base import Sampler

from jaxrens_al.run.templates import template_dir

logger = logging.getLogger(__name__)

# jaxrens backend.type discriminator values (jaxrens cli/schema/backend.py).
_BACKEND_TYPES = ("neuralil", "mace", "nequix", "lj", "jaxmd")


def _set_dotted(config: dict, dotted_key: str, value) -> None:
    """Set ``config["a"]["b"] = value`` from a dotted key ``"a.b"``.

    Mirrors jaxrens' own ``--set a.b=value`` override syntax so a
    ``remote_handler`` ``Schedule`` can ramp nested config fields across
    iterations (e.g. ``{"run.n_live": 400}``). Intermediate dicts are created
    as needed; a non-dict intermediate is an error.
    """
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        nxt = node.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise WorkflowError(
                f"Cannot apply override '{dotted_key}': '{part}' is not a "
                f"mapping in the jaxrens config."
            )
        node = nxt
    node[parts[-1]] = value


class JaxrensNSSampler(DetachableTaskMixin, Sampler):
    """Native jaxrens nested-sampling Sampler (``remote_handler.Sampler``).

    A pure producer: runs NS and returns *all* dead points. Candidate
    selection is a downstream *selector* stage (see the module docstring).

    Args:
        remote_fs: FileSystemHandlerBase for the machine where jaxrens runs.
        cluster_config: ClusterConfig (partition, conda env, header template).
        config_file: Base jaxrens YAML config. Owns the NS settings; this
            sampler only overwrites ``backend.checkpoint_path`` and the
            ``output`` block per iteration.
        n_gpu_parallel: GPUs requested per job (``--n-gpus`` and the header's
            ``--gres=gpu:`` count). Multi-replica runs (one trajectory per
            replica) are driven by the jaxrens config itself, not by this flag.
        backend_type: If given, force ``backend.type`` in the config (one of
            ``neuralil``/``mace``/``nequix``/``lj``/``jaxmd``). If None, the
            base config's ``backend.type`` is kept.
        start_config_file: Optional starting structure. Shipped to the cluster
            and set as ``init.start_config_file`` (replacing any
            ``start_species``/``start_walker_set``).
        out_file_prefix: jaxrens ``output.out_file_prefix`` and the trajectory
            file stem this sampler reads back.
        additional_files: Extra files to ship alongside the job.
        remote_savepath: Remote computation dir; falls back to
            ``cluster_config.savepath``.
        loop_time: Poll interval (seconds) for job completion.
        job_type_template: SLURM body template (default ``body_jaxrens.sh.j2``,
            shipped by jaxrens_al -- must be on ``remote_handler``'s template
            search path).
        stub: Dry-run without a cluster; emit mock NS dead points for testing.
        stub_n_dead: Number of mock dead points to emit in ``stub`` mode.
        collect_only: If True, ``update`` reconstructs the job submitted by an
            earlier, ``collect_only=False`` call (from the manifest written
            beside ``<calcdir>/ns_sampler/results``, via
            ``SingleScriptTask.from_manifest``) instead of shipping the model,
            building a fresh config, and submitting a new job; and
            ``get_sampled_configs`` skips the (already-done) submission.
            ``sampler_input`` is not read in this mode. This is what lets
            ``StagedActiveLearningHandler`` reattach to (rather than
            duplicate) an in-flight NS job after the *driving process* --
            not necessarily the job itself -- dies (a dropped SSH session, a
            killed local process). See ``DetachableTaskMixin``.
        reattach_max_wait: Forwarded to ``DetachableTaskMixin.wait_if_needed``
            as ``max_wait``: when ``collect_only`` and the reattached job
            has not finished yet, how long (seconds) to keep polling for its
            completion marker (every ``loop_time``) before giving up and
            raising, instead of failing on the very first check.
        max_resubmits: Auto-resume across the cluster's walltime limit. When
            > 1 (and not ``collect_only``), a fresh submit fires the first
            job with ``only_start=True`` and additionally pre-queues up to
            ``max_resubmits - 1`` standby continuations in the *same* remote
            directory, each ``sbatch --dependency=afternotok:<previous>`` on
            the identical rendered script -- SLURM only starts a standby
            link if the previous one ends in anything but ``COMPLETED``
            (covers a walltime ``TIMEOUT`` as well as a genuine failure,
            uniformly, with no need to classify which); if a link actually
            succeeds, later ones just sit ``PENDING`` until cancelled. The
            body template (``body_jaxrens.sh.j2``) self-detects ``--resume``
            vs. ``--force`` from whether a checkpoint already exists, so
            every link runs the same script. Waiting for completion then
            reuses ``wait_if_needed``'s file-based polling (job identity
            doesn't matter -- only ``is_finished.info`` in the shared
            directory), and any still-``PENDING`` standby links are
            cancelled once it appears. ``1`` (default) is today's
            behaviour: one job, no chain. Size ``reattach_max_wait`` to
            cover the whole chain (e.g. ``max_resubmits`` times the
            cluster's walltime) if you want a bound at all.
    """

    # calcdir/ns_sampler -- see DetachableTaskMixin.
    _iteration_subdir = "ns_sampler"

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        config_file: os.PathLike,
        n_gpu_parallel: int = 1,
        backend_type: Optional[str] = None,
        start_config_file: Optional[os.PathLike] = None,
        out_file_prefix: str = "ns",
        additional_files: Optional[list[os.PathLike]] = None,
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30.0,
        job_type_template: str = "body_jaxrens.sh.j2",
        stub: bool = False,
        stub_n_dead: int = 64,
        collect_only: bool = False,
        reattach_max_wait: Optional[float] = None,
        max_resubmits: int = 1,
    ):
        super().__init__()
        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.config_file = Path(config_file)
        self.n_gpu_parallel = n_gpu_parallel
        self.backend_type = backend_type
        self.start_config_file = (
            Path(start_config_file) if start_config_file else None
        )
        self.out_file_prefix = out_file_prefix
        self.additional_files = list(additional_files or [])
        self.remote_savepath = (
            Path(remote_savepath) if remote_savepath else None
        )
        self.loop_time = loop_time
        self.job_type_template = job_type_template
        self.stub = stub
        self.stub_n_dead = stub_n_dead
        self.collect_only = collect_only
        self.reattach_max_wait = reattach_max_wait
        if max_resubmits < 1:
            raise WorkflowError("max_resubmits must be >= 1.")
        self.max_resubmits = max_resubmits

        if backend_type is not None and backend_type not in _BACKEND_TYPES:
            raise WorkflowError(
                f"Unknown jaxrens backend_type '{backend_type}'. "
                f"Expected one of {_BACKEND_TYPES}."
            )
        if not self.stub and not self.config_file.exists():
            raise FileSystemError(
                f"Base jaxrens config not found: {self.config_file}"
            )

    # -- config assembly --------------------------------------------------

    def _load_base_config(self) -> dict:
        with open(self.config_file, "r") as f:
            return yaml.safe_load(f) or {}

    def _build_config(
        self,
        model_remote: str,
        start_remote: Optional[str],
        overrides: dict,
    ) -> dict:
        """Return the per-iteration jaxrens config (pure; easy to unit-test).

        Overwrites the model checkpoint and the output block, optionally the
        backend type and start structure, then applies dotted-key overrides
        from the iteration's schedule kwargs.
        """
        config = self._load_base_config()

        backend = config.setdefault("backend", {})
        if self.backend_type is not None:
            backend["type"] = self.backend_type
        backend["checkpoint_path"] = model_remote

        if start_remote is not None:
            init = config.setdefault("init", {})
            init["start_config_file"] = start_remote
            # init demands exactly one start source -- drop the others.
            init.pop("start_species", None)
            init.pop("start_walker_set", None)

        output = config.setdefault("output", {})
        output.setdefault("working_dir", ".")
        output["out_file_prefix"] = self.out_file_prefix
        output["format"] = "extxyz"

        for key, value in (overrides or {}).items():
            _set_dotted(config, key, value)

        return config

    # -- Sampler interface ------------------------------------------------

    def update(
        self,
        calcdir: os.PathLike,
        sampler_input: Union[os.PathLike, list],
        **kwargs,
    ) -> None:
        """Prepare (and stage) the jaxrens run for this AL iteration.

        Args:
            calcdir: This iteration's working directory.
            sampler_input: Trained model file (or list -- the latest is used)
                to plug into ``backend.checkpoint_path``. Not read when
                ``collect_only`` (see the class docstring).
            **kwargs: Schedule overrides as dotted jaxrens keys, e.g.
                ``{"run.n_live": 400, "run.n_mcmc_steps": 30}``.
        """
        self.ns_dir = self.iteration_dir(calcdir)
        self.results_dir = self.results_dir_for(calcdir)
        os.makedirs(self.ns_dir, exist_ok=True)
        self.config_path = self.ns_dir / "ns_config.yaml"

        if self.stub:
            return

        if not self.collect_only:
            model_file = sampler_input
            if isinstance(model_file, (list, tuple)):
                model_file = model_file[-1]
            self.model_file = Path(model_file)
        it = Path(calcdir).name

        def _build_task() -> SingleScriptTask:
            task = SingleScriptTask(
                script_file=None,
                input_file=self.config_path,
                local_savepath=self.results_dir,
                cluster_config=self.cluster_config,
                remote_fs=self.remote_fs,
                logfile="ns.log",
                remote_savepath=self.remote_savepath,
                collect_only=False,
                loop_time=self.loop_time,
                jobname=f"AL_NS_{it}",
                n_processors=self.n_gpu_parallel,
                additional_files=self.additional_files,
                job_type_template=self.job_type_template,
            )

            # Ship the model (and start structure) and point the config at
            # the resulting remote paths. ``copy_path`` (not ``copy_file``)
            # so a mace-jax model *bundle directory* ships recursively, not
            # just a single-file checkpoint.
            model_remote = str(task.copy_path(self.model_file))
            start_remote = None
            if self.start_config_file is not None:
                start_remote = str(task.copy_file(self.start_config_file))

            config = self._build_config(
                model_remote, start_remote, dict(kwargs)
            )
            with open(self.config_path, "w") as f:
                yaml.safe_dump(
                    config, f, default_flow_style=False, sort_keys=False
                )
            return task

        # collect_only=True reconstructs via SingleScriptTask.from_manifest
        # (the mixin's default) rather than building a fresh task, which is
        # exactly why _build_task above (never called on that path) is the
        # only place touching sampler_input/model shipping/config writing.
        # build_or_resume_single_task also does the allocate_data() call.
        self.ns_task = self.build_or_resume_single_task(
            self.results_dir, _build_task
        )

    def get_sampled_configs(self) -> Path:
        """Run jaxrens, collect every dead point, and write them to extxyz.

        Returns:
            Path to ``sampled_configs.extxyz`` (the full dead-point pool) for
            the next stage -- a selector, or the evaluator directly.
        """
        if self.stub:
            logger.info("stub mode -- creating mock NS dead points")
            all_ats = self._create_stub_outputs()
        else:
            error_message = (
                "jaxrens NS job did not finish successfully (no "
                f"is_finished.info in {self.results_dir})."
            )
            if self.max_resubmits > 1 and not self.collect_only:
                # Fresh submit with auto-resume chaining: fire job 1
                # (only_start) and pre-queue the standby continuations
                # ourselves -- submit_if_needed's plain blocking submit()
                # only ever tracks one job id, which is wrong once a later
                # chain link may be the one that actually finishes.
                self._submit_chain()
                # Reuse wait_if_needed's file-based polling branch (job
                # identity doesn't matter, only is_finished.info in this
                # directory) by pretending to be a reattach for the
                # duration of the wait -- submission above already did
                # what submit_if_needed would have, so this doesn't
                # double-submit anything.
                prev_collect_only = self.collect_only
                self.collect_only = True
                try:
                    self.wait_if_needed(
                        self.ns_task,
                        max_wait=self.reattach_max_wait,
                        error_message=error_message,
                    )
                finally:
                    self.collect_only = prev_collect_only
            else:
                self.submit_if_needed(self.ns_task)
                self.wait_if_needed(
                    self.ns_task,
                    max_wait=self.reattach_max_wait,
                    error_message=error_message,
                )
            self._cleanup_chain()
            self.ns_task.collect_results()
            all_ats = self._read_trajectories()

        os.makedirs(self.results_dir, exist_ok=True)
        output_file = self.results_dir / "sampled_configs.extxyz"
        ase.io.write(output_file, all_ats)
        logger.info(f"Wrote {len(all_ats)} NS dead points to {output_file}")
        return output_file

    # -- helpers ----------------------------------------------------------

    def _chain_path(self) -> Path:
        """Local sidecar tracking this NS job's auto-resume chain job ids.

        Mirrors ``DetachableTaskMixin.manifest_path``'s
        ``<results_dir>.manifest.json`` naming, so a driver that restarts
        mid-chain finds it the same deterministic way the manifest is found.
        """
        return Path(str(self.results_dir) + ".chain.json")

    def _submit_chain(self) -> None:
        """Fire the NS job plus ``max_resubmits - 1`` standby continuations.

        Each standby link is ``sbatch --dependency=afternotok:<previous>``
        against the *same* rendered ``run_slurm.sh`` in the *same* remote
        directory -- SLURM only starts it once the previous link ends in
        anything but ``COMPLETED`` (a walltime ``TIMEOUT`` included, with no
        need to tell that apart from a genuine failure here). If a link
        actually succeeds, every later one just sits ``PENDING`` until
        :meth:`_cleanup_chain` cancels it. The body template self-detects
        ``--resume`` vs. ``--force`` from an existing checkpoint, so every
        link runs the identical script -- nothing here needs to know which
        attempt it is.
        """
        self.ns_task.submit(only_start=True)
        job_ids = [str(self.ns_task.executor.pid)]
        calcdir = self.ns_task.calcdir
        for i in range(self.max_resubmits - 1):
            link = i + 2  # link 1 is the job just fired above
            result = self.remote_fs.run_command(
                f"cd {calcdir} && sbatch "
                f"-e slurm_chain{link}.err -o slurm_chain{link}.out "
                f"--dependency=afternotok:{job_ids[-1]} run_slurm.sh"
            )
            match = re.search(r"(\d+)", result.stdout)
            if not match:
                logger.warning(
                    "Could not parse a job id from chain link %d's sbatch "
                    "output (stdout=%r stderr=%r); stopping the chain at "
                    "%d link(s) instead of the requested %d.",
                    link,
                    result.stdout,
                    result.stderr,
                    len(job_ids),
                    self.max_resubmits,
                )
                break
            job_ids.append(match.group(1))
        logger.info(
            "Submitted jaxrens NS auto-resume chain in %s: jobs %s",
            calcdir,
            job_ids,
        )
        with open(self._chain_path(), "w") as f:
            json.dump(job_ids, f)

    def _cleanup_chain(self) -> None:
        """Cancel any still-pending standby chain jobs once collected.

        Safe to call unconditionally -- after a fresh chain submit, after
        reattaching to one, or when chaining was never used at all (no
        sidecar file, a no-op). Cancelling an already-finished or vanished
        job id is a harmless no-op for ``scancel``.
        """
        chain_path = self._chain_path()
        if not chain_path.exists():
            return
        with open(chain_path) as f:
            job_ids = json.load(f)
        if job_ids:
            result = self.remote_fs.run_command(f"scancel {' '.join(job_ids)}")
            if result.returncode != 0:
                logger.warning(
                    "scancel for leftover NS chain jobs %s returned %d: %s",
                    job_ids,
                    result.returncode,
                    result.stderr,
                )
        chain_path.unlink()

    def _read_trajectories(self) -> list[Atoms]:
        """Read jaxrens ``.traj.extxyz`` outputs, tagging each with run_id."""
        prefix = self.out_file_prefix
        # Multi-replica runs write one trajectory per replica.
        tfiles = sorted(self.results_dir.rglob(f"{prefix}.run*.traj.extxyz"))
        if not tfiles:
            tfiles = sorted(self.results_dir.rglob(f"{prefix}.traj.extxyz"))
        if not tfiles:
            raise WorkflowError(
                f"No jaxrens trajectories ('{prefix}.traj.extxyz' / "
                f"'{prefix}.run*.traj.extxyz') found under "
                f"{self.results_dir}. Did the NS run finish?"
            )

        all_ats: list[Atoms] = []
        for run_id, tfile in enumerate(tfiles):
            frames = ase.io.read(tfile, index=":")
            for at in frames:
                at.info["run_id"] = run_id
            all_ats.extend(frames)

        if not all_ats:
            raise WorkflowError(
                f"jaxrens trajectories under {self.results_dir} are empty."
            )
        return all_ats

    def _create_stub_outputs(self) -> list[Atoms]:
        """Mock NS dead points for stub-mode testing (jaxrens info keys)."""
        logger.info(f"Creating {self.stub_n_dead} stub NS dead points")
        n_runs = max(1, min(self.n_gpu_parallel, 4))
        all_ats: list[Atoms] = []
        for run_id in range(n_runs):
            per_run = self.stub_n_dead // n_runs
            if run_id == n_runs - 1:  # last run mops up the remainder
                per_run = self.stub_n_dead - per_run * (n_runs - 1)
            for i in range(per_run):
                symbols = ["Cu"] * 16 + ["O"] * 4
                positions = np.random.uniform(0, 8, (len(symbols), 3))
                atoms = Atoms(
                    symbols=symbols,
                    positions=positions,
                    cell=[8, 8, 8],
                    pbc=True,
                )
                atoms.info["run_id"] = run_id
                atoms.info["iter"] = i
                atoms.info["ns_energy"] = -100.0 + np.random.uniform(-50, 0)
                all_ats.append(atoms)
        return all_ats
