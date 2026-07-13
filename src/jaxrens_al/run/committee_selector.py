"""Committee-uncertainty candidate selection with its own remote execution.

``remote_handler``'s ``UncertaintySelector`` is a *pure, host-side* strategy: it
ranks a pool by an uncertainty already stored on each frame (e.g. jaxrens'
in-NS ``ns_force_std``). This module covers the other case -- committee
disagreement of a set of *independently trained* MLIPs (MACE models converted to
mace-jax bundles), which must be **evaluated on a GPU node** because a mace-jax
bundle pulls in ``cuequivariance``/CUDA and cannot load in the CPU host process
where a selector normally runs.

:class:`CommitteeUncertaintySelector` is therefore a first-class
:class:`remote_handler.workflows.active_learning_base.Selector` *actor* (the ABC
explicitly blesses "selection that needs its own execution ... as a remote job").
Each iteration it:

1. runs **one** ``RemoteTrajectoryProcessor`` over the sampler's pool with the
   :class:`~jaxrens_al.run.processors.committee_eval.CommitteeEvalProcessor`
   step (loads the committee on a GPU, writes per-config energy std +
   per-atom force std under the shared uncertainty keys), then
2. hands the annotated pool to a stock ``UncertaintySelector`` for the cheap
   host-side ranking/subsampling.

So the GPU evaluation and the ranking are cleanly split, selection stays out of
the sampler (per the repo's rule), and all the normalization / threshold /
energy-vs-force-weight logic is reused from the stock selector. Wire it via
``StagedActiveLearningHandler.from_actors(selector=CommitteeUncertaintySelector(...))``
-- ``from_actors`` accepts a ``Selector`` actor directly.

The ``cluster`` argument must name a **GPU** ``ClusterConfig`` (mace-jax needs
CUDA); ``"local"`` runs the pipeline in-process (only viable for a CPU backend /
tests). Needs the ``[run]`` extra -- keep it off the dashboard import path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from remote_handler.workflows.active_learning_base import Selector
from remote_handler.workflows.selectors import UncertaintySelector

from jaxrens_al.keys import ENERGY_UNCERTAINTY_KEY, FORCES_UNCERTAINTY_KEY

logger = logging.getLogger(__name__)


class CommitteeUncertaintySelector(Selector):
    """Select candidates by the disagreement of a committee of trained MLIPs.

    Args:
        models: The committee members -- at least two mace-jax bundle
            *directories* (or neuralil ``.pkl`` files); passed to
            :class:`~jaxrens_al.run.processors.committee_eval.CommitteeEvalProcessor`.
        n_configs: Number of highest-uncertainty configs to select (may be
            overridden per iteration by a schedule kwarg of the same name). Can
            be ``None`` when ``threshold`` is set (pure threshold mode).
        cluster: Where to run the committee evaluation -- a ``ClusterConfig``
            name/path (resolved under ``REMOTE_HANDLER_CONFIG_DIR``), or
            ``"local"``/``None`` for scheduler-free in-process execution. Use a
            **GPU** cluster for mace-jax bundles.
        backend: jaxrens backend the members load with (``"mace"`` / etc.).
        energy_weight / forces_weight: Relative weights of the (normalized)
            energy- and force-uncertainty terms in the combined ranking score
            (default: forces only, matching the in-NS committee convention).
        threshold: Optional minimum score for selection (see
            ``UncertaintySelector``); combinable with ``n_configs``.
        max_neighbors: Static neighbour-buffer size for each backend call.
        shard_size / n_processors / num_parallel / loop_time: array-job knobs
            forwarded to the ``RemoteTrajectoryProcessor``.
        output_filename: Name of the written extxyz inside the stage calcdir.
        reuse: Skip re-annotating a pool whose committee output already exists
            and is no older than the input (a stage retry contacts no cluster).
    """

    def __init__(
        self,
        models,
        *,
        n_configs: int | None = None,
        cluster: str | None = "local",
        backend: str = "mace",
        energy_weight: float = 0.0,
        forces_weight: float = 1.0,
        threshold: float | None = None,
        max_neighbors: int = 50,
        shard_size: int = 512,
        n_processors: int = 1,
        num_parallel: int = 8,
        loop_time: float = 30.0,
        output_filename: str = "selected.extxyz",
        reuse: bool = True,
    ):
        self.models = [str(m) for m in models]
        if len(self.models) < 2:
            raise ValueError(
                "CommitteeUncertaintySelector needs at least two models."
            )
        self.cluster = cluster
        self.backend = backend
        self.energy_key = ENERGY_UNCERTAINTY_KEY
        self.forces_key = FORCES_UNCERTAINTY_KEY
        self.max_neighbors = int(max_neighbors)
        self.shard_size = int(shard_size)
        self.n_processors = int(n_processors)
        self.num_parallel = int(num_parallel)
        self.loop_time = float(loop_time)
        self.output_filename = output_filename
        self.reuse = bool(reuse)
        # The stock host-side ranker reads exactly the keys committee_eval writes.
        self._ranker = UncertaintySelector(
            n_configs=n_configs,
            energy_key=self.energy_key,
            forces_key=self.forces_key,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            threshold=threshold,
        )
        self._output: Path | None = None

    def update(self, calcdir, selector_input, **kwargs) -> None:
        import ase.io
        from ase import Atoms

        # A schedule may ramp how many candidates to keep per iteration.
        if kwargs.get("n_configs") is not None:
            self._ranker.n_configs = int(kwargs["n_configs"])

        paths = (
            [selector_input]
            if isinstance(selector_input, (str, Path))
            else list(selector_input)
        )
        annotated = self._annotate(calcdir, paths)

        # One group per input file, so a grouping-aware ranker still works;
        # UncertaintySelector flattens anyway.
        grouped: list[list[Atoms]] = []
        for p in annotated:
            frames = ase.io.read(str(p), index=":")
            if isinstance(frames, Atoms):
                frames = [frames]
            grouped.append(frames)

        selected = self._ranker.select(grouped)

        out_dir = Path(calcdir) / "selector"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / self.output_filename
        ase.io.write(str(out_path), selected)
        self._output = out_path
        logger.info(
            "CommitteeUncertaintySelector: wrote %d selected configs to %s",
            len(selected),
            out_path,
        )

    def _annotate(self, calcdir, paths) -> list[Path]:
        """Run the committee eval over ``paths``; return the annotated paths."""
        from remote_handler.core.atoms_shard import front_tag
        from remote_handler.processing import RemoteTrajectoryProcessor

        from jaxrens_al.run.processing_resolver import (
            _up_to_date,
            execution_context,
        )
        from jaxrens_al.run.processors.committee_eval import (
            CommitteeEvalProcessor,
        )
        from jaxrens_al.run.resolver_base import stable_tag

        workdir = Path(calcdir) / "committee_eval"
        ctx = execution_context(self.cluster, workdir)

        # Tag keyed by the committee + eval knobs, so a different committee
        # writes to a different output (no stale reuse) and an identical one
        # reuses its on-disk annotation.
        tag = stable_tag(
            "committee",
            (
                tuple(sorted(self.models)),
                self.backend,
                self.energy_key,
                self.forces_key,
                self.max_neighbors,
            ),
        )
        mapping = {p: Path(front_tag(str(p), tag)) for p in paths}
        todo = [
            p for p in paths if not (self.reuse and _up_to_date(p, mapping[p]))
        ]
        if todo:
            logger.info(
                "CommitteeUncertaintySelector: committee of %d over %d of %d "
                "file(s) (%d reused)",
                len(self.models),
                len(todo),
                len(paths),
                len(paths) - len(todo),
            )
            RemoteTrajectoryProcessor(
                traj_files=[str(p) for p in todo],
                cluster_config=ctx["cluster_config"],
                remote_fs=ctx["remote_fs"],
                n_processors=self.n_processors,
                shard_size=self.shard_size,
                savepath=str(ctx["remote_savepath"]),
                num_parallel=self.num_parallel,
                processors=[
                    CommitteeEvalProcessor(
                        self.models,
                        backend=self.backend,
                        energy_key=self.energy_key,
                        forces_key=self.forces_key,
                        max_neighbors=self.max_neighbors,
                    )
                ],
                tag=tag,
                local_savepath=ctx["local_savepath"],
                jobname="AL_COMMITTEE_EVAL",
                loop_time=self.loop_time,
                every=1,
            ).run()
        return [mapping[p] for p in paths]

    def get_selected_configs(self) -> Path:
        if self._output is None:
            raise RuntimeError(
                "CommitteeUncertaintySelector has not produced output yet."
            )
        return self._output
