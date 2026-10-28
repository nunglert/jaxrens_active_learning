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

The committee can be **static** (a fixed set of bundles passed as ``models=``) or
**dynamic** -- resolved each iteration from an ``update`` keyword argument
(``committee_input=``), which is how an AL loop feeds *this iteration's* trained
ensemble in. The trainer stage produces the committee, but in the sampler ->
selector -> evaluator -> trainer ring the trainer's output feeds the *sampler*
(the wrap-around edge), not the selector -- so the committee is a **cross-ring**
input: a named secondary stream from the trainer stage, delivered to
:meth:`update` under ``committee_input``. Wire it with
:func:`jaxrens_al.run.staged_al.build_committee_al_handler`, which sets that
stream to the trainer's *latest* output (``accumulate=False``) -- so the value is
already exactly this iteration's ensemble. On a cold-start iteration the ensemble
does not exist yet (fewer than ``min_committee`` members); the selector then
falls back to a deterministic stride over the pool, mirroring how
``QUESTSSelector`` falls back to diversity.

So the GPU evaluation and the ranking are cleanly split, selection stays out of
the sampler (per the repo's rule), and all the normalization / threshold /
energy-vs-force-weight logic is reused from the stock selector.

The ``cluster`` argument must name a **GPU** ``ClusterConfig`` (mace-jax needs
CUDA); ``"local"`` runs the pipeline in-process (only viable for a CPU backend /
tests). Needs the ``[run]`` extra -- keep it off the dashboard import path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from remote_handler.workflows.active_learning_base import Selector
from remote_handler.workflows.selectors import (
    StrideSelector,
    UncertaintySelector,
)

from jaxrens_al.keys import (
    COMMITTEE_FORCES_KEY,
    ENERGY_UNCERTAINTY_KEY,
    FORCES_UNCERTAINTY_KEY,
)

logger = logging.getLogger(__name__)


class RelativeForceUncertaintyRanker:
    """Rank a pool by *relative* committee force disagreement.

    The stock ``UncertaintySelector`` ranks on the mean per-atom magnitude of
    the force std. That is an **absolute** quantity, so it is dominated by
    configurations whose forces are large to begin with -- expanded and
    disordered, high-enthalpy cells. On run13's pool it sent ~90 % of the DFT
    budget to near-gas configurations (median 58 A^3/atom against a pool median
    of 26); see ``run13/analysis/pca_map``.

    This ranks on

        sigma_F = sum_ik s_i^(k) / sum_ik |fbar_i^(k)|

    where ``s_i^(k)`` is the committee std of force component ``k`` of atom
    ``i`` (exactly what ``forces_key`` stores) and ``fbar_i^(k)`` the committee
    mean (``mean_forces_key``). Dividing by the *total force magnitude* asks
    "how uncertain is this force relative to how large it is", which does not
    reward a configuration merely for being violent.

    The denominator uses **magnitudes**. Summing the signed components instead
    gives the net force, which vanishes identically by Newton's third law
    (measured at ~1e-16 against ~1e1 for the absolute sum), so the literal
    signed form is degenerate, not merely noisy.

    Duck-types the part of ``UncertaintySelector`` this module uses: a mutable
    ``n_configs`` and ``select(grouped) -> list[Atoms]``.
    """

    #: eV/A floor on the denominator, so a force-free cell cannot divide by 0.
    EPS = 1e-3

    def __init__(
        self,
        n_configs: int | None = None,
        *,
        forces_key: str = FORCES_UNCERTAINTY_KEY,
        mean_forces_key: str = COMMITTEE_FORCES_KEY,
        threshold: float | None = None,
    ):
        if n_configs is None and threshold is None:
            raise ValueError(
                "RelativeForceUncertaintyRanker needs n_configs or threshold."
            )
        self.n_configs = n_configs
        self.forces_key = forces_key
        self.mean_forces_key = mean_forces_key
        self.threshold = threshold

    def score(self, atoms) -> float:
        import numpy as np

        for key in (self.forces_key, self.mean_forces_key):
            if key not in atoms.arrays:
                raise KeyError(
                    f"RelativeForceUncertaintyRanker: '{key}' not in "
                    f"atoms.arrays (have: {list(atoms.arrays)}). The relative "
                    "metric needs BOTH the committee force std and its mean; "
                    "the mean is written by a jaxrens_al new enough to have "
                    "COMMITTEE_FORCES_KEY -- re-annotate with an updated "
                    "cluster-side install."
                )
        s = np.asarray(atoms.arrays[self.forces_key], dtype=float)
        fbar = np.asarray(atoms.arrays[self.mean_forces_key], dtype=float)
        return float(s.sum() / (np.abs(fbar).sum() + self.EPS))

    def select(self, configs, reference=None) -> list:
        import numpy as np

        flat = (
            [at for group in configs for at in group]
            if (configs and isinstance(configs[0], list))
            else list(configs)
        )
        if not flat:
            logger.warning("RelativeForceUncertaintyRanker: empty pool")
            return []

        scores = np.array([self.score(at) for at in flat])
        order = np.argsort(scores)[::-1]
        if self.threshold is not None:
            order = order[scores[order] >= self.threshold]
        if self.n_configs is not None:
            order = order[: min(self.n_configs, len(order))]

        selected = [flat[i] for i in order]
        for at, i in zip(selected, order):
            # The raw ratio, not a min-max rescaling: it is a physical
            # quantity (fractional force error) and worth keeping readable.
            at.info["uncertainty_score"] = float(scores[i])
        logger.info(
            "RelativeForceUncertaintyRanker: selected %d of %d "
            "(score range %.4g - %.4g)",
            len(selected),
            len(flat),
            scores.min(),
            scores.max(),
        )
        return selected


class CommitteeUncertaintySelector(Selector):
    """Select candidates by the disagreement of a committee of trained MLIPs.

    Provide the committee either statically (``models=``) or dynamically
    (``committee_input=``, the name of an ``update`` keyword whose value is the
    committee's bundle paths -- e.g. the trainer stage's per-iteration output);
    exactly one of the two.

    Args:
        models: A *static* committee -- at least ``min_committee`` mace-jax
            bundle *directories* (or neuralil ``.pkl`` files). Mutually
            exclusive with ``committee_input``.
        committee_input: Name of the ``update`` keyword carrying the committee's
            bundle paths, resolved fresh each iteration (an AL loop feeds the
            trained ensemble here). Wire it with
            :func:`jaxrens_al.run.staged_al.build_committee_al_handler`, which
            delivers the trainer's latest ensemble under this keyword. Mutually
            exclusive with ``models``.
        n_configs: Number of highest-uncertainty configs to select (may be
            overridden per iteration by a schedule ``n_configs`` kwarg). Can be
            ``None`` when ``threshold`` is set (pure threshold mode).
        cluster: Where to run the committee evaluation -- a ``ClusterConfig``
            name/path (resolved under ``REMOTE_HANDLER_CONFIG_DIR``) or object,
            or ``"local"``/``None`` for scheduler-free in-process execution. Use
            a **GPU** cluster for mace-jax bundles.
        backend: jaxrens backend the members load with (``"mace"`` / etc.).
        energy_weight / forces_weight: Relative weights of the (normalized)
            energy- and force-uncertainty terms in the combined ranking score
            (default: forces only, matching the in-NS committee convention).
            Only used by ``force_metric="absolute"``.
        force_metric: How force disagreement is turned into a ranking score.
            ``"absolute"`` (default, unchanged behaviour) is the stock
            ``UncertaintySelector``: the mean per-atom magnitude of the force
            std, optionally combined with the energy term. ``"relative"`` is
            :class:`RelativeForceUncertaintyRanker` --
            ``sum_ik s_i^(k) / sum_ik |fbar_i^(k)|`` -- which normalises by the
            structure's own force magnitude and so does not systematically
            spend the DFT budget on expanded, high-force cells (the bias
            documented in ``run13/analysis/pca_map``). It needs the committee
            *mean* forces, so the pool must be annotated by a jaxrens_al new
            enough to write ``COMMITTEE_FORCES_KEY``; ``energy_weight`` /
            ``forces_weight`` do not apply.
        threshold: Optional minimum score for selection (see
            ``UncertaintySelector``); combinable with ``n_configs``.
        pre_stride: If set, deterministically stride the pool down to at most
            this many configs *before* the committee GPU evaluation -- bounds
            the per-iteration eval cost over a large dead-point pool (mirrors the
            stride a ``PipelineSelector`` would put ahead of the ranker). Unset:
            evaluate the whole pool.
        min_committee: Minimum members for a committee evaluation (>= 2). Below
            it, the iteration cold-starts (see ``cold_start``).
        cold_start: When the committee has fewer than ``min_committee`` members
            (e.g. iteration 0, before any ensemble is trained), fall back to a
            deterministic stride of ``n_configs`` over the pool. If ``False``,
            raise instead.
        max_neighbors: Static neighbour-buffer size for each backend call
            (the starting bucket when ``max_neighbors_list`` is given).
        batch_size: Configs per vmapped backend call in the committee eval.
            Unset scores one config per call, which leaves the GPU idle on
            small cells; the fixed per-shard cost is the model load + JIT
            compile, so pair a larger ``batch_size`` with a larger
            ``shard_size`` to get the benefit.
        max_neighbors_list: Neighbour-bucket ladder for the committee eval --
            sizes the buffer from the geometry instead of pinning it at
            ``max_neighbors``, with the ladder bounding the compile count.
            Only used together with ``batch_size``.
        max_neighbors_offset: Headroom above the observed neighbour count.
        shard_size / n_processors / num_parallel / loop_time: array-job knobs
            forwarded to the ``RemoteTrajectoryProcessor``.
        output_filename: Name of the written extxyz inside the stage calcdir.
        reuse: Skip re-annotating a pool whose committee output already exists
            and is no older than the input (a stage retry contacts no cluster).
        single_file_ensemble: The committee is ONE checkpoint holding every
            member (a NeuralIL ensemble ``.pkl``, which is what
            ``NeuralilILTrainer`` emits) -- scored per member through the
            backend's ``members`` method. A single file then counts as a full
            committee (``min_committee`` does not apply). See
            :class:`NeuralILCommitteeUncertaintySelector`.
        supercell_trafo: Backend periodic-image expansion ``(a, b, c)`` for
            the committee eval -- match the NS config's
            ``backend.supercell_trafo``. ``None`` = backend default, which for
            NeuralIL is (1, 1, 1): too small whenever the model cutoff exceeds
            half the cell width.
    """

    def __init__(
        self,
        models=None,
        *,
        committee_input: str | None = None,
        n_configs: int | None = None,
        cluster: str | None = "local",
        backend: str = "mace",
        energy_weight: float = 0.0,
        forces_weight: float = 1.0,
        force_metric: str = "absolute",
        threshold: float | None = None,
        pre_stride: int | None = None,
        min_committee: int = 2,
        cold_start: bool = True,
        max_neighbors: int = 50,
        batch_size: int | None = None,
        max_neighbors_list=None,
        max_neighbors_offset: int = 4,
        single_file_ensemble: bool = False,
        supercell_trafo=None,
        shard_size: int = 512,
        n_processors: int = 1,
        num_parallel: int = 8,
        loop_time: float = 30.0,
        output_filename: str = "selected.extxyz",
        reuse: bool = True,
    ):
        if (models is None) == (committee_input is None):
            raise ValueError(
                "Provide exactly one of 'models' (static committee) or "
                "'committee_input' (resolved from an update kwarg)."
            )
        if min_committee < 2:
            raise ValueError("min_committee must be at least 2 (a committee).")
        self.single_file_ensemble = bool(single_file_ensemble)
        self.supercell_trafo = (
            tuple(int(x) for x in supercell_trafo) if supercell_trafo else None
        )
        self.static_models = (
            [str(m) for m in models] if models is not None else None
        )
        if self.static_models is not None:
            if self.single_file_ensemble and len(self.static_models) != 1:
                raise ValueError(
                    "single_file_ensemble needs exactly one static model (the "
                    f"ensemble checkpoint); got {len(self.static_models)}."
                )
            if (
                not self.single_file_ensemble
                and len(self.static_models) < min_committee
            ):
                raise ValueError(
                    f"A static committee needs at least {min_committee} "
                    f"models; got {len(self.static_models)}."
                )
        self.committee_input = committee_input
        self.pre_stride = int(pre_stride) if pre_stride is not None else None
        self.cluster = cluster
        self.backend = backend
        self.energy_key = ENERGY_UNCERTAINTY_KEY
        self.forces_key = FORCES_UNCERTAINTY_KEY
        self.mean_forces_key = COMMITTEE_FORCES_KEY
        if force_metric not in ("absolute", "relative"):
            raise ValueError(
                "force_metric must be 'absolute' or 'relative'; got "
                f"{force_metric!r}"
            )
        self.force_metric = force_metric
        # A single-file ensemble is a whole committee in one path.
        self.min_committee = (
            1 if self.single_file_ensemble else int(min_committee)
        )
        self.cold_start = bool(cold_start)
        self.max_neighbors = int(max_neighbors)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.max_neighbors_list = (
            sorted(int(b) for b in max_neighbors_list)
            if max_neighbors_list
            else []
        )
        self.max_neighbors_offset = int(max_neighbors_offset)
        self.shard_size = int(shard_size)
        self.n_processors = int(n_processors)
        self.num_parallel = int(num_parallel)
        self.loop_time = float(loop_time)
        self.output_filename = output_filename
        self.reuse = bool(reuse)
        # Both rankers read exactly the keys committee_eval writes, and both
        # expose a mutable n_configs + select(grouped) so a schedule can ramp
        # the count and `update` can call either one identically.
        if self.force_metric == "relative":
            self._ranker = RelativeForceUncertaintyRanker(
                n_configs=n_configs,
                forces_key=self.forces_key,
                mean_forces_key=self.mean_forces_key,
                threshold=threshold,
            )
        else:
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

        # A schedule may ramp how many candidates to keep per iteration.
        if kwargs.get("n_configs") is not None:
            self._ranker.n_configs = int(kwargs["n_configs"])

        paths = (
            [selector_input]
            if isinstance(selector_input, (str, Path))
            else list(selector_input)
        )
        committee = self._resolve_committee(kwargs)

        if len(committee) < self.min_committee:
            selected = self._cold_start(committee, paths)
        else:
            eval_paths = self._prestride(calcdir, paths)
            annotated = self._annotate(calcdir, eval_paths, committee)
            selected = self._ranker.select(self._read_groups(annotated))

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

    def _resolve_committee(self, kwargs) -> list[str]:
        """The committee for this iteration -- static, or from the kwarg.

        A dynamic committee arrives under ``committee_input``, wired to a
        producer stage's latest output (``accumulate=False``) by
        :func:`jaxrens_al.run.staged_al.build_committee_al_handler`, so it is
        already exactly this iteration's ensemble -- no filtering needed.
        """
        if self.static_models is not None:
            return self.static_models
        raw = kwargs.get(self.committee_input)
        return [str(p) for p in raw] if raw else []

    @staticmethod
    def _read_groups(paths) -> list:
        """Read each file into one group (``list[list[Atoms]]``)."""
        import ase.io
        from ase import Atoms

        grouped = []
        for p in paths:
            frames = ase.io.read(str(p), index=":")
            if isinstance(frames, Atoms):
                frames = [frames]
            grouped.append(frames)
        return grouped

    def _cold_start(self, committee, paths) -> list:
        """No (full) committee yet -> deterministic stride over the raw pool."""
        if not self.cold_start:
            raise ValueError(
                f"Committee has {len(committee)} member(s), fewer than the "
                f"required {self.min_committee}, and cold_start is disabled."
            )
        grouped = self._read_groups(paths)
        n = self._ranker.n_configs
        logger.info(
            "CommitteeUncertaintySelector: cold start (%d < %d committee "
            "members) -- striding the pool to %s configs",
            len(committee),
            self.min_committee,
            n if n is not None else "all",
        )
        if n is None:  # no cap without a committee to score -> keep the pool
            return [at for group in grouped for at in group]
        return StrideSelector(n).select(grouped)

    def _prestride(self, calcdir, paths) -> list:
        """Cap the pool to ``pre_stride`` configs before the GPU eval.

        Writes the strided pool to one file and returns ``[that file]``; a no-op
        (returns ``paths`` unchanged) when ``pre_stride`` is unset or the pool
        already fits.
        """
        if self.pre_stride is None:
            return paths
        import ase.io

        grouped = self._read_groups(paths)
        n_total = sum(len(g) for g in grouped)
        if n_total <= self.pre_stride:
            return paths
        capped = StrideSelector(self.pre_stride).select(grouped)
        out = Path(calcdir) / "selector" / "pool_capped.extxyz"
        os.makedirs(out.parent, exist_ok=True)
        ase.io.write(str(out), capped)
        logger.info(
            "CommitteeUncertaintySelector: pre-strided pool %d -> %d before "
            "committee eval",
            n_total,
            len(capped),
        )
        return [out]

    def _annotate(self, calcdir, paths, committee) -> list[Path]:
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

        # Tag keyed by the committee + eval knobs, so a different committee (each
        # AL iteration) writes to a different output while an identical one
        # reuses its on-disk annotation.
        tag = stable_tag(
            "committee",
            (
                tuple(sorted(committee)),
                self.backend,
                self.energy_key,
                self.forces_key,
                self.mean_forces_key,
                self.max_neighbors,
                self.batch_size,
                tuple(self.max_neighbors_list),
                self.max_neighbors_offset,
            )
            # Only when set, so a multi-file committee keeps its existing tag
            # (and its on-disk annotations stay reusable).
            + (
                (self.single_file_ensemble, self.supercell_trafo)
                if self.single_file_ensemble or self.supercell_trafo
                else ()
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
                len(committee),
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
                        committee,
                        backend=self.backend,
                        energy_key=self.energy_key,
                        forces_key=self.forces_key,
                        mean_forces_key=self.mean_forces_key,
                        max_neighbors=self.max_neighbors,
                        batch_size=self.batch_size,
                        max_neighbors_list=self.max_neighbors_list,
                        max_neighbors_offset=self.max_neighbors_offset,
                        single_file_ensemble=self.single_file_ensemble,
                        supercell_trafo=self.supercell_trafo,
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


class NeuralILCommitteeUncertaintySelector(CommitteeUncertaintySelector):
    """:class:`CommitteeUncertaintySelector` for a NeuralIL ensemble.

    ``NeuralilILTrainer`` writes its whole ensemble (``n_ensemble`` members)
    into ONE ``.pkl``, so the committee is a single file scored per member
    via the NeuralIL backend's ``members`` method (``single_file_ensemble``)
    rather than one model per file. Everything else -- the GPU job, the
    pre-stride, the absolute/relative rankers, the iteration-0 stride cold
    start, wiring via :func:`jaxrens_al.run.staged_al.
    build_committee_al_handler` -- is the parent's.

    ``supercell_trafo`` is required: the NeuralIL backend defaults to
    (1, 1, 1), which silently truncates the neighbour list whenever the model
    cutoff exceeds half the cell width. Pass the NS config's
    ``backend.supercell_trafo``.

    ``full_ensemble`` scores the uncertainty with EVERY trained member instead
    of the reduced file the loop hands to the sampler: the trainer's
    ``save_reduced`` keeps the best members by force error, which tends to
    lower their disagreement. Only this selector changes -- the NS keeps the
    reduced model. The full file is looked up through the recorded trainer
    output (see :func:`jaxrens_al.dashboard.run_layout.
    neuralil_full_ensemble_for`); a missing one raises instead of silently
    falling back to the reduced ensemble. Needs a dynamic committee
    (``committee_input``) fed by the trainer stage.
    """

    def __init__(
        self,
        models=None,
        *,
        supercell_trafo,
        full_ensemble=False,
        **kwargs,
    ):
        if models is not None and not isinstance(models, (list, tuple)):
            models = [models]
        if full_ensemble and models is not None:
            raise ValueError(
                "full_ensemble needs a dynamic committee (committee_input) fed "
                "by the trainer stage; a static 'models=' path is used as is."
            )
        super().__init__(
            models,
            backend="neuralil",
            single_file_ensemble=True,
            supercell_trafo=supercell_trafo,
            **kwargs,
        )
        self.full_ensemble = bool(full_ensemble)

    def _resolve_committee(self, kwargs) -> list[str]:
        committee = super()._resolve_committee(kwargs)
        if not self.full_ensemble:
            return committee
        from jaxrens_al.dashboard.run_layout import neuralil_full_ensemble_for

        full = []
        for path in committee:
            found = neuralil_full_ensemble_for(path)
            if found is None:
                raise FileNotFoundError(
                    f"full_ensemble=True but no full ensemble file was found "
                    f"for the trainer output {path} (expected "
                    f"<calchome>/<k>/training/results/full_model_<k>.pkl, "
                    f"written when the trainer runs with save_reduced). "
                    f"Refusing to fall back to the reduced ensemble."
                )
            full.append(str(found))
        return full
