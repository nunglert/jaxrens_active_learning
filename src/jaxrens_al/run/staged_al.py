"""Factories that assemble staged active-learning loops for jaxrens_al.

``remote_handler.workflows.StagedActiveLearningHandler`` is a *generic* stage-list
loop; its ``from_actors`` classmethod is a convenience constructor with
opinionated defaults and a single cross-ring hook (``selector_reference_from``,
hardwired to ``accumulate=True``). Rather than depend on that sugar -- and rather
than subclass the handler, whose run/resume/progress behaviour we do not change
(only the wiring differs, so a factory, not a subclass, is the right shape) --
these builders assemble the stage list themselves through the handler's **public**
``__init__`` / ``Stage`` / ``InputSource`` API.

That keeps every AL-topology decision here in jaxrens_al (which stage feeds the
selector its committee, whether the last iteration trains, ...) and touches no
``remote_handler`` internals, so it is robust to that package being an installed,
non-editable dependency. Two patterns are provided:

* :func:`build_committee_al_handler` -- committee-uncertainty selection, whose
  committee is the trainer's *latest* ensemble (``accumulate=False``, which
  ``from_actors`` cannot express).
* :func:`build_quests_al_handler` -- QUESTS structural-novelty selection, scored
  against the evaluator's *accumulated* labelled data (``accumulate=True``).

Both share :func:`_staged_al_handler`, which encodes the one canonical topology:
``sampler -> selector -> evaluator -> trainer``, entering at the sampler and
exiting at the sampler (so the final iteration only samples -- it does not train a
model that would never be sampled from; this matches ``from_actors``' default).
"""

from __future__ import annotations

import inspect
import logging

from remote_handler.workflows.selectors import (
    ConfigSelector,
    PipelineSelector,
    QUESTSSelector,
    StrideSelector,
)
from remote_handler.workflows.staged_active_learning import (
    InputSource,
    SelectorActor,
    Stage,
    StagedActiveLearningHandler,
)

logger = logging.getLogger(__name__)


def _hook_kwargs(on_stage_complete) -> dict:
    """``on_stage_complete`` for the handler, if it supports hooks.

    An older ``remote_handler`` has no such argument: the hooks are then
    dropped with a warning rather than crashing the loop at startup.
    """
    if not on_stage_complete:
        return {}
    params = inspect.signature(StagedActiveLearningHandler).parameters
    if "on_stage_complete" not in params:
        logger.warning(
            "This remote_handler's StagedActiveLearningHandler takes no "
            "on_stage_complete hooks; ignoring %r (update remote_handler).",
            on_stage_complete,
        )
        return {}
    return {"on_stage_complete": list(on_stage_complete)}


def _staged_al_handler(
    *,
    sampler,
    selector,
    evaluator,
    trainer,
    calchome,
    selector_extra_inputs: dict | None = None,
    num_iterations: int = 5,
    entry_stage: str = "sampler",
    exit_stage: str = "sampler",
    initial_trainer_out=None,
    initial_sampler_out=None,
    initial_evaluator_out=None,
    sampler_schedule=None,
    selector_schedule=None,
    evaluator_schedule=None,
    trainer_schedule=None,
    combine_sampler_outputs: bool = False,
    combine_evaluator_outputs: bool = True,
    combine_trainer_outputs: bool = False,
    sampler_output_extension: str = "extxyz",
    selector_output_extension: str = "extxyz",
    evaluator_output_extension: str = "extxyz",
    trainer_output_extension: str = "pkl",
    on_stage_complete: list | None = None,
) -> StagedActiveLearningHandler:
    """Assemble the canonical 4-stage AL loop from actors + selector wiring.

    The shared workhorse behind the public factories. ``selector_extra_inputs``
    is the selector stage's cross-ring secondary streams (e.g. the committee, or
    the novelty reference). A pure ``ConfigSelector`` strategy is wrapped in a
    ``SelectorActor``; a ready-made ``Selector`` actor is used as-is.

    Args:
        sampler / selector / evaluator / trainer: The four stage actors.
        calchome: Working directory for the loop.
        selector_extra_inputs: ``{kwarg -> InputSource}`` delivered to the
            selector's ``update`` (the strategy-specific wiring).
        num_iterations: Number of AL iterations.
        entry_stage / exit_stage: Where iteration 0 starts and the final
            iteration stops (both default to ``"sampler"`` -- the final iteration
            only samples).
        initial_trainer_out: Seed for the sampler before the trainer has run
            (e.g. a foundation model for iteration 0).
        initial_sampler_out / initial_evaluator_out: Seeds for the evaluator /
            trainer stages (usually unused).
        *_schedule: Optional per-stage ``Schedule``.
        combine_*_outputs: Whether a stage's consumer receives all completed
            iterations' outputs (default: only the evaluator combines, so the
            trainer always trains on the full labelled set).
        *_output_extension: Collected-output file extension per stage.
        on_stage_complete: Hooks called after every recorded stage step
            (e.g. :class:`jaxrens_al.run.dashboard_hook.DashboardHook`).
    """
    selector_actor = (
        SelectorActor(selector)
        if isinstance(selector, ConfigSelector)
        else selector
    )
    stages = [
        Stage(
            name="sampler",
            actor=sampler,
            combine=combine_sampler_outputs,
            schedule=sampler_schedule,
            output_extension=sampler_output_extension,
            initial_input=initial_trainer_out,
        ),
        Stage(
            name="selector",
            actor=selector_actor,
            schedule=selector_schedule,
            output_extension=selector_output_extension,
            extra_inputs=selector_extra_inputs or {},
        ),
        Stage(
            name="evaluator",
            actor=evaluator,
            combine=combine_evaluator_outputs,
            schedule=evaluator_schedule,
            output_extension=evaluator_output_extension,
            initial_input=initial_sampler_out,
        ),
        Stage(
            name="trainer",
            actor=trainer,
            combine=combine_trainer_outputs,
            schedule=trainer_schedule,
            output_extension=trainer_output_extension,
            initial_input=initial_evaluator_out,
        ),
    ]
    return StagedActiveLearningHandler(
        stages=stages,
        calchome=str(calchome),
        num_iterations=num_iterations,
        entry_stage=entry_stage,
        exit_stage=exit_stage,
        **_hook_kwargs(on_stage_complete),
    )


def build_committee_al_handler(
    *,
    sampler,
    selector,
    evaluator,
    trainer,
    calchome,
    committee_from: str = "trainer",
    **handler_kwargs,
) -> StagedActiveLearningHandler:
    """Build a committee-uncertainty AL loop.

    Wires the selector's committee to ``committee_from``'s **latest** output
    (``accumulate=False``) -- the cross-ring input ``from_actors`` cannot express
    -- delivered under the keyword the selector declares (its
    ``committee_input``). Iteration 0 has no producer output yet, so the committee
    is empty and the selector cold-starts.

    Args:
        sampler / selector / evaluator / trainer: The four stage actors. The
            selector must be in dynamic-committee mode -- it must expose a
            ``committee_input`` (the ``update`` keyword the committee is delivered
            under), e.g. :class:`~jaxrens_al.run.committee_selector.
            CommitteeUncertaintySelector` built with ``committee_input=...``.
        calchome: Working directory for the loop.
        committee_from: Stage whose per-iteration output is the committee
            (default ``"trainer"`` -- its trained ensemble).
        **handler_kwargs: Forwarded to :func:`_staged_al_handler`
            (``num_iterations``, schedules, ``combine_*_outputs``,
            ``*_output_extension``, ``initial_*_out``, ``entry_stage`` /
            ``exit_stage``).

    Raises:
        ValueError: If ``selector`` exposes no ``committee_input`` (i.e. is not
            in dynamic-committee mode).
    """
    committee_kwarg = getattr(selector, "committee_input", None)
    if not committee_kwarg:
        raise ValueError(
            "build_committee_al_handler needs a selector in dynamic-committee "
            "mode: it must expose a 'committee_input' (the update keyword the "
            "committee is delivered under). A static 'models=' committee does "
            "not use this builder."
        )
    return _staged_al_handler(
        sampler=sampler,
        selector=selector,
        evaluator=evaluator,
        trainer=trainer,
        calchome=calchome,
        selector_extra_inputs={
            committee_kwarg: InputSource(
                stage=committee_from,
                accumulate=False,  # this iteration's ensemble, not all history
                include_current=False,
            )
        },
        **handler_kwargs,
    )


def build_quests_al_handler(
    *,
    sampler,
    evaluator,
    trainer,
    calchome,
    n_configs: int,
    stride_pool: int | None = None,
    method: str = "msc",
    quests_kwargs: dict | None = None,
    reference_from: str = "evaluator",
    initial_reference=None,
    **handler_kwargs,
) -> StagedActiveLearningHandler:
    """Build a QUESTS structural-novelty AL loop.

    The selector is a ``QUESTSSelector`` in novelty mode, scored against
    ``reference_from``'s **accumulated** outputs (the labelled data built up so
    far, ``accumulate=True``); with no reference yet (iteration 0 and no
    ``initial_reference``) it falls back to diversity mode. When ``stride_pool``
    is set the QUESTS step sits behind a ``StrideSelector`` that caps the
    (descriptor-heavy) pool.

    Needs the ``quests`` package at *run* time (``QUESTSSelector`` imports it
    lazily inside its scoring); building the handler does not import it.

    Args:
        sampler / evaluator / trainer: The three stage actors.
        calchome: Working directory for the loop.
        n_configs: Number of configs QUESTS selects for the evaluator.
        stride_pool: Optional cap -- stride the pool to at most this many configs
            before QUESTS. ``None`` runs QUESTS on the whole pool.
        method: QUESTS diversity-mode strategy (``"msc"`` / ``"fps"`` /
            ``"random"`` / ``"k_means"``); ignored once a novelty reference is
            present.
        quests_kwargs: Extra ``QUESTSSelector`` kwargs (``k``, ``cutoff``,
            ``bandwidth``, ``pca_output``).
        reference_from: Stage whose accumulated outputs are the novelty reference
            (default ``"evaluator"`` -- the labelled data so far).
        initial_reference: Optional cold-start reference (e.g. the starting
            training set) so novelty is active from iteration 0.
        **handler_kwargs: Forwarded to :func:`_staged_al_handler` (see
            :func:`build_committee_al_handler`).
    """
    quests = QUESTSSelector(
        n_configs=n_configs, method=method, **(quests_kwargs or {})
    )
    selector = (
        PipelineSelector([StrideSelector(n_configs=stride_pool), quests])
        if stride_pool is not None
        else quests
    )
    return _staged_al_handler(
        sampler=sampler,
        selector=selector,
        evaluator=evaluator,
        trainer=trainer,
        calchome=calchome,
        selector_extra_inputs={
            "reference": InputSource(
                stage=reference_from,
                accumulate=True,  # novelty vs. ALL labelled data so far
                include_current=False,
                initial=initial_reference,
            )
        },
        **handler_kwargs,
    )
