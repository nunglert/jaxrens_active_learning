"""Tests for the staged-AL loop factories (committee / QUESTS).

Both assemble a ``StagedActiveLearningHandler`` from the public ``__init__`` /
``Stage`` / ``InputSource`` API -- no ``from_actors``. Construction never calls
the stage actors, so plain sentinels suffice; the tests assert the assembled
topology and the selector's cross-ring wiring. Needs the [run] extra
(remote_handler); self-skips otherwise.
"""

import pytest

pytest.importorskip("remote_handler.workflows.staged_active_learning")

from remote_handler.workflows.selectors import (  # noqa: E402
    ConfigSelector,
    PipelineSelector,
    QUESTSSelector,
)
from remote_handler.workflows.staged_active_learning import (  # noqa: E402
    SelectorActor,
    StagedActiveLearningHandler,
)

# Run-side: needs the producer package (remote_handler).
pytestmark = pytest.mark.integration


def _seed(tmp_path):
    """A sampler iter-0 model seed (a bundle dir), moved into calchome on init."""
    seed = tmp_path / "foundation.bundle"
    seed.mkdir()
    return seed


def _stage(handler, name):
    (st,) = [s for s in handler.stages if s.name == name]
    return st


# ---------------------------------------------------------------------------
# committee factory
# ---------------------------------------------------------------------------
def test_committee_builder_topology_and_wiring(tmp_path):
    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector
    from jaxrens_al.run.staged_al import build_committee_al_handler

    selector = CommitteeUncertaintySelector(
        committee_input="committee", n_configs=5, cluster="local"
    )
    handler = build_committee_al_handler(
        sampler=object(),
        selector=selector,
        evaluator=object(),
        trainer=object(),
        calchome=tmp_path / "calchome",
        num_iterations=3,
        initial_trainer_out=[_seed(tmp_path)],
    )
    assert isinstance(handler, StagedActiveLearningHandler)
    assert [s.name for s in handler.stages] == [
        "sampler",
        "selector",
        "evaluator",
        "trainer",
    ]
    # A Selector actor is used as-is (not wrapped in SelectorActor).
    assert _stage(handler, "selector").actor is selector

    # Committee = the trainer's LATEST output (accumulate=False), keyed by the
    # selector's committee_input.
    src = _stage(handler, "selector").extra_inputs["committee"]
    assert src.stage == "trainer"
    assert src.accumulate is False
    assert src.include_current is False


def test_committee_builder_exits_at_sampler(tmp_path):
    """The final iteration only samples (exit_stage defaults to 'sampler')."""
    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector
    from jaxrens_al.run.staged_al import build_committee_al_handler

    handler = build_committee_al_handler(
        sampler=object(),
        selector=CommitteeUncertaintySelector(
            committee_input="committee", n_configs=5
        ),
        evaluator=object(),
        trainer=object(),
        calchome=tmp_path / "calchome",
        num_iterations=2,
        initial_trainer_out=[_seed(tmp_path)],
    )
    assert handler.exit_index == 0  # sampler is stage 0
    # Last iteration's plan holds only the sampler; earlier ones the full ring.
    last = handler.num_iterations - 1
    last_stages = [si for it, si in handler.plan if it == last]
    assert last_stages == [0]


def test_committee_builder_requires_dynamic_committee(tmp_path):
    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector
    from jaxrens_al.run.staged_al import build_committee_al_handler

    static = CommitteeUncertaintySelector(
        models=[tmp_path / "m0", tmp_path / "m1"], n_configs=5
    )
    with pytest.raises(ValueError, match="dynamic-committee mode"):
        build_committee_al_handler(
            sampler=object(),
            selector=static,
            evaluator=object(),
            trainer=object(),
            calchome=tmp_path / "calchome",
        )


# ---------------------------------------------------------------------------
# QUESTS factory
# ---------------------------------------------------------------------------
def test_quests_builder_wraps_pipeline_and_wires_reference(tmp_path):
    from jaxrens_al.run.staged_al import build_quests_al_handler

    handler = build_quests_al_handler(
        sampler=object(),
        evaluator=object(),
        trainer=object(),
        calchome=tmp_path / "calchome",
        n_configs=20,
        stride_pool=200,
        num_iterations=3,
        initial_trainer_out=[_seed(tmp_path)],
    )
    assert [s.name for s in handler.stages] == [
        "sampler",
        "selector",
        "evaluator",
        "trainer",
    ]
    # A ConfigSelector strategy is wrapped in a SelectorActor; with a stride cap
    # it is a PipelineSelector.
    actor = _stage(handler, "selector").actor
    assert isinstance(actor, SelectorActor)
    assert isinstance(actor.selector, PipelineSelector)

    # Novelty reference = the evaluator's ACCUMULATED labelled data.
    src = _stage(handler, "selector").extra_inputs["reference"]
    assert src.stage == "evaluator"
    assert src.accumulate is True
    assert src.include_current is False


def test_quests_builder_without_stride_is_bare_quests(tmp_path):
    from jaxrens_al.run.staged_al import build_quests_al_handler

    handler = build_quests_al_handler(
        sampler=object(),
        evaluator=object(),
        trainer=object(),
        calchome=tmp_path / "calchome",
        n_configs=20,
        initial_trainer_out=[_seed(tmp_path)],
    )
    actor = _stage(handler, "selector").actor
    assert isinstance(actor, SelectorActor)
    assert isinstance(actor.selector, QUESTSSelector)
    assert isinstance(actor.selector, ConfigSelector)


def test_quests_builder_seeds_initial_reference(tmp_path):
    """A cold-start reference is registered as the selector's reference seed."""
    from jaxrens_al.run.staged_al import build_quests_al_handler

    train_db = tmp_path / "initial_train.extxyz"
    train_db.write_text("")  # a plain seed file, moved into calchome on init
    handler = build_quests_al_handler(
        sampler=object(),
        evaluator=object(),
        trainer=object(),
        calchome=tmp_path / "calchome",
        n_configs=20,
        initial_reference=train_db,
        initial_trainer_out=[_seed(tmp_path)],
    )
    # The InputSource carries the seed; the handler stashes it under
    # (selector, reference) in extra_seeds (relative to calchome).
    assert handler.extra_seeds[("selector", "reference")]
