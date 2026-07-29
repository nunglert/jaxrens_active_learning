"""Tests for the committee-uncertainty selector actor.

Needs the [run] extra (remote_handler's workflows + processing); self-skips
otherwise. The committee GPU evaluation is faked (a stand-in
``RemoteTrajectoryProcessor`` writes the uncertainty keys where the real handler
would), so the actor's dispatch -> annotate -> host-side ranking flow is
exercised with no cluster, no jaxrens, no real model.
"""

import pytest

pytest.importorskip("remote_handler.processing")
pytest.importorskip("remote_handler.workflows.active_learning_base")

# Run-side: needs the producer package (remote_handler).
pytestmark = pytest.mark.integration


class _FakeCommitteeHandler:
    """Stand-in for RemoteTrajectoryProcessor: writes each frame's committee
    uncertainty keys where the real handler's redistribute would (front_tag),
    so the selector is testable without a cluster or real models."""

    calls = []

    def __init__(self, *, traj_files, processors, tag, **kw):
        self.traj_files = traj_files
        self.tag = tag
        self.proc = processors[0]
        _FakeCommitteeHandler.calls.append(
            {
                "tag": tag,
                "n_traj": len(traj_files),
                "n_models": len(self.proc.models),
            }
        )

    def run(self):
        import ase.io
        import numpy as np
        from remote_handler.core.atoms_shard import front_tag

        for src in self.traj_files:
            frames = ase.io.read(src, index=":")
            for j, at in enumerate(frames):
                # Increasing force disagreement per frame -> deterministic rank.
                at.info[self.proc.energy_key] = 0.0
                at.new_array(
                    self.proc.forces_key, np.full((len(at), 3), float(j))
                )
            ase.io.write(str(front_tag(src, self.tag)), frames)


def _pool(path, n):
    import ase.io
    from ase import Atoms

    ase.io.write(
        str(path),
        [
            Atoms(
                "H2",
                positions=[[0, 0, 0], [0, 0, 0.7]],
                cell=[5, 5, 5],
                pbc=True,
            )
            for _ in range(n)
        ],
    )


def test_selector_rejects_single_static_model(tmp_path):
    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector

    with pytest.raises(ValueError, match="at least 2"):
        CommitteeUncertaintySelector(models=[tmp_path / "only"])


def test_selector_requires_exactly_one_committee_source(tmp_path):
    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector

    with pytest.raises(ValueError, match="exactly one"):
        CommitteeUncertaintySelector()  # neither models nor committee_input
    with pytest.raises(ValueError, match="exactly one"):
        CommitteeUncertaintySelector(
            models=["a", "b"], committee_input="committee_models"
        )


def test_selector_is_a_selector_actor():
    from remote_handler.workflows.active_learning_base import Selector

    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector

    sel = CommitteeUncertaintySelector(models=["m0", "m1"], n_configs=2)
    assert isinstance(sel, Selector)
    # Not produced yet -> get_selected_configs raises rather than returning junk.
    with pytest.raises(RuntimeError, match="has not produced"):
        sel.get_selected_configs()


def test_selector_runs_committee_job_then_ranks(monkeypatch, tmp_path):
    """update -> one committee job -> stock UncertaintySelector picks top-N."""
    import ase.io

    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector

    pool = tmp_path / "pool.extxyz"
    _pool(pool, 4)  # force disagreement 0, 1, 2, 3 across the 4 frames

    _FakeCommitteeHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor",
        _FakeCommitteeHandler,
    )

    sel = CommitteeUncertaintySelector(
        models=[tmp_path / "m0", tmp_path / "m1"],
        n_configs=2,
        cluster="local",
    )
    calcdir = tmp_path / "iter0"
    calcdir.mkdir()
    sel.update(calcdir, pool)

    # Exactly one committee job over the single pool, with both members.
    assert len(_FakeCommitteeHandler.calls) == 1
    assert _FakeCommitteeHandler.calls[0]["n_models"] == 2
    assert _FakeCommitteeHandler.calls[0]["n_traj"] == 1

    out = sel.get_selected_configs()
    assert out == calcdir / "selector" / "selected.extxyz"
    selected = ase.io.read(str(out), index=":")
    # Top-2 by force disagreement are the last two frames (j = 3, 2).
    assert len(selected) == 2
    scores = sorted(at.info["uncertainty_score"] for at in selected)
    assert scores == pytest.approx([2 / 3, 1.0])


def test_selector_reuses_existing_annotation(monkeypatch, tmp_path):
    """A retry with the same committee + pool submits no new job."""
    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector

    pool = tmp_path / "pool.extxyz"
    _pool(pool, 3)
    _FakeCommitteeHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor",
        _FakeCommitteeHandler,
    )
    calcdir = tmp_path / "iter0"
    calcdir.mkdir()

    def _fresh():
        return CommitteeUncertaintySelector(
            models=[tmp_path / "m0", tmp_path / "m1"],
            n_configs=2,
            cluster="local",
        )

    _fresh().update(calcdir, pool)
    assert len(_FakeCommitteeHandler.calls) == 1
    # Same committee + pool -> annotation on disk is reused, no second job.
    _fresh().update(calcdir, pool)
    assert len(_FakeCommitteeHandler.calls) == 1


def test_dynamic_committee_from_update_kwarg(monkeypatch, tmp_path):
    """The committee is resolved per iteration from an update kwarg."""
    import ase.io

    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector

    pool = tmp_path / "pool.extxyz"
    _pool(pool, 4)
    _FakeCommitteeHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor",
        _FakeCommitteeHandler,
    )

    sel = CommitteeUncertaintySelector(
        committee_input="committee_models",
        n_configs=2,
        cluster="local",
    )
    calcdir = tmp_path / "iter1"
    calcdir.mkdir()
    # An AL loop feeds the trainer stage's ensemble here (3 bundles).
    sel.update(
        calcdir,
        pool,
        committee_models=[tmp_path / "m0", tmp_path / "m1", tmp_path / "m2"],
    )

    assert len(_FakeCommitteeHandler.calls) == 1
    assert _FakeCommitteeHandler.calls[0]["n_models"] == 3
    assert len(ase.io.read(str(sel.get_selected_configs()), index=":")) == 2


def test_cold_start_strides_when_committee_too_small(monkeypatch, tmp_path):
    """Iteration 0 (empty committee) falls back to a stride, no cluster job."""
    import ase.io

    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector

    pool = tmp_path / "pool.extxyz"
    _pool(pool, 6)
    _FakeCommitteeHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor",
        _FakeCommitteeHandler,
    )

    sel = CommitteeUncertaintySelector(
        committee_input="committee_models",
        n_configs=3,
        cluster="local",
    )
    calcdir = tmp_path / "iter0"
    calcdir.mkdir()
    sel.update(calcdir, pool)  # no committee_models -> cold start

    assert _FakeCommitteeHandler.calls == []  # no committee job ran
    assert len(ase.io.read(str(sel.get_selected_configs()), index=":")) == 3


def test_cold_start_can_be_disabled(tmp_path):
    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector

    pool = tmp_path / "pool.extxyz"
    _pool(pool, 4)
    sel = CommitteeUncertaintySelector(
        committee_input="committee_models",
        n_configs=2,
        cold_start=False,
    )
    calcdir = tmp_path / "iter0"
    calcdir.mkdir()
    with pytest.raises(ValueError, match="cold_start is disabled"):
        sel.update(calcdir, pool)  # single-member (empty) committee


def test_schedule_kwarg_overrides_n_configs(monkeypatch, tmp_path):
    """A per-iteration ``n_configs`` kwarg (from a Schedule) caps the count."""
    import ase.io

    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector

    pool = tmp_path / "pool.extxyz"
    _pool(pool, 5)
    _FakeCommitteeHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor",
        _FakeCommitteeHandler,
    )

    sel = CommitteeUncertaintySelector(
        models=[tmp_path / "m0", tmp_path / "m1"],
        n_configs=2,
        cluster="local",
    )
    calcdir = tmp_path / "iter0"
    calcdir.mkdir()
    sel.update(calcdir, pool, n_configs=3)  # schedule bumps 2 -> 3

    selected = ase.io.read(str(sel.get_selected_configs()), index=":")
    assert len(selected) == 3


# ---------------------------------------------------------------------------
# Relative force uncertainty: sum_ik s_i^(k) / sum_ik |fbar_i^(k)|.
# ---------------------------------------------------------------------------
def _annotated(sigma, fbar, **info):
    """A frame carrying a committee force std and mean."""
    import numpy as np
    from ase import Atoms

    n = len(sigma)
    at = Atoms(f"Sn{n}", positions=np.zeros((n, 3)), cell=[9] * 3, pbc=True)
    at.new_array("forces_uncertainty", np.asarray(sigma, dtype=float))
    at.new_array("committee_forces", np.asarray(fbar, dtype=float))
    at.info.update(info)
    return at


def test_relative_ranker_scores_the_quoted_formula():
    """score == sum of the stds / sum of |mean force| components."""
    import numpy as np

    from jaxrens_al.run.committee_selector import (
        RelativeForceUncertaintyRanker,
    )

    sigma = [[0.1, 0.2, 0.3], [0.4, 0.0, 0.0]]  # sum = 1.0
    fbar = [[1.0, -2.0, 0.0], [0.0, 0.0, 3.0]]  # sum |.| = 6.0
    r = RelativeForceUncertaintyRanker(n_configs=1)
    got = r.score(_annotated(sigma, fbar))
    assert got == pytest.approx(1.0 / (6.0 + r.EPS), rel=1e-9)


def test_relative_ranker_is_insensitive_to_overall_force_scale():
    """Scaling a structure's forces up must not make it look uncertain.

    This is the whole point: the absolute metric ranks a violent cell top just
    for being violent, the relative one does not.
    """
    from jaxrens_al.run.committee_selector import (
        RelativeForceUncertaintyRanker,
    )

    calm = _annotated([[0.10, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    violent = _annotated([[1.00, 0.0, 0.0]], [[10.0, 0.0, 0.0]])  # 10x both
    r = RelativeForceUncertaintyRanker(n_configs=1)
    r.EPS = 0.0
    assert r.score(calm) == pytest.approx(r.score(violent), rel=1e-9)

    # ... whereas the absolute magnitude ranks `violent` an order up.
    import numpy as np

    assert np.abs(violent.arrays["forces_uncertainty"]).sum() == pytest.approx(
        10 * np.abs(calm.arrays["forces_uncertainty"]).sum()
    )


def test_relative_ranker_picks_the_highest_ratio():
    from jaxrens_al.run.committee_selector import (
        RelativeForceUncertaintyRanker,
    )

    low = _annotated([[0.1, 0, 0]], [[10.0, 0, 0]], tag="low")  # 0.01
    high = _annotated([[0.5, 0, 0]], [[1.0, 0, 0]], tag="high")  # 0.5
    mid = _annotated([[0.2, 0, 0]], [[2.0, 0, 0]], tag="mid")  # 0.1
    ranker = RelativeForceUncertaintyRanker(n_configs=2)
    got = ranker.select([[low, high, mid]])
    assert [a.info["tag"] for a in got] == ["high", "mid"]
    # The raw ratio is kept as the score, not a min-max rescaling.
    assert got[0].info["uncertainty_score"] == pytest.approx(0.5, rel=1e-2)


def test_relative_ranker_needs_the_committee_mean_forces():
    """A pool annotated before COMMITTEE_FORCES_KEY existed fails loudly."""
    import numpy as np
    from ase import Atoms

    from jaxrens_al.run.committee_selector import (
        RelativeForceUncertaintyRanker,
    )

    at = Atoms("Sn", positions=np.zeros((1, 3)), cell=[9] * 3, pbc=True)
    at.new_array("forces_uncertainty", np.ones((1, 3)))
    with pytest.raises(KeyError, match="committee_forces"):
        RelativeForceUncertaintyRanker(n_configs=1).score(at)


def test_selector_force_metric_switches_the_ranker(tmp_path):
    from jaxrens_al.run.committee_selector import (
        CommitteeUncertaintySelector,
        RelativeForceUncertaintyRanker,
    )

    models = []
    for i in range(2):
        m = tmp_path / f"m{i}.pkl"
        m.write_text("x")
        models.append(m)

    default = CommitteeUncertaintySelector(models=models, n_configs=5)
    assert not isinstance(default._ranker, RelativeForceUncertaintyRanker)

    rel = CommitteeUncertaintySelector(
        models=models, n_configs=5, force_metric="relative"
    )
    assert isinstance(rel._ranker, RelativeForceUncertaintyRanker)
    # A schedule ramping n_configs must still reach the swapped-in ranker.
    rel._ranker.n_configs = 7
    assert rel._ranker.n_configs == 7

    with pytest.raises(ValueError, match="force_metric"):
        CommitteeUncertaintySelector(
            models=models, n_configs=5, force_metric="nonsense"
        )
