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


def test_selector_rejects_single_model(tmp_path):
    from jaxrens_al.run.committee_selector import CommitteeUncertaintySelector

    with pytest.raises(ValueError, match="at least two"):
        CommitteeUncertaintySelector(models=[tmp_path / "only"])


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
