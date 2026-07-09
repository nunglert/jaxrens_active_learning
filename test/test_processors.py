"""Tests for the run-side Steinhardt trajectory processor.

Needs the [run] extra (remote_handler's ``processing`` framework); the module
self-skips otherwise. The host-side spec + arg schema + registration are
exercised with NO cluster or jaxrens -- only remote_handler + ase on the host.
The one test that runs the actual ``jaxrens`` computation self-skips when
jaxrens is absent.
"""

import pytest

pytest.importorskip("remote_handler.processing")

from remote_handler.exceptions import ScriptInputError  # noqa: E402
from remote_handler.processing.base import Processor  # noqa: E402
from remote_handler.processing.registry import (  # noqa: E402
    REGISTRY,
    validate_args,
)

from jaxrens_al.processors import (  # noqa: E402
    SteinhardtArgs,
    SteinhardtProcessor,
)

# Run-side: needs the producer package (remote_handler).
pytestmark = pytest.mark.integration


def test_steinhardt_processor_is_a_processor_spec():
    proc = SteinhardtProcessor(r_cut=3.5, l_values=(4, 6))
    assert isinstance(proc, Processor)
    assert proc.name == "steinhardt"
    # Nothing to stage: jaxrens supplies the science from the cluster env.
    assert proc.shared_files() == []


def test_registration_module_defaults_to_this_module():
    """The framework auto-resolves it -- driver imports this to register."""
    assert (
        SteinhardtProcessor.registration_module
        == "jaxrens_al.processors.steinhardt"
    )


def test_build_args_round_trips_through_the_schema():
    proc = SteinhardtProcessor(r_cut=4.2, l_values=(4, 6), r_cut_min=1.5)
    args = proc.build_args("/remote/savepath")
    assert args == {"l_values": [4, 6], "r_cut": 4.2, "r_cut_min": 1.5}
    # The emitted payload passes the same schema the driver validates against.
    SteinhardtArgs.from_dict(args).validate()


def test_processor_validates_l_values():
    with pytest.raises(ValueError, match="at least one l value"):
        SteinhardtProcessor(r_cut=3.5, l_values=())


def test_driver_registered_steinhardt_with_schema():
    """Importing the module registered the step's run + arg schema."""
    assert "steinhardt" in REGISTRY
    # validate_args resolves the schema registered under the name.
    validate_args("steinhardt", {"l_values": [6], "r_cut": 3.5})


def test_validate_args_rejects_bad_payload():
    with pytest.raises(ScriptInputError):
        validate_args("steinhardt", {"l_values": [], "r_cut": 3.5})
    with pytest.raises(ScriptInputError):
        validate_args("steinhardt", {"l_values": [6], "r_cut": -1.0})
    with pytest.raises(ScriptInputError):  # r_cut_min must be < r_cut
        validate_args(
            "steinhardt", {"l_values": [6], "r_cut": 3.5, "r_cut_min": 4.0}
        )


class _FakeHandler:
    """Stand-in for RemoteTrajectoryProcessor: on run() it writes each input's
    tagged output where the real handler's redistribute would (front_tag), with
    a synthetic annotation -- so the resolver logic is testable with no
    cluster."""

    calls = []

    def __init__(self, *, traj_files, processors, tag, **kw):
        self.traj_files = traj_files
        self.tag = tag
        _FakeHandler.calls.append(
            {"processors": processors, "tag": tag, "n_traj": len(traj_files)}
        )

    def run(self):
        import ase.io
        import numpy as np
        from remote_handler.core.atoms_shard import front_tag

        for src in self.traj_files:
            frames = ase.io.read(src, index=":")
            for at in frames:
                for key in ("q6", "w6"):
                    at.new_array(key, np.zeros(len(at)))
                    at.info[f"{key}_mean"] = 0.0
            ase.io.write(str(front_tag(src, self.tag)), frames)


def test_resolve_processing_dedups_into_one_job_and_delivers(
    monkeypatch, tmp_path
):
    """Two panels on the same pool + need -> one step, one job, both fed."""
    import ase.io
    from ase import Atoms

    from jaxrens_al.panels.steinhardt import (
        SteinhardtDistributionPanel,
        SteinhardtHeatmapPanel,
    )
    from jaxrens_al.processing_resolver import resolve_processing

    pool = tmp_path / "pool.extxyz"
    ase.io.write(
        str(pool),
        [
            Atoms(
                "Cu2O",
                positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                cell=[5, 5, 5],
                pbc=True,
            )
            for _ in range(3)
        ],
    )
    _FakeHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor", _FakeHandler
    )

    p1 = SteinhardtDistributionPanel(
        configs=pool, r_cut=3.5, l_values=(6,), datapath=tmp_path
    )
    p2 = SteinhardtHeatmapPanel(
        configs=pool, r_cut=3.5, l=6, datapath=tmp_path
    )
    ctx = dict(cluster_config=None, remote_fs=None, remote_savepath="/remote")
    resolve_processing([p1, p2], **ctx)

    # Identical needs -> one deduped step, one job over the single pool.
    assert len(_FakeHandler.calls) == 1
    assert len(_FakeHandler.calls[0]["processors"]) == 1
    assert _FakeHandler.calls[0]["n_traj"] == 1

    # Both panels received the annotated pool and can read it (3 configs x 3).
    assert len(p1.compute_data({})["q"]["6"]) == 9

    # Reuse: a fresh panel with the same need finds the artifact on disk.
    p3 = SteinhardtDistributionPanel(
        configs=pool, r_cut=3.5, l_values=(6,), datapath=tmp_path
    )
    resolve_processing([p3], **ctx)
    assert len(_FakeHandler.calls) == 1


def _write_pool(path):
    import ase.io
    from ase import Atoms

    ase.io.write(
        str(path),
        [
            Atoms(
                "Cu2O",
                positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                cell=[5, 5, 5],
                pbc=True,
            )
            for _ in range(2)
        ],
    )


def _fresh_panel(configs, tmp_path):
    """A brand-new panel object, as a re-render would build."""
    from jaxrens_al.panels.steinhardt import SteinhardtDistributionPanel

    return SteinhardtDistributionPanel(
        configs=configs, r_cut=3.5, l_values=(6,), datapath=tmp_path
    )


def test_resolve_processing_only_annotates_missing_inputs(
    monkeypatch, tmp_path
):
    """A re-render with an added trajectory annotates only the new one."""
    from jaxrens_al.processing_resolver import resolve_processing

    pool_a = tmp_path / "a.extxyz"
    _write_pool(pool_a)
    _FakeHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor", _FakeHandler
    )
    ctx = dict(cluster_config=None, remote_fs=None, remote_savepath="/remote")

    resolve_processing([_fresh_panel(pool_a, tmp_path)], **ctx)
    assert _FakeHandler.calls[-1]["n_traj"] == 1  # a annotated

    # Re-render: a second trajectory joins; only it should be processed, a
    # reused from disk (both panels fresh, as a real re-render builds them).
    pool_b = tmp_path / "b.extxyz"
    _write_pool(pool_b)
    resolve_processing(
        [_fresh_panel(pool_a, tmp_path), _fresh_panel(pool_b, tmp_path)], **ctx
    )
    assert len(_FakeHandler.calls) == 2
    assert _FakeHandler.calls[-1]["n_traj"] == 1  # only b, a was reused


def test_resolve_processing_reannotates_when_input_is_newer(
    monkeypatch, tmp_path
):
    """A trajectory that changed after its annotation is re-processed."""
    import os

    from jaxrens_al.processing_resolver import resolve_processing

    pool = tmp_path / "pool.extxyz"
    _write_pool(pool)
    _FakeHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor", _FakeHandler
    )
    ctx = dict(cluster_config=None, remote_fs=None, remote_savepath="/remote")
    resolve_processing([_fresh_panel(pool, tmp_path)], **ctx)
    assert len(_FakeHandler.calls) == 1

    # Bump the input's mtime past its annotation -> stale -> re-annotate.
    future = os.path.getmtime(pool) + 100
    os.utime(pool, (future, future))
    resolve_processing([_fresh_panel(pool, tmp_path)], **ctx)
    assert len(_FakeHandler.calls) == 2


def test_execution_context_selects_local_or_config(tmp_path):
    """'local' builds a local context; a name goes to from_file."""
    from remote_handler.exceptions import ConfigurationError

    from jaxrens_al.processing_resolver import execution_context

    ctx = execution_context("local", tmp_path / "wd")
    assert ctx["cluster_config"].executor == "direct"
    assert ctx["cluster_config"].remote_machine == "local"
    assert ctx["remote_savepath"] and ctx["local_savepath"]

    # A missing config name is resolved via ClusterConfig.from_file (-> error).
    with pytest.raises(ConfigurationError):
        execution_context("/does/not/exist.json", tmp_path / "wd")


def test_run_annotates_atoms():
    """The cluster-side ``run`` attaches q_l/w_l via jaxrens (self-skips)."""
    pytest.importorskip("jaxrens")
    import numpy as np
    from ase import Atoms

    from jaxrens_al.processors import steinhardt

    rng = np.random.default_rng(0)
    atoms = Atoms(
        symbols=["Cu"] * 12 + ["O"] * 4,
        positions=rng.uniform(0, 6, (16, 3)),
        cell=[6, 6, 6],
        pbc=True,
    )
    out = steinhardt.run([atoms], {"l_values": [4, 6], "r_cut": 3.5})
    at = out[0]
    for key in ("q4", "w4", "q6", "w6"):
        assert key in at.arrays and len(at.arrays[key]) == len(at)
    assert "q6_mean" in at.info and "w4_mean" in at.info


def test_steps_merge_steinhardt_needs_by_cutoff():
    """Several Steinhardt panels (same r_cut, different l) -> ONE processor.

    A single pipeline can't hold two 'steinhardt' steps (remote_handler resolves
    processors by their fixed name), so needs sharing a cutoff must fold into one
    processor whose l_values are the union -- computed in a single pass.
    """
    from jaxrens_al.panels.processing import ProcessingNeed
    from jaxrens_al.processing_resolver import _steps_from_needs

    needs = [
        ProcessingNeed.make(
            ("t.extxyz",),
            "steinhardt",
            {"l_values": (6,), "r_cut": 3.5, "r_cut_min": 0.0},
        ),
        ProcessingNeed.make(
            ("t.extxyz",),
            "steinhardt",
            {"l_values": (4,), "r_cut": 3.5, "r_cut_min": 0.0},
        ),
    ]
    steps = _steps_from_needs(needs)
    assert len(steps) == 1
    assert steps[0].l_values == [4, 6]
    assert steps[0].name == "steinhardt"


def test_steps_reject_multiple_steinhardt_cutoffs():
    """Distinct r_cut can't coexist (cutoff-free q<l> keys) -> clear error."""
    from jaxrens_al.panels.processing import ProcessingNeed
    from jaxrens_al.processing_resolver import _steps_from_needs

    needs = [
        ProcessingNeed.make(
            ("t.extxyz",),
            "steinhardt",
            {"l_values": (6,), "r_cut": 3.5, "r_cut_min": 0.0},
        ),
        ProcessingNeed.make(
            ("t.extxyz",),
            "steinhardt",
            {"l_values": (6,), "r_cut": 3.0, "r_cut_min": 0.0},
        ),
    ]
    with pytest.raises(ValueError, match="share one"):
        _steps_from_needs(needs)


# ---------------------------------------------------------------------------
# EOS reference producer (build_eos_reference) -- a DFT-eval producer, not a
# trajectory Processor; lives in the same run-side subpackage.
# ---------------------------------------------------------------------------
def test_build_eos_reference_writes_per_phase_files(tmp_path, monkeypatch):
    import ase.io
    import numpy as np
    from ase import Atoms

    import jaxrens_al.reference_db as refdb
    from jaxrens_al.processors.eos_reference import build_eos_reference

    def _phase(mp_id):
        at = Atoms(
            "Sn2", positions=np.zeros((2, 3)), cell=np.eye(3) * 6, pbc=True
        )
        at.info["mp_id"] = mp_id
        return at

    monkeypatch.setattr(
        refdb,
        "fetch_mp_elemental",
        lambda *a, **k: [_phase("mp-1"), _phase("mp-2")],
    )

    class _FakeEvaluator:
        def update(self, calcdir, evaluator_input, **kw):
            self._in = str(evaluator_input)
            self._out = tmp_path / "out" / str(len(str(calcdir))) / "e.extxyz"

        def get_evaluated_configs(self):
            frames = ase.io.read(self._in, index=":")
            for at in frames:
                at.info["REF_energy"] = -1.0 * len(at)
                at.arrays["REF_forces"] = np.zeros((len(at), 3))
            self._out.parent.mkdir(parents=True, exist_ok=True)
            ase.io.write(str(self._out), frames)
            return self._out

    files = build_eos_reference(
        _FakeEvaluator(),
        element="Sn",
        vol_scalings=[0.98, 1.0, 1.02],
        target_cutoff=2.0,
        out_dir=tmp_path / "ref",
        workdir=tmp_path / "work",
    )
    assert {f.name for f in files} == {"eos_mp-1.extxyz", "eos_mp-2.extxyz"}
    for f in files:
        frames = ase.io.read(str(f), index=":")
        assert len(frames) == 3  # one per volume scaling
        assert all("REF_energy" in at.info for at in frames)
