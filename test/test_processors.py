"""Tests for the run-side trajectory processors and processing resolver.

Needs the [run] extra (remote_handler's ``processing`` framework); the module
self-skips otherwise. The host-side specs + arg schemas + registration are
exercised with NO cluster or jaxrens -- only remote_handler + ase on the host.
The Steinhardt *compute* is now a remote_handler built-in (its processor lives
in ``remote_handler.processing.processors``, tested there); what remains here is
jaxrens_al's own resolver logic that wires that built-in from panel needs.
"""

import pytest

pytest.importorskip("remote_handler.processing")

from remote_handler.exceptions import ScriptInputError  # noqa: E402
from remote_handler.processing.base import Processor  # noqa: E402
from remote_handler.processing.registry import (  # noqa: E402
    REGISTRY,
    validate_args,
)

# Run-side: needs the producer package (remote_handler).
pytestmark = pytest.mark.integration


def test_steinhardt_builtin_is_available_from_remote_handler():
    """The Steinhardt compute now ships as a remote_handler built-in.

    jaxrens_al no longer owns the processor; the resolver imports it from
    remote_handler. This smoke-checks the built-in the resolver depends on is
    present + registered (its own correctness is tested in remote_handler).
    """
    from remote_handler.processing.processors import SteinhardtProcessor

    proc = SteinhardtProcessor(r_cut=3.5, l_values=(4, 6))
    assert isinstance(proc, Processor)
    assert proc.name == "steinhardt"
    assert "steinhardt" in REGISTRY
    validate_args("steinhardt", {"l_values": [6], "r_cut": 3.5})


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

    from jaxrens_al.dashboard.panels.steinhardt import (
        SteinhardtDistributionPanel,
        SteinhardtHeatmapPanel,
    )
    from jaxrens_al.run.processing_resolver import resolve_processing

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
    from jaxrens_al.dashboard.panels.steinhardt import (
        SteinhardtDistributionPanel,
    )

    return SteinhardtDistributionPanel(
        configs=configs, r_cut=3.5, l_values=(6,), datapath=tmp_path
    )


def test_resolve_processing_only_annotates_missing_inputs(
    monkeypatch, tmp_path
):
    """A re-render with an added trajectory annotates only the new one."""
    from jaxrens_al.run.processing_resolver import resolve_processing

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

    from jaxrens_al.run.processing_resolver import resolve_processing

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

    from jaxrens_al.run.processing_resolver import execution_context

    ctx = execution_context("local", tmp_path / "wd")
    assert ctx["cluster_config"].executor == "direct"
    assert ctx["cluster_config"].remote_machine == "local"
    assert ctx["remote_savepath"] and ctx["local_savepath"]

    # A missing config name is resolved via ClusterConfig.from_file (-> error).
    with pytest.raises(ConfigurationError):
        execution_context("/does/not/exist.json", tmp_path / "wd")


def test_steps_merge_steinhardt_needs_by_cutoff():
    """Several Steinhardt panels (same r_cut, different l) -> ONE processor.

    A single pipeline can't hold two 'steinhardt' steps (remote_handler resolves
    processors by their fixed name), so needs sharing a cutoff must fold into one
    processor whose l_values are the union -- computed in a single pass.
    """
    from jaxrens_al.dashboard.panels.processing import ProcessingNeed
    from jaxrens_al.run.processing_resolver import _steps_from_needs

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
    from jaxrens_al.dashboard.panels.processing import ProcessingNeed
    from jaxrens_al.run.processing_resolver import _steps_from_needs

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


def test_steps_build_one_rdf_processor_and_reject_mixed_grids():
    """RDF panels differing only in their *reduction* share one need; two
    different (cutoff, nbins) grids can't be annotated in one pass."""
    from jaxrens_al.dashboard.panels.processing import ProcessingNeed
    from jaxrens_al.run.processing_resolver import _steps_from_needs

    same = [
        ProcessingNeed.make(
            ("t.extxyz",), "rdf", {"cutoff": 6.0, "nbins": 120}
        ),
        ProcessingNeed.make(
            ("u.extxyz",), "rdf", {"cutoff": 6.0, "nbins": 120}
        ),
    ]
    steps = _steps_from_needs(same)
    assert len(steps) == 1
    assert steps[0].name == "rdf"
    assert (steps[0].cutoff, steps[0].nbins) == (6.0, 120)

    mixed = same + [
        ProcessingNeed.make(
            ("t.extxyz",), "rdf", {"cutoff": 5.0, "nbins": 120}
        ),
    ]
    with pytest.raises(ValueError, match="share one"):
        _steps_from_needs(mixed)


# ---------------------------------------------------------------------------
# model_eval processor + resolver (remote MLIP evaluation for EOS / parity).
# ---------------------------------------------------------------------------
def test_model_eval_processor_is_a_processor_spec(tmp_path):
    from jaxrens_al.run.processors import ModelEvalArgs, ModelEvalProcessor

    # A bundle *directory* is staged as a single .tar.
    bundle = tmp_path / "trainer_it0.bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    (bundle / "params.msgpack").write_bytes(b"\x00")

    proc = ModelEvalProcessor(
        bundle, backend="mace", energy_key="me", want_forces=True
    )
    assert isinstance(proc, Processor)
    assert proc.name == "model_eval"
    shared = proc.shared_files()
    assert len(shared) == 1 and shared[0].endswith(".tar")

    args = proc.build_args("/remote/savepath")
    assert args["model_remote"] == "/remote/savepath/trainer_it0.bundle.tar"
    assert args["is_archive"] is True and args["backend"] == "mace"
    assert args["want_forces"] is True
    ModelEvalArgs.from_dict(args).validate()


def test_model_eval_processor_forwards_the_bucket_ladder(tmp_path):
    from jaxrens_al.run.processors import ModelEvalArgs, ModelEvalProcessor

    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(b"\x00")
    proc = ModelEvalProcessor(
        pkl,
        backend="neuralil",
        batch_size=8,
        max_neighbors_list=[60, 20, 40],
        max_neighbors_offset=2,
    )
    args = proc.build_args("/remote")
    assert args["batch_size"] == 8
    assert args["max_neighbors_list"] == [20, 40, 60]  # sorted
    assert args["max_neighbors_offset"] == 2
    ModelEvalArgs.from_dict(args).validate()

    # Unset -> the static-buffer defaults (no ladder, no batching).
    plain = ModelEvalProcessor(pkl, backend="neuralil").build_args("/remote")
    # to_dict() drops None-valued fields, so an unset batch_size is absent.
    assert "batch_size" not in plain and plain["max_neighbors_list"] == []


def test_model_eval_args_reject_unsorted_ladder():
    from remote_handler.exceptions import ScriptInputError

    from jaxrens_al.run.processors import ModelEvalArgs

    args = ModelEvalArgs(model_remote="m", max_neighbors_list=[40, 20])
    with pytest.raises(ScriptInputError, match="ascending"):
        args.validate()


def test_model_eval_processor_stages_plain_file(tmp_path):
    from jaxrens_al.run.processors import ModelEvalProcessor

    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(b"\x00")
    proc = ModelEvalProcessor(pkl, backend="neuralil")
    assert proc.shared_files() == [str(pkl)]
    args = proc.build_args("/remote")
    assert args["is_archive"] is False
    assert args["model_remote"] == "/remote/model.pkl"


class _FakeExecutor:
    """Stand-in for ``RemoteTask.executor``: the resolver's submit-all-then-
    collect-all split only touches ``wait_until_finished``."""

    def wait_until_finished(self):
        pass


class _FakeModelEvalHandler:
    """Fake RemoteTrajectoryProcessor: writes each model's keys onto the frames
    where the real handler's redistribute would (front_tag), so the resolver is
    testable without a cluster or a real model. Exposes the same
    allocate_data()/submit(only_start=)/executor.wait_until_finished()/
    collect_and_harvest() split the resolver's submit-all-then-collect-all
    pattern (:func:`~jaxrens_al.run.processing_resolver._submit_pending` /
    ``_collect_pending``) calls instead of the blocking ``run()``."""

    calls = []

    def __init__(self, *, traj_files, processors, tag, **kw):
        self.traj_files = traj_files
        self.tag = tag
        self.proc = processors[0]
        self.executor = _FakeExecutor()
        _FakeModelEvalHandler.calls.append(
            {
                "tag": tag,
                "n_traj": len(traj_files),
                "model": self.proc.model,
                "proc": self.proc,
            }
        )

    def allocate_data(self):
        pass

    def submit(self, only_start=False):
        pass

    def collect_and_harvest(self):
        import ase.io
        import numpy as np
        from remote_handler.core.atoms_shard import front_tag

        for src in self.traj_files:
            frames = ase.io.read(src, index=":")
            for at in frames:
                at.info[self.proc.energy_key] = -1.0 * len(at)
                if self.proc.want_forces:
                    at.new_array(self.proc.forces_key, np.zeros((len(at), 3)))
            ase.io.write(str(front_tag(src, self.tag)), frames)


def _refs(tmp_path, name):
    import ase.io
    from ase import Atoms

    path = tmp_path / name
    ase.io.write(
        str(path),
        [Atoms("Cu", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)],
    )
    return path


def test_resolve_model_evaluations_one_job_per_model(monkeypatch, tmp_path):
    """Two panels sharing a model -> one job; a second model -> a second job."""
    from jaxrens_al.dashboard.panels.model_eval import ModelEvalNeed
    from jaxrens_al.run.processing_resolver import resolve_model_evaluations

    ref = _refs(tmp_path, "eos_a.extxyz")
    _FakeModelEvalHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor",
        _FakeModelEvalHandler,
    )

    class _Panel:
        def __init__(self, model):
            self._model = model
            self._need = ModelEvalNeed.make([ref], model, "mace")
            self.received = None

        def model_eval_needs(self):
            return [self._need]

        def receive_model_eval(self, mapping):
            # Nested {model -> {input -> annotated}}: read our own model's map.
            self.received = mapping.get(self._model, {}).get(str(ref))

    ctx = dict(cluster_config=None, remote_fs=None, remote_savepath="/remote")
    p1, p2 = _Panel("bundleA"), _Panel("bundleA")  # same model
    resolve_model_evaluations([p1, p2], **ctx)
    assert len(_FakeModelEvalHandler.calls) == 1  # deduped to one job
    assert p1.received is not None and p1.received == p2.received

    # A distinct model on the SAME shared input triggers a second job; each
    # panel must read ITS model's annotation, not the last one written (the
    # EOS-across-iterations bug: shared reference files, per-iteration models).
    p1b, p3 = _Panel("bundleA"), _Panel("bundleB")
    resolve_model_evaluations([p1b, p3], **ctx)
    assert len(_FakeModelEvalHandler.calls) == 2  # only bundleB ran (A reused)
    # Each panel reads its own model's annotation (distinct tags); the shared
    # reference file did NOT get overwritten to bundleB for the bundleA panel.
    assert p1b.received != p3.received
    assert p1b.received == p1.received  # same model+input -> same annotation


def test_resolve_model_evaluations_forwards_ladder_keeps_static_tag(
    monkeypatch, tmp_path
):
    """A laddered need reaches the processor under its own tag, while a
    static-buffer need keeps its pre-ladder tag (old annotations reused)."""
    from jaxrens_al.dashboard.panels.model_eval import ModelEvalNeed
    from jaxrens_al.run.processing_resolver import (
        resolve_model_evaluations,
        stable_tag,
    )

    ref = _refs(tmp_path, "eos_a.extxyz")
    _FakeModelEvalHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor",
        _FakeModelEvalHandler,
    )

    class _Panel:
        def __init__(self, need):
            self._need = need

        def model_eval_needs(self):
            return [self._need]

        def receive_model_eval(self, mapping):
            pass

    static = ModelEvalNeed.make([ref], "bundleA", "mace")
    laddered = ModelEvalNeed.make(
        [ref], "bundleA", "mace", max_neighbors_list=[20, 40, 60]
    )
    ctx = dict(cluster_config=None, remote_fs=None, remote_savepath="/remote")
    resolve_model_evaluations([_Panel(static), _Panel(laddered)], **ctx)

    calls = {
        tuple(c["proc"].max_neighbors_list): c
        for c in _FakeModelEvalHandler.calls
    }
    assert set(calls) == {(), (20, 40, 60)}  # two distinct jobs
    assert calls[()]["tag"] == stable_tag(
        "meval", ("bundleA", "mace", False, 50, None)
    )
    assert calls[(20, 40, 60)]["tag"] != calls[()]["tag"]


def test_run_model_eval_annotates_atoms(tmp_path):
    """The cluster-side ``run`` writes model energy via jaxrens (self-skips)."""
    pytest.importorskip("jaxrens")
    import numpy as np
    from ase import Atoms
    from jaxrens.backends.base import BackendResult

    from jaxrens_al.run.processors import model_eval

    atoms = Atoms(
        "Cu2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[6, 6, 6], pbc=True
    )

    def _fake_backend(*a, **k):
        class _B:
            atomic_numbers = [29]
            symbol_map = {"Cu": 0}

            def __call__(self, pos, types, cell, mn):
                # A real backend returns a BackendResult; score_atoms reads
                # its .overflow to reject a truncated neighbour list.
                return BackendResult(energy=np.float64(-3.5))

        return _B()

    import jaxrens.backends.loader as loader

    orig = loader.load_backend
    loader.load_backend = _fake_backend
    try:
        out = model_eval.run(
            [atoms],
            {
                "backend": "mace",
                "model_remote": "ignored",
                "is_archive": False,
                "energy_key": "me",
                "want_forces": False,
                "max_neighbors": 10,
            },
        )
    finally:
        loader.load_backend = orig
    assert out[0].info["me"] == -3.5


def test_score_atoms_min_image_cutoff_scales_energy_per_input_cell():
    """A unit cell scored via a cutoff supercell returns per-input-cell energy.

    The backend sees the (larger) supercell and returns its total energy; the
    scaling divides it by the replication factor, so a caller dividing by the
    *input* cell's atom count still gets the per-atom energy -- the property the
    EOS panel relies on to compare unit-cell DFT to a supercell model score.
    """
    pytest.importorskip("jaxrens")
    import numpy as np
    from ase import Atoms
    from jaxrens.backends.base import BackendResult

    from jaxrens_al.dashboard.panels.model_eval import score_atoms
    from jaxrens_al.supercell import min_image_supercell

    a = 5.43  # 8-atom diamond-Si cell, max sphere radius 2.715 < 6.0
    base = Atoms(
        "Si8",
        cell=[a, a, a],
        pbc=True,
        scaled_positions=[
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
    )
    rep = len(min_image_supercell(base, 6.0)) // len(base)
    assert rep > 1  # this cell really does need replicating

    seen = {}

    def _fake_backend(*a, **k):
        class _B:
            atomic_numbers = [14]
            symbol_map = {"Si": 0}

            def __call__(self, pos, types, cell, mn):
                seen["n"] = len(types)  # backend sees the supercell
                return BackendResult(energy=np.float64(-80.0))

        return _B()

    import jaxrens.backends.loader as loader

    orig = loader.load_backend
    loader.load_backend = _fake_backend
    try:
        energies, _ = score_atoms(
            "ignored", "mace", [base], min_image_cutoff=6.0
        )
    finally:
        loader.load_backend = orig

    assert seen["n"] == len(base) * rep  # scored on the supercell
    assert energies[0] == pytest.approx(-80.0 / rep)  # scaled back per cell

    # Forces + a supercell cutoff is rejected (per-atom forces can't map back).
    with pytest.raises(ValueError, match="want_forces"):
        score_atoms(
            "x", "mace", [base], want_forces=True, min_image_cutoff=6.0
        )


def _stub_backend(needed: int = 28):
    """A vmap-able, differentiable stand-in for a real jaxrens backend.

    Reports ``needed`` neighbours per atom regardless of geometry and flags
    ``overflow`` whenever the requested bucket cannot hold them, which is what
    drives the ladder sizing and the escalate-and-retry path.
    """
    import jax.numpy as jnp
    from jaxrens.backends.base import BackendResult

    class _B:
        atomic_numbers = [14]
        symbol_map = {"Si": 0}
        calls: list[int] = []

        def __call__(self, pos, types, cell, mn, ensemble_params=None):
            _B.calls.append(int(mn))
            return BackendResult(
                energy=jnp.sum(pos**2),
                max_neighbor_count=jnp.asarray(needed),
                overflow=jnp.asarray(needed > mn),
            )

        def max_neighbors_for(self, pos, cell):
            return jnp.asarray(needed)

    return _B


def _si_cells(n_frames, n_atoms=4, seed=0):
    import numpy as np
    from ase import Atoms

    rng = np.random.default_rng(seed)
    return [
        Atoms(
            f"Si{n_atoms}",
            positions=rng.random((n_atoms, 3)) * 4.0,
            cell=[5.43] * 3,
            pbc=True,
        )
        for _ in range(n_frames)
    ]


@pytest.mark.parametrize("want_forces", [False, True])
def test_score_atoms_batched_matches_the_unbatched_loop(
    monkeypatch, want_forces
):
    """vmapped batching is a pure speed change: same numbers, same order."""
    pytest.importorskip("jaxrens")
    import jaxrens.backends.loader as loader
    import numpy as np

    from jaxrens_al.dashboard.panels.model_eval import score_atoms

    B = _stub_backend()
    monkeypatch.setattr(loader, "load_backend", lambda *a, **k: B())
    frames = _si_cells(7)

    ref_e, ref_f = score_atoms(
        "m", "mace", frames, want_forces=want_forces, max_neighbors=50
    )
    bat_e, bat_f = score_atoms(
        "m",
        "mace",
        frames,
        want_forces=want_forces,
        max_neighbors=50,
        batch_size=3,
    )
    np.testing.assert_allclose(bat_e, ref_e, rtol=1e-6)
    assert len(bat_e) == len(frames)  # padding of the ragged tail is dropped
    if want_forces:
        for a, b in zip(ref_f, bat_f):
            np.testing.assert_allclose(b, a, rtol=1e-6)


def test_score_atoms_batches_group_by_atom_count(monkeypatch):
    """Mixed cell sizes still work -- they cannot stack into one batch."""
    pytest.importorskip("jaxrens")
    import jaxrens.backends.loader as loader
    import numpy as np

    from jaxrens_al.dashboard.panels.model_eval import score_atoms

    B = _stub_backend()
    monkeypatch.setattr(loader, "load_backend", lambda *a, **k: B())
    frames = _si_cells(3, n_atoms=4) + _si_cells(2, n_atoms=6, seed=1)

    ref, _ = score_atoms("m", "mace", frames, max_neighbors=50)
    bat, _ = score_atoms("m", "mace", frames, max_neighbors=50, batch_size=4)
    np.testing.assert_allclose(bat, ref, rtol=1e-6)


def test_score_atoms_sizes_the_neighbour_bucket_from_the_ladder(monkeypatch):
    """With a ladder the bucket is chosen from geometry, not pinned at 50."""
    pytest.importorskip("jaxrens")

    import jaxrens.backends.loader as loader

    from jaxrens_al.dashboard.panels.model_eval import score_atoms

    B = _stub_backend(needed=28)
    B.calls.clear()
    monkeypatch.setattr(loader, "load_backend", lambda *a, **k: B())

    score_atoms(
        "m",
        "mace",
        _si_cells(4),
        max_neighbors=50,
        batch_size=2,
        max_neighbors_list=[20, 30, 40, 50, 60],
        max_neighbors_offset=4,
    )
    # 28 + 4 = 32 -> the smallest ladder entry that fits is 40, not the
    # fixed 50 the call was given.
    assert set(B.calls) == {40}


def test_score_atoms_ladder_without_batch_size_still_buckets(monkeypatch):
    """A ladder alone (no batch_size) is honoured, not silently ignored."""
    pytest.importorskip("jaxrens")

    import jaxrens.backends.loader as loader

    from jaxrens_al.dashboard.panels.model_eval import score_atoms

    B = _stub_backend(needed=28)
    B.calls.clear()
    monkeypatch.setattr(loader, "load_backend", lambda *a, **k: B())

    score_atoms(
        "m",
        "mace",
        _si_cells(3),
        max_neighbors=50,
        max_neighbors_list=[20, 30, 40, 50, 60],
        max_neighbors_offset=4,
    )
    assert set(B.calls) == {40}


def test_score_atoms_grows_the_bucket_on_overflow(monkeypatch):
    """An overflow report escalates to the next ladder entry and retries."""
    pytest.importorskip("jaxrens")

    # Geometry sizing under-reports (says 8), the forward pass really needs 44.
    import jax.numpy as jnp
    import jaxrens.backends.loader as loader
    from jaxrens.backends.base import BackendResult

    from jaxrens_al.dashboard.panels.model_eval import score_atoms

    class _B:
        atomic_numbers = [14]
        symbol_map = {"Si": 0}
        calls: list[int] = []

        def __call__(self, pos, types, cell, mn, ensemble_params=None):
            _B.calls.append(int(mn))
            return BackendResult(
                energy=jnp.sum(pos**2),
                max_neighbor_count=jnp.asarray(44),
                overflow=jnp.asarray(44 > mn),
            )

        def max_neighbors_for(self, pos, cell):
            return jnp.asarray(8)

    monkeypatch.setattr(loader, "load_backend", lambda *a, **k: _B())
    score_atoms(
        "m",
        "mace",
        _si_cells(2),
        batch_size=2,
        max_neighbors_list=[20, 30, 40, 50, 60],
        max_neighbors_offset=4,
    )
    # 8 + 4 -> 20, overflows; 44 + 4 = 48 -> grows straight to 50.
    assert _B.calls == [20, 50]


def test_score_atoms_ladder_exhaustion_is_an_error(monkeypatch):
    """A ladder that cannot hold the observed need fails loudly."""
    pytest.importorskip("jaxrens")

    import jaxrens.backends.loader as loader

    from jaxrens_al.dashboard.panels.model_eval import score_atoms

    B = _stub_backend(needed=100)
    monkeypatch.setattr(loader, "load_backend", lambda *a, **k: B())
    with pytest.raises(RuntimeError, match="ladder exhausted"):
        score_atoms(
            "m",
            "mace",
            _si_cells(2),
            batch_size=2,
            max_neighbors_list=[20, 30],
        )


# ---------------------------------------------------------------------------
# committee_eval processor (committee disagreement of N MLIPs).
# ---------------------------------------------------------------------------
def test_committee_eval_processor_is_a_processor_spec(tmp_path):
    from jaxrens_al.run.processors import (
        CommitteeEvalArgs,
        CommitteeEvalProcessor,
    )

    bundles = []
    for i in range(2):  # each member is a bundle *directory* -> one .tar
        b = tmp_path / f"member_{i}.bundle"
        b.mkdir()
        (b / "config.json").write_text("{}")
        bundles.append(b)

    proc = CommitteeEvalProcessor(bundles, backend="mace")
    assert isinstance(proc, Processor)
    assert proc.name == "committee_eval"
    shared = proc.shared_files()
    assert len(shared) == 2 and all(s.endswith(".tar") for s in shared)

    args = proc.build_args("/remote/savepath")
    assert args["model_remotes"] == [
        "/remote/savepath/member_0.bundle.tar",
        "/remote/savepath/member_1.bundle.tar",
    ]
    assert args["is_archive"] is True and args["backend"] == "mace"
    # Defaults to the shared uncertainty keys UncertaintySelector reads.
    assert args["energy_key"] == "energy_uncertainty"
    assert args["forces_key"] == "forces_uncertainty"
    CommitteeEvalArgs.from_dict(args).validate()


def test_committee_eval_processor_rejects_single_model(tmp_path):
    from jaxrens_al.run.processors import CommitteeEvalProcessor

    only = tmp_path / "only.bundle"
    only.mkdir()
    with pytest.raises(ValueError, match="at least two"):
        CommitteeEvalProcessor([only])


def test_committee_eval_args_reject_short_committee():
    from jaxrens_al.run.processors import CommitteeEvalArgs

    with pytest.raises(ScriptInputError, match="at least two"):
        CommitteeEvalArgs(model_remotes=["only"]).validate()


def test_committee_eval_processor_single_file_ensemble(tmp_path):
    from jaxrens_al.run.processors import (
        CommitteeEvalArgs,
        CommitteeEvalProcessor,
    )

    pkl = tmp_path / "ensemble.pkl"
    pkl.write_bytes(b"\x00")
    proc = CommitteeEvalProcessor(
        [pkl],
        backend="neuralil",
        single_file_ensemble=True,
        supercell_trafo=(2, 2, 2),
    )
    assert proc.shared_files() == [str(pkl)]
    args = proc.build_args("/remote")
    assert args["model_remotes"] == ["/remote/ensemble.pkl"]
    assert args["single_file_ensemble"] is True
    assert args["supercell_trafo"] == [2, 2, 2]
    CommitteeEvalArgs.from_dict(args).validate()

    with pytest.raises(ValueError, match="exactly one"):
        CommitteeEvalProcessor([pkl, pkl], single_file_ensemble=True)


@pytest.mark.parametrize(
    "kw, match",
    [
        (
            {"model_remotes": ["a", "b"], "single_file_ensemble": True},
            "exactly one",
        ),
        (
            {"model_remotes": ["a", "b"], "supercell_trafo": [2, 2]},
            "three positive ints",
        ),
    ],
)
def test_committee_eval_args_single_file_validation(kw, match):
    from jaxrens_al.run.processors import CommitteeEvalArgs

    with pytest.raises(ScriptInputError, match=match):
        CommitteeEvalArgs(**kw).validate()


def test_run_committee_eval_single_file_ensemble(monkeypatch):
    """One ensemble checkpoint -> std/mean over its members, per config."""
    import numpy as np
    from ase import Atoms

    from jaxrens_al.dashboard.panels import model_eval as me
    from jaxrens_al.run.processors import committee_eval

    atoms = [
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=[5] * 3, pbc=True)
        for _ in range(2)
    ]
    seen = {}

    def _members(path, backend, al, **kw):
        seen.update(path=path, backend=backend, **kw)
        # Per config: (M=2,) energies and (M, N=2, 3) forces.
        energies = [np.array([1.0, 2.0]), np.array([3.0, 5.0])]
        forces = [np.stack([np.zeros((2, 3)), np.ones((2, 3))])] * 2
        return energies, forces

    monkeypatch.setattr(me, "score_members", _members)
    monkeypatch.setattr(
        me,
        "score_atoms",
        lambda *a, **k: pytest.fail("single-file path must not score files"),
    )
    out = committee_eval.run(
        atoms,
        {
            "backend": "neuralil",
            "model_remotes": ["ens.pkl"],
            "is_archive": False,
            "energy_key": "energy_uncertainty",
            "forces_key": "forces_uncertainty",
            "single_file_ensemble": True,
            "supercell_trafo": [2, 2, 2],
        },
    )
    assert seen["path"] == "ens.pkl" and seen["backend"] == "neuralil"
    assert seen["supercell_trafo"] == [2, 2, 2]
    assert out[0].info["energy_uncertainty"] == pytest.approx(0.5)
    assert out[1].info["energy_uncertainty"] == pytest.approx(1.0)
    np.testing.assert_allclose(out[0].arrays["forces_uncertainty"], 0.5)
    np.testing.assert_allclose(out[0].arrays["committee_forces"], 0.5)


def test_score_members_keeps_the_ensemble_axis(monkeypatch):
    """score_members returns (M,) energies / (M, N, 3) forces per config."""
    pytest.importorskip("jaxrens")
    from typing import NamedTuple

    import jax.numpy as jnp
    import jaxrens.backends.loader as loader
    import numpy as np
    from ase import Atoms

    from jaxrens_al.dashboard.panels.model_eval import score_members

    class _Res(NamedTuple):
        energy: jnp.ndarray
        forces: jnp.ndarray
        max_neighbor_count: jnp.ndarray
        overflow: jnp.ndarray
        energy_members: jnp.ndarray
        forces_members: jnp.ndarray

    loaded = {}

    class _Ensemble:
        atomic_numbers = [14]
        symbol_map = {"Si": 0}

        def members(self, p, t, c, max_neighbors):
            e = jnp.stack([p.sum(), 2.0 * p.sum()])  # M = 2
            f = jnp.stack([jnp.zeros_like(p), jnp.ones_like(p)])
            return _Res(
                e.mean(), f.mean(0), jnp.asarray(3), jnp.asarray(False), e, f
            )

    def _load(kind, **kw):
        loaded.update(kind=kind, **kw)
        return _Ensemble()

    monkeypatch.setattr(loader, "load_backend", _load)
    frames = [
        Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[6] * 3, pbc=True)
    ]
    energies, forces = score_members(
        "ens.pkl", "neuralil", frames, supercell_trafo=[2, 2, 2]
    )
    assert loaded["kind"] == "neuralil"
    assert loaded["supercell_trafo"] == (2, 2, 2)
    assert energies[0].shape == (2,) and forces[0].shape == (2, 2, 3)
    np.testing.assert_allclose(energies[0], [3.0, 6.0])


def test_driver_registered_committee_eval_with_schema():
    assert "committee_eval" in REGISTRY
    validate_args("committee_eval", {"model_remotes": ["a", "b"]})


def test_run_committee_eval_writes_disagreement(monkeypatch):
    """``run`` annotates per-config energy std + per-atom force std."""
    import numpy as np
    from ase import Atoms

    from jaxrens_al.dashboard.panels import model_eval as me
    from jaxrens_al.run.processors import committee_eval

    atoms = [
        Atoms(
            "H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=[5] * 3, pbc=True
        ),
        Atoms(
            "H2", positions=[[0, 0, 0], [0, 0, 0.9]], cell=[5] * 3, pbc=True
        ),
    ]
    # Two members with distinct predictions -> nonzero disagreement. Each
    # score_atoms call returns (energies, per-config force arrays).
    outputs = {
        "m0": ([1.0, 3.0], [np.zeros((2, 3)), np.zeros((2, 3))]),
        "m1": ([2.0, 5.0], [np.ones((2, 3)), np.ones((2, 3))]),
    }
    monkeypatch.setattr(
        me,
        "score_atoms",
        lambda path, backend, al, **kw: outputs[path],
    )

    out = committee_eval.run(
        atoms,
        {
            "backend": "mace",
            "model_remotes": ["m0", "m1"],
            "is_archive": False,
            "energy_key": "energy_uncertainty",
            "forces_key": "forces_uncertainty",
            "mean_energy_key": "committee_energy",
            "mean_forces_key": "committee_forces",
            "max_neighbors": 10,
        },
    )
    # population std of [1, 2] = 0.5; of [3, 5] = 1.0; mean of [1, 2] = 1.5.
    assert out[0].info["energy_uncertainty"] == pytest.approx(0.5)
    assert out[1].info["energy_uncertainty"] == pytest.approx(1.0)
    assert out[0].info["committee_energy"] == pytest.approx(1.5)
    # per-atom force std between the all-0 and all-1 members is 0.5 everywhere.
    np.testing.assert_allclose(out[0].arrays["forces_uncertainty"], 0.5)
    # ... and their mean is 0.5 too -- the denominator of |sigma_F| / |F|.
    np.testing.assert_allclose(out[0].arrays["committee_forces"], 0.5)


def test_run_committee_eval_defaults_the_mean_forces_key(monkeypatch):
    """``mean_forces_key`` is optional: omitting it still writes the mean."""
    import numpy as np
    from ase import Atoms

    from jaxrens_al.dashboard.panels import model_eval as me
    from jaxrens_al.keys import COMMITTEE_FORCES_KEY
    from jaxrens_al.run.processors import committee_eval

    atoms = [
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=[5] * 3, pbc=True)
    ]
    outputs = {
        "m0": ([1.0], [np.zeros((2, 3))]),
        "m1": ([2.0], [np.full((2, 3), 4.0)]),
    }
    monkeypatch.setattr(
        me,
        "score_atoms",
        lambda path, backend, al, **kw: outputs[path],
    )

    out = committee_eval.run(
        atoms,
        {
            "backend": "mace",
            "model_remotes": ["m0", "m1"],
            "is_archive": False,
            "energy_key": "energy_uncertainty",
            "forces_key": "forces_uncertainty",
        },
    )
    np.testing.assert_allclose(out[0].arrays[COMMITTEE_FORCES_KEY], 2.0)


def test_committee_eval_processor_passes_the_mean_forces_key(tmp_path):
    """The host-side spec forwards ``mean_forces_key`` into the remote args."""
    from jaxrens_al.run.processors.committee_eval import CommitteeEvalProcessor

    models = []
    for i in range(2):
        m = tmp_path / f"model_{i}.pkl"
        m.write_text("x")
        models.append(m)
    proc = CommitteeEvalProcessor(models, mean_forces_key="ensemble_forces")
    assert proc.build_args("/remote")["mean_forces_key"] == "ensemble_forces"


# ---------------------------------------------------------------------------
# relax + spacegroup processors and the phase-labeling resolver.
# ---------------------------------------------------------------------------
def test_spacegroup_run_labels_structures():
    """The cluster-side ``run`` assigns a space group via spglib (self-skips)."""
    pytest.importorskip("spglib")
    from ase.build import bulk

    from jaxrens_al.run.processors import spacegroup

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)  # -> Fm-3m (225)
    out = spacegroup.run([atoms], {"symprec": 1e-3})
    assert out[0].info["spacegroup"] == 225
    assert out[0].info["spacegroup_symbol"].startswith("Fm-3m")


def test_spacegroup_processor_spec():
    from jaxrens_al.run.processors import SpaceGroupArgs, SpaceGroupProcessor

    proc = SpaceGroupProcessor(symprec=1e-2)
    assert isinstance(proc, Processor) and proc.name == "spacegroup"
    assert proc.shared_files() == []
    args = proc.build_args("/remote")
    assert args["symprec"] == 1e-2
    SpaceGroupArgs.from_dict(args).validate()


def test_relax_processor_stages_bundle_dir(tmp_path):
    from jaxrens_al.run.processors import RelaxArgs, RelaxProcessor

    bundle = tmp_path / "trainer_it0.bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    proc = RelaxProcessor(
        bundle, backend="mace", method="fire", fmax=0.02, steps=50
    )
    assert proc.name == "relax"
    shared = proc.shared_files()
    assert len(shared) == 1 and shared[0].endswith(".tar")
    args = proc.build_args("/remote/savepath")
    assert args["model_remote"] == "/remote/savepath/trainer_it0.bundle.tar"
    assert args["is_archive"] is True and args["fmax"] == 0.02
    assert args["method"] == "fire"
    RelaxArgs.from_dict(args).validate()


def test_relax_args_rejects_unknown_method():
    from remote_handler.exceptions import ScriptInputError

    from jaxrens_al.run.processors import RelaxArgs

    with pytest.raises(ScriptInputError, match="method"):
        RelaxArgs(model_remote="m", method="nope").validate()


def test_relax_args_rejects_empty_bucket_ladder():
    from remote_handler.exceptions import ScriptInputError

    from jaxrens_al.run.processors import RelaxArgs

    with pytest.raises(ScriptInputError, match="max_neighbors_list"):
        RelaxArgs(model_remote="m", max_neighbors_list=()).validate()


def test_run_subbatches_by_batch_size(monkeypatch):
    """A same-size shard is relaxed in sub-batches of at most ``batch_size``."""
    pytest.importorskip("jax")
    pytest.importorskip("jaxrens.io.formats")
    from types import SimpleNamespace

    import ase
    import jaxrens.io.formats as fmts
    import numpy as np

    from jaxrens_al.run.processors import relax as relax_mod

    atoms = [
        ase.Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.7]],
            cell=np.eye(3) * 5,
            pbc=True,
        )
        for _ in range(5)  # 5 frames, batch_size 2 -> chunks [2, 2, 1]
    ]

    monkeypatch.setattr(relax_mod, "_materialise_model", lambda a, c: "model")
    monkeypatch.setattr(
        relax_mod, "_load_backend", lambda p, b: (object(), {"H": 0})
    )
    monkeypatch.setattr(relax_mod, "_probe_peak", lambda *a, **k: None)
    monkeypatch.setattr(
        fmts,
        "ase_atoms_to_walker",
        lambda at, sm: SimpleNamespace(
            positions=np.asarray(at.get_positions()),
            types=np.zeros(len(at), int),
            cell=np.asarray(at.get_cell()[:]),
        ),
    )

    sizes = []

    def _fake_make(backend, fmax, steps, method):
        def _f(pos, types, cell, buffer):
            sizes.append(int(pos.shape[0]))
            n = pos.shape[0]
            # Shift positions so relaxed != sampled, to check both are kept.
            return np.asarray(pos) + 1.0, np.ones(n, bool), np.zeros(n)

        return _f

    monkeypatch.setattr(relax_mod, "_make_relax_batch", _fake_make)

    out = relax_mod.run(
        atoms,
        {
            "fmax": 0.05,
            "steps": 10,
            "method": "lbfgs",
            "batch_size": 2,
            "max_neighbors": 50,
            "max_neighbors_list": (30, 35),
            "max_neighbors_offset": 5,
            "backend": "mace",
            "model_remote": "m",
            "is_archive": False,
        },
        calcpath=".",
    )
    from jaxrens_al.keys import INITIAL_POSITIONS_KEY, RELAX_ENERGY_KEY

    assert sizes == [2, 2, 1]
    assert all(
        "is_converged" in a.info and RELAX_ENERGY_KEY in a.info for a in out
    )
    # Active positions are the relaxed basin; the sampled geometry is retained
    # in the initial_positions array (one frame carries both).
    for a in out:
        assert INITIAL_POSITIONS_KEY in a.arrays
        np.testing.assert_allclose(
            a.get_positions(), a.arrays[INITIAL_POSITIONS_KEY] + 1.0
        )
        np.testing.assert_allclose(
            a.arrays[INITIAL_POSITIONS_KEY], [[0, 0, 0], [0, 0, 0.7]]
        )


def test_run_isolates_a_failing_structure(monkeypatch):
    """A single crashing relaxation is marked failed; the shard still returns.

    The vmapped sub-batch aborts as a whole, so the run retries it structure by
    structure -- the good ones relax, only the culprit is emitted as failed.
    """
    pytest.importorskip("jax")
    pytest.importorskip("jaxrens.io.formats")
    import math
    from types import SimpleNamespace

    import ase
    import jaxrens.io.formats as fmts
    import numpy as np

    from jaxrens_al.keys import (
        RELAX_ENERGY_KEY,
        RELAX_ERROR_KEY,
        RELAX_FAILED_KEY,
    )
    from jaxrens_al.run.processors import relax as relax_mod

    # 5 two-atom structures; index 2 is "bad" (cell marker 999) and crashes.
    atoms = []
    for k in range(5):
        a = ase.Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.7]],
            cell=np.eye(3) * 5,
            pbc=True,
        )
        if k == 2:
            c = a.get_cell()[:]
            c[0, 0] = 999.0
            a.set_cell(c)
        atoms.append(a)

    monkeypatch.setattr(relax_mod, "_materialise_model", lambda a, c: "model")
    monkeypatch.setattr(
        relax_mod, "_load_backend", lambda p, b: (object(), {"H": 0})
    )
    monkeypatch.setattr(relax_mod, "_probe_peak", lambda *a, **k: None)
    monkeypatch.setattr(
        fmts,
        "ase_atoms_to_walker",
        lambda at, sm: SimpleNamespace(
            positions=np.asarray(at.get_positions()),
            types=np.zeros(len(at), int),
            cell=np.asarray(at.get_cell()[:]),
        ),
    )

    def _fake_make(backend, fmax, steps, method):
        def _f(pos, types, cell, buffer):
            # Any structure carrying the 999 cell marker blows up the sub-batch.
            if bool((np.asarray(cell)[:, 0, 0] == 999.0).any()):
                raise RuntimeError("boom: neighbour bucket overflow")
            n = pos.shape[0]
            return np.asarray(pos) + 1.0, np.ones(n, bool), np.zeros(n)

        return _f

    monkeypatch.setattr(relax_mod, "_make_relax_batch", _fake_make)

    out = relax_mod.run(
        atoms,
        {
            "fmax": 0.05,
            "steps": 10,
            "method": "lbfgs",
            "batch_size": 32,  # all 5 in one sub-batch -> it aborts, then retry
            "max_neighbors": 50,
            "backend": "mace",
            "model_remote": "m",
            "is_archive": False,
        },
        calcpath=".",
    )

    assert len(out) == 5
    # The 4 good structures relaxed; none flagged failed.
    for k in (0, 1, 3, 4):
        assert out[k].info["is_converged"] is True
        assert RELAX_FAILED_KEY not in out[k].info
        np.testing.assert_allclose(
            out[k].get_positions(), atoms[k].get_positions() + 1.0
        )
    # The culprit is emitted failed: sampled geometry kept, reason recorded.
    bad = out[2]
    assert bad.info["is_converged"] is False
    assert bad.info[RELAX_FAILED_KEY] is True
    assert "boom" in bad.info[RELAX_ERROR_KEY]
    assert math.isnan(bad.info[RELAX_ENERGY_KEY])
    np.testing.assert_allclose(bad.get_positions(), atoms[2].get_positions())


def test_pick_next_bucket_grows_by_ladder_with_offset():
    """The dynamic buffer picks the smallest rung covering peak+offset."""
    from jaxrens_al.run.processors.relax import _pick_next_bucket

    ladder = (30, 35, 40, 45, 50)
    # peak 28 + offset 5 = 33 -> smallest rung >= 33 is 35.
    assert _pick_next_bucket(28, 0, ladder, 5) == 35
    # must also strictly exceed the current bucket (growth on overflow).
    assert _pick_next_bucket(28, 35, ladder, 5) == 40
    # ladder exhausted -> user-actionable error.
    with pytest.raises(RuntimeError, match="max_neighbors_list"):
        _pick_next_bucket(60, 0, ladder, 5)


def test_probe_peak_uses_backend_probe_and_falls_back():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from jaxrens_al.run.processors.relax import _probe_peak

    pos = jnp.zeros((2, 4, 3))
    cell = jnp.zeros((2, 3, 3))

    class _Probing:
        def max_neighbors_for(self, positions, cell):
            # per-atom counts; batch max here is 12.
            return jnp.array([12, 3, 7, 1])

    class _NoProbe:
        pass

    assert _probe_peak(_Probing(), pos, cell) == 12
    assert _probe_peak(_NoProbe(), pos, cell) is None


@pytest.mark.parametrize("method", ["fire", "lbfgs"])
def test_make_relax_batch_minimises_a_vmapped_bucket(method):
    """The jax core relaxes a whole batch in one compiled kernel.

    Uses a toy quadratic 'backend' (no jaxrens / GPU needed) with a distinct
    minimum per structure, so a correct batched relaxation must move each frame
    to its own target and flag convergence.
    """
    jax = pytest.importorskip("jax")
    pytest.importorskip("jax_md")
    pytest.importorskip("optax")
    import jax.numpy as jnp

    from jaxrens_al.run.processors.relax import _make_relax_batch

    targets = jnp.array(  # one per structure in the batch, 2 atoms x 3D
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )

    class _ToyBackend:
        """energy = sum (pos - target)^2; forces via autodiff. Per-call target
        is picked from ``cell[0,0]`` (an index we smuggle through)."""

        def __call__(self, pos, types, cell, max_neighbors, **kw):
            i = cell[0, 0].astype(int)
            e = jnp.sum((pos - targets[i]) ** 2)
            return _Res(energy=e)

    import collections

    global _Res
    _Res = collections.namedtuple("_Res", ["energy"])

    relax = _make_relax_batch(
        _ToyBackend(), fmax=1e-3, steps=200, method=method
    )
    pos = jnp.zeros((2, 2, 3))
    types = jnp.zeros((2, 2), int)
    # smuggle the per-structure target index in via cell[:,0,0] = {0,1}
    cell = jnp.zeros((2, 3, 3)).at[:, 0, 0].set(jnp.array([0.0, 1.0]))
    relaxed, converged, energy = relax(pos, types, cell, 8)

    import numpy as np

    assert np.all(np.asarray(converged))
    np.testing.assert_allclose(
        np.asarray(relaxed), np.asarray(targets), atol=1e-2
    )
    # Energy at each minimum is ~0 (quadratic well bottom).
    np.testing.assert_allclose(np.asarray(energy), 0.0, atol=1e-3)


class _FakePhaseHandler:
    """Fake RemoteTrajectoryProcessor for the phase pipeline: writes a
    spacegroup onto each frame where redistribute would (front_tag). Same
    allocate_data/submit/executor.wait_until_finished/collect_and_harvest
    split as :class:`_FakeModelEvalHandler`."""

    calls = []

    def __init__(self, *, traj_files, processors, tag, **kw):
        self.traj_files = traj_files
        self.tag = tag
        self.steps = [p.name for p in processors]
        self.executor = _FakeExecutor()
        _FakePhaseHandler.calls.append(
            {"tag": tag, "n_traj": len(traj_files), "steps": self.steps}
        )

    def allocate_data(self):
        pass

    def submit(self, only_start=False):
        pass

    def collect_and_harvest(self):
        import ase.io
        from remote_handler.core.atoms_shard import front_tag

        for src in self.traj_files:
            frames = ase.io.read(src, index=":")
            for at in frames:
                at.info["is_converged"] = True
                at.info["spacegroup"] = 225
                at.info["spacegroup_symbol"] = "Fm-3m"
            ase.io.write(str(front_tag(src, self.tag)), frames)


def test_resolve_phase_labeling_one_job_per_model(monkeypatch, tmp_path):
    """Relax+spacegroup is one [relax, spacegroup] job per model; delivered
    per-model so shared inputs don't clobber."""
    from jaxrens_al.dashboard.panels.phase import PhaseNeed
    from jaxrens_al.run.processing_resolver import resolve_phase_labeling

    traj = _refs(tmp_path, "ns.run00.traj.extxyz")
    _FakePhaseHandler.calls.clear()
    monkeypatch.setattr(
        "remote_handler.processing.RemoteTrajectoryProcessor",
        _FakePhaseHandler,
    )

    class _Panel:
        def __init__(self, model):
            self._model = model
            self._need = PhaseNeed.make([traj], model, "mace")
            self.received = None

        def phase_needs(self):
            return [self._need]

        def receive_phase_labels(self, mapping):
            self.received = mapping.get(self._model, {}).get(str(traj))

    ctx = dict(cluster_config=None, remote_fs=None, remote_savepath="/remote")
    p = _Panel("bundleA")
    resolve_phase_labeling([p], **ctx)
    assert len(_FakePhaseHandler.calls) == 1
    assert _FakePhaseHandler.calls[0]["steps"] == ["relax", "spacegroup"]
    assert p.received is not None
    # The delivered file carries the space-group label.
    import ase.io

    assert ase.io.read(str(p.received), index=0).info["spacegroup"] == 225

    # Reuse: a re-run with the same model submits no new job.
    p2 = _Panel("bundleA")
    resolve_phase_labeling([p2], **ctx)
    assert len(_FakePhaseHandler.calls) == 1


def test_committee_eval_forwards_the_batching_knobs(tmp_path):
    """batch_size / ladder reach the remote args and are validated."""
    from jaxrens_al.run.processors.committee_eval import (
        CommitteeEvalArgs,
        CommitteeEvalProcessor,
    )

    models = []
    for i in range(2):
        m = tmp_path / f"m{i}.pkl"
        m.write_text("x")
        models.append(m)
    proc = CommitteeEvalProcessor(
        models, batch_size=64, max_neighbors_list=[30, 40, 50]
    )
    args = proc.build_args("/remote")
    assert args["batch_size"] == 64
    assert args["max_neighbors_list"] == [30, 40, 50]

    with pytest.raises(ScriptInputError, match="batch_size"):
        CommitteeEvalArgs(model_remotes=["a", "b"], batch_size=0).validate()
    with pytest.raises(ScriptInputError, match="ascending"):
        CommitteeEvalArgs(
            model_remotes=["a", "b"], max_neighbors_list=[50, 30]
        ).validate()
