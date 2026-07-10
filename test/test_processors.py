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

from jaxrens_al.run.processors import (  # noqa: E402
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
        == "jaxrens_al.run.processors.steinhardt"
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


def test_run_annotates_atoms():
    """The cluster-side ``run`` attaches q_l/w_l via jaxrens (self-skips)."""
    pytest.importorskip("jaxrens")
    import numpy as np
    from ase import Atoms

    from jaxrens_al.run.processors import steinhardt

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


def test_model_eval_processor_stages_plain_file(tmp_path):
    from jaxrens_al.run.processors import ModelEvalProcessor

    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(b"\x00")
    proc = ModelEvalProcessor(pkl, backend="neuralil")
    assert proc.shared_files() == [str(pkl)]
    args = proc.build_args("/remote")
    assert args["is_archive"] is False
    assert args["model_remote"] == "/remote/model.pkl"


class _FakeModelEvalHandler:
    """Fake RemoteTrajectoryProcessor: writes each model's keys onto the frames
    where the real handler's redistribute would (front_tag), so the resolver is
    testable without a cluster or a real model."""

    calls = []

    def __init__(self, *, traj_files, processors, tag, **kw):
        self.traj_files = traj_files
        self.tag = tag
        self.proc = processors[0]
        _FakeModelEvalHandler.calls.append(
            {"tag": tag, "n_traj": len(traj_files), "model": self.proc.model}
        )

    def run(self):
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


def test_run_model_eval_annotates_atoms(tmp_path):
    """The cluster-side ``run`` writes model energy via jaxrens (self-skips)."""
    pytest.importorskip("jaxrens")
    import numpy as np
    from ase import Atoms

    from jaxrens_al.run.processors import model_eval

    atoms = Atoms(
        "Cu2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[6, 6, 6], pbc=True
    )

    def _fake_backend(*a, **k):
        class _B:
            atomic_numbers = [29]
            symbol_map = {"Cu": 0}

            def __call__(self, pos, types, cell, mn):
                return (np.float64(-3.5),)

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
            return np.asarray(pos), np.ones(n, bool), np.zeros(n)

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
    assert sizes == [2, 2, 1]
    assert all(
        "is_converged" in a.info and "relax_energy" in a.info for a in out
    )


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
    spacegroup onto each frame where redistribute would (front_tag)."""

    calls = []

    def __init__(self, *, traj_files, processors, tag, **kw):
        self.traj_files = traj_files
        self.tag = tag
        self.steps = [p.name for p in processors]
        _FakePhaseHandler.calls.append(
            {"tag": tag, "n_traj": len(traj_files), "steps": self.steps}
        )

    def run(self):
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
