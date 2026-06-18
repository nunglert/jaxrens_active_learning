"""End-to-end AL-loop integration tests.

A cheap mirror of ``experiments/run1/run_al.py``: same components
(``JaxrensNSSampler`` + ``VASPEvaluator`` + ``NeuralilILTrainer`` /
``MaceFinetuningTrainer``, driven by ``ActiveLearningHandler``), but each
actor's ``stub=True`` path is taken so nothing leaves the local filesystem.
The goal is to catch wiring/API drift across alviz / remote_handler /
jaxrens that pure import-checks miss -- not to test the science.

Skipped without the ``[run]`` stack (remote_handler + jaxrens).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("remote_handler")
pytest.importorskip("jaxrens")

import ase.io  # noqa: E402
from remote_handler.core.cluster_handler import ClusterConfig  # noqa: E402
from remote_handler.core.file_handler import LocalFileSystem  # noqa: E402
from remote_handler.workflows import (  # noqa: E402
    active_learning_base,
    evaluators,
    trainers,
)

from jaxrens_al import template_dir  # noqa: E402
from jaxrens_al.sampler import JaxrensNSSampler  # noqa: E402

# Marked integration: needs the [run] stack. Dashboard-side files use `local`.
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared fixtures: build the minimal cluster wiring the stub paths accept.
# In stub mode the actors guard `if not self.stub:` before constructing any
# SingleScriptTask/VASPHandler, so these placeholder values never reach SSH
# or SLURM -- they just need to construct.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_cluster(tmp_path, monkeypatch):
    """LocalFileSystem + a dict-built ClusterConfig with placeholder fields."""
    # Even though stub mode bypasses template rendering, set this so any
    # non-stub validation that might run during construction has a real dir.
    monkeypatch.setenv("REMOTE_HANDLER_SLURM_TEMPLATES", str(template_dir()))
    fs = LocalFileSystem()
    cluster = ClusterConfig.from_dict(
        {
            "remote_machine": "localhost",
            "partition": "test",
            "conda_env": "test",
            "conda_source": "/opt/conda",
            "savepath": str(tmp_path / "remote"),
            "template": "body_jaxrens.sh.j2",
        }
    )
    return fs, cluster


@pytest.fixture
def ns_yaml(tmp_path):
    """A minimal-but-valid jaxrens NS config -- the sampler will patch it."""
    cfg = tmp_path / "ns.yaml"
    cfg.write_text(
        "run:\n"
        "  n_live: 10\n"
        "backend:\n"
        "  type: neuralil\n"
        "  checkpoint_path: PLACEHOLDER\n"
        "init:\n"
        "  start_species: '29 16, 8 4'\n"
        "output:\n"
        "  format: extxyz\n"
        "  out_file_prefix: ns\n"
    )
    return cfg


@pytest.fixture
def fake_neuralil_model(tmp_path):
    """A stub trained-model file shaped like the trainer's output bundle.

    ``ActiveLearningHandler.move_file_to_home`` shutil-copies it into the run
    dir during __init__, so the path must exist; the stub sampler never reads
    its contents, only its path (it lands in ``backend.checkpoint_path``).
    """
    path = tmp_path / "model_init.pkl"
    path.write_bytes(b"stub")
    return path


@pytest.fixture
def fake_mace_jax_bundle(tmp_path):
    """A stub mace-jax bundle DIRECTORY (config.json + params.msgpack).

    The MACE-driven phase ships a directory, not a single file, through the
    handler (``move_file_to_home`` recursively copies). Mirror the real
    foundation-bundle shape so the stub path exercises ``shutil.copytree``.
    """
    bundle = tmp_path / "foundation_mace_jax"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    (bundle / "params.msgpack").write_bytes(b"")
    return bundle


@pytest.fixture
def fake_dft_script(tmp_path):
    """A placeholder evaluator script path -- stub VASPEvaluator never runs it."""
    path = tmp_path / "dft_singlepoint.py"
    path.write_text("# placeholder; never executed in stub mode\n")
    return path


@pytest.fixture
def fake_train_script(tmp_path):
    """A placeholder trainer script path -- stub trainers never run it."""
    path = tmp_path / "training.py"
    path.write_text("# placeholder; never executed in stub mode\n")
    return path


# ---------------------------------------------------------------------------
# Test 1: minimal single-phase loop. Two iterations, NeuralIL trainer.
# ---------------------------------------------------------------------------
def test_single_phase_loop_runs_end_to_end(
    fake_cluster,
    ns_yaml,
    fake_neuralil_model,
    fake_dft_script,
    fake_train_script,
    tmp_path,
):
    """One AL loop, 2 iterations, every actor in stub mode.

    Exercises the full Sampler -> Evaluator -> Trainer cycle through
    ``ActiveLearningHandler``, producing the calchome layout that the
    dashboard reads. The assertions match what ``jaxrens_al.run_layout``
    expects (so a green test here means the dashboard CLI will also work --
    see ``test_dashboard_renders_pipeline_output``).
    """
    fs, cluster = fake_cluster
    calchome = tmp_path / "calchome"

    sampler = JaxrensNSSampler(
        remote_fs=fs,
        cluster_config=cluster,
        config_file=ns_yaml,
        sample_size=10,
        n_gpu_parallel=1,
        backend_type="neuralil",
        out_file_prefix="ns",
        remote_savepath=cluster.savepath,
        stub=True,
    )
    evaluator = evaluators.VASPEvaluator(
        singlepoint_script=fake_dft_script,
        remote_fs=fs,
        cluster_config=cluster,
        num_parallel=1,
        incar_kwargs={"xc": "PBE"},
        remote_savepath=cluster.savepath,
        stub=True,
    )
    trainer = trainers.NeuralilILTrainer(
        training_script=fake_train_script,
        remote_fs=fs,
        cluster_config=cluster,
        training_kwargs={"n_ensemble": 2, "n_epochs": 1},
        remote_savepath=cluster.savepath,
        stub=True,
    )

    handler = active_learning_base.ActiveLearningHandler(
        sampler=sampler,
        evaluator=evaluator,
        trainer=trainer,
        calchome=str(calchome),
        cycle_entry_point="sampler",
        initial_trainer_out=fake_neuralil_model,
        num_iterations=2,
    )
    handler.run()

    # AL_progress.json: the source of truth jaxrens_al.run_layout reads from.
    progress_file = calchome / "AL_progress.json"
    assert progress_file.exists()
    progress = json.loads(progress_file.read_text())
    assert len(progress) >= 2
    for entry in progress[:2]:
        assert entry.get(
            "evaluator_output"
        ), "evaluator should have produced configs"
        assert entry.get(
            "sampler_output"
        ), "sampler should have produced configs"

    # Numeric per-iteration dirs (calchome/0, calchome/1) populated.
    for i in range(2):
        it_dir = calchome / str(i)
        assert it_dir.is_dir(), f"missing iteration dir {it_dir}"
        assert (it_dir / "ns_sampler" / "results").exists()
        assert (it_dir / "training" / "results").exists()

    # Sampled configs carry the jaxrens-info keys the panels read.
    rec = progress[0]["sampler_output"]
    rel = rec[0] if isinstance(rec, list) else rec
    frames = ase.io.read(calchome / rel, index=":")
    assert len(frames) == sampler.sample_size
    for at in frames:
        assert "ns_energy" in at.info
        assert "run_id" in at.info

    # Evaluated configs carry the REF_* labels the dashboard expects.
    rec = progress[0]["evaluator_output"]
    rel = rec[0] if isinstance(rec, list) else rec
    frames = ase.io.read(calchome / rel, index=":")
    assert all("REF_energy" in at.info for at in frames)
    assert all("REF_forces" in at.arrays for at in frames)


# ---------------------------------------------------------------------------
# Test 2: two-phase loop. Phase 1 (MACE finetuning, QUESTS diversity) ->
# phase 2 (NeuralIL ensemble, entered at training, seeded from phase 1).
# Mirrors the experiments/run1/run_al.py shape at toy scale.
# ---------------------------------------------------------------------------
def test_two_phase_loop_seeds_phase2_from_phase1(
    fake_cluster,
    ns_yaml,
    fake_mace_jax_bundle,
    fake_dft_script,
    fake_train_script,
    tmp_path,
):
    """Phase 1 produces evaluated configs; phase 2 enters at training with them.

    The two-phase shape is the part the simple loop can't catch: directory
    bundles flowing through ``move_file_to_home``, ``collect_evaluated_outputs``
    crossing phase boundaries, ``cycle_entry_point="training"`` with
    ``initial_evaluator_out``.
    """
    from remote_handler.workflows.selectors import QUESTSSelector

    fs, cluster = fake_cluster
    calchome = tmp_path / "calchome"

    # ---- phase 1: MACE foundation bootstrap, 1 iter --------------------
    p1_sampler = JaxrensNSSampler(
        remote_fs=fs,
        cluster_config=cluster,
        config_file=ns_yaml,
        sample_size=8,
        selector=QUESTSSelector(n_configs=8, method="msc"),
        n_gpu_parallel=1,
        backend_type="mace",
        out_file_prefix="mace_test",
        remote_savepath=cluster.savepath,
        stub=True,
    )
    p1_evaluator = evaluators.VASPEvaluator(
        singlepoint_script=fake_dft_script,
        remote_fs=fs,
        cluster_config=cluster,
        num_parallel=1,
        incar_kwargs={"xc": "PBE"},
        remote_savepath=cluster.savepath,
        stub=True,
    )
    # MaceFinetuningTrainer: foundation_model passed but never read in stub.
    p1_trainer = trainers.MaceFinetuningTrainer(
        remote_fs=fs,
        cluster_config=cluster,
        training_script=fake_train_script,
        config_file=fake_train_script,  # placeholder MACE training config
        foundation_model=fake_mace_jax_bundle,
        remote_mace_train_runner=fake_train_script,
        num_models=1,
        convert_to_jax=True,
        remote_savepath=cluster.savepath,
        stub=True,
    )
    phase1 = calchome / "phase1_mace"
    active_learning_base.ActiveLearningHandler(
        sampler=p1_sampler,
        evaluator=p1_evaluator,
        trainer=p1_trainer,
        calchome=str(phase1),
        cycle_entry_point="sampler",
        initial_trainer_out=[fake_mace_jax_bundle],  # directory bundle
        combine_evaluator_outputs=True,
        num_iterations=1,
    ).run()

    # Collect phase-1's evaluated configs the way the example's
    # ``collect_evaluated_outputs`` does (the bridge across phases).
    p1_progress = json.loads((phase1 / "AL_progress.json").read_text())
    seed_evaluated = []
    for entry in p1_progress:
        out = entry.get("evaluator_output") or []
        if isinstance(out, str):
            out = [out]
        seed_evaluated.extend(phase1 / rel for rel in out)
    assert (
        seed_evaluated
    ), "phase 1 produced no evaluated configs to seed phase 2"

    # ---- phase 2: NeuralIL ensemble, entered at training ---------------
    p2_sampler = JaxrensNSSampler(
        remote_fs=fs,
        cluster_config=cluster,
        config_file=ns_yaml,
        sample_size=8,
        n_gpu_parallel=1,
        backend_type="neuralil",
        out_file_prefix="ns",
        remote_savepath=cluster.savepath,
        stub=True,
    )
    p2_evaluator = evaluators.VASPEvaluator(
        singlepoint_script=fake_dft_script,
        remote_fs=fs,
        cluster_config=cluster,
        num_parallel=1,
        incar_kwargs={"xc": "PBE"},
        remote_savepath=cluster.savepath,
        stub=True,
    )
    p2_trainer = trainers.NeuralilILTrainer(
        training_script=fake_train_script,
        remote_fs=fs,
        cluster_config=cluster,
        training_kwargs={"n_ensemble": 2, "n_epochs": 1},
        remote_savepath=cluster.savepath,
        stub=True,
    )
    phase2 = calchome / "phase2_neuralil"
    active_learning_base.ActiveLearningHandler(
        sampler=p2_sampler,
        evaluator=p2_evaluator,
        trainer=p2_trainer,
        calchome=str(phase2),
        cycle_entry_point="training",
        initial_evaluator_out=seed_evaluated,
        combine_evaluator_outputs=True,
        num_iterations=1,
    ).run()

    # Both phases left behind a complete calchome the dashboard can read.
    for phase in (phase1, phase2):
        assert (phase / "AL_progress.json").exists()
        assert (phase / "0").is_dir()


# ---------------------------------------------------------------------------
# Test 3: the dashboard CLI builds a PDF off the stubbed-loop output. This
# closes the loop -- producer (run_al) -> consumer (dashboard) -- end to end.
# ---------------------------------------------------------------------------
def test_dashboard_renders_pipeline_output(
    fake_cluster,
    ns_yaml,
    fake_neuralil_model,
    fake_dft_script,
    fake_train_script,
    tmp_path,
):
    """Run the loop, then point ``jaxrens-al-dashboard`` at the calchome.

    Exercises the dashboard side against an artifact tree produced by the
    real run-side actors (not the synthetic ``make_calchome`` fixture the
    dashboard suites use), so an output-format change on either side that
    ``run_layout`` couldn't parse fails this test.
    """
    from jaxrens_al.cli import main as dashboard_main

    fs, cluster = fake_cluster
    calchome = tmp_path / "calchome"

    sampler = JaxrensNSSampler(
        remote_fs=fs,
        cluster_config=cluster,
        config_file=ns_yaml,
        sample_size=10,
        n_gpu_parallel=1,
        backend_type="neuralil",
        out_file_prefix="ns",
        remote_savepath=cluster.savepath,
        stub=True,
    )
    evaluator = evaluators.VASPEvaluator(
        singlepoint_script=fake_dft_script,
        remote_fs=fs,
        cluster_config=cluster,
        num_parallel=1,
        incar_kwargs={"xc": "PBE"},
        remote_savepath=cluster.savepath,
        stub=True,
    )
    trainer = trainers.NeuralilILTrainer(
        training_script=fake_train_script,
        remote_fs=fs,
        cluster_config=cluster,
        training_kwargs={"n_ensemble": 2, "n_epochs": 1},
        remote_savepath=cluster.savepath,
        stub=True,
    )
    active_learning_base.ActiveLearningHandler(
        sampler=sampler,
        evaluator=evaluator,
        trainer=trainer,
        calchome=str(calchome),
        cycle_entry_point="sampler",
        initial_trainer_out=fake_neuralil_model,
        num_iterations=2,
    ).run()

    out_pdf = tmp_path / "dashboard.pdf"
    rc = dashboard_main([str(calchome), "-o", str(out_pdf), "--force"])
    assert rc == 0
    assert out_pdf.exists() and out_pdf.stat().st_size > 0
