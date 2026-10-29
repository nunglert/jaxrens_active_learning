"""Tests for the project's native jaxrens Sampler (JaxrensNSSampler).

Needs the [run] extra (remote_handler); the module self-skips otherwise. The
stub-mode and config-assembly tests run with NO cluster, SSH, or jaxrens --
only remote_handler + ase + pyyaml on the host.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("remote_handler")
import ase.io  # noqa: E402
from remote_handler.core.cluster_handler import ClusterConfig  # noqa: E402
from remote_handler.exceptions import TaskError, WorkflowError  # noqa: E402
from remote_handler.workflows.active_learning_base import Sampler  # noqa: E402

from jaxrens_al.run.sampler import JaxrensNSSampler, _set_dotted  # noqa: E402
from jaxrens_al.run.templates import template_dir  # noqa: E402

# Run-side: needs the producer package (remote_handler), unlike the
# dashboard-side suites which are all `local`.
pytestmark = pytest.mark.integration


@pytest.fixture
def mock_remote_fs():
    fs = MagicMock()
    fs.slurm_config = {"savepath": "/remote/path"}
    return fs


@pytest.fixture
def mock_cluster_config():
    return ClusterConfig(
        remote_machine="mock",
        partition="default",
        conda_env="test",
        conda_source="/opt/conda",
        savepath="/tmp/remote_handler",
        template="mock.sh.j2",
    )


def _write_base_config(path):
    """Minimal-but-valid-shaped jaxrens config the sampler will patch."""
    path.write_text(
        "run:\n"
        "  n_live: 200\n"
        "backend:\n"
        "  type: neuralil\n"
        "  checkpoint_path: PLACEHOLDER\n"
        "init:\n"
        "  start_species: '29 16, 8 4'\n"
    )
    return path


def test_jaxrens_sampler_is_a_sampler(
    mock_remote_fs, mock_cluster_config, tmp_path
):
    cfg = _write_base_config(tmp_path / "ns.yaml")

    sampler = JaxrensNSSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        config_file=cfg,
        n_gpu_parallel=4,
        stub=True,
    )

    # Satisfies the orchestrator's Sampler interface...
    assert isinstance(sampler, Sampler)
    # ...with real (not delegated) attributes.
    assert sampler.stub is True
    assert sampler.n_gpu_parallel == 4


def _policy_cluster(resilient):
    return ClusterConfig(
        remote_machine="mock",
        partition="default",
        conda_env="test",
        conda_source="/opt/conda",
        savepath="/tmp/remote_handler",
        template="mock.sh.j2",
        resilient_connection=resilient,
    )


def test_sampler_adopts_cluster_connection_policy(tmp_path):
    """An unpolicied filesystem gets the cluster's policy at construction.

    The chain's sbatch/scancel run on ``remote_fs`` directly, on the reattach
    path before any task (which would adopt it) exists.
    """
    from remote_handler.core import resilience

    fs = MagicMock()
    fs.connection_policy = resilience.DISABLED
    JaxrensNSSampler(
        remote_fs=fs,
        cluster_config=_policy_cluster(True),
        config_file=_write_base_config(tmp_path / "ns.yaml"),
        stub=True,
    )
    assert fs.connection_policy.enabled


def test_sampler_keeps_an_explicit_enabled_policy(tmp_path):
    """Upgrade-only: a caller's enabled policy is never replaced."""
    from remote_handler.core import resilience

    mine = resilience.ConnectionPolicy(enabled=True)
    fs = MagicMock()
    fs.connection_policy = mine
    JaxrensNSSampler(
        remote_fs=fs,
        cluster_config=_policy_cluster(False),
        config_file=_write_base_config(tmp_path / "ns.yaml"),
        stub=True,
    )
    assert fs.connection_policy is mine


def test_unknown_backend_type_rejected(
    mock_remote_fs, mock_cluster_config, tmp_path
):
    cfg = _write_base_config(tmp_path / "ns.yaml")
    with pytest.raises(WorkflowError):
        JaxrensNSSampler(
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            config_file=cfg,
            backend_type="not-a-backend",
            stub=True,
        )


def test_build_config_injects_model_and_output(
    mock_remote_fs, mock_cluster_config, tmp_path
):
    """_build_config overwrites checkpoint + output and applies overrides."""
    cfg = _write_base_config(tmp_path / "ns.yaml")
    sampler = JaxrensNSSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        config_file=cfg,
        backend_type="neuralil",
        out_file_prefix="ns",
        stub=True,
    )

    built = sampler._build_config(
        model_remote="/remote/models/model_3.pkl",
        start_remote="/remote/start.xyz",
        overrides={"run.n_live": 400, "run.n_mcmc_steps": 30},
    )

    assert built["backend"]["checkpoint_path"] == "/remote/models/model_3.pkl"
    assert built["backend"]["type"] == "neuralil"
    assert built["output"]["format"] == "extxyz"
    assert built["output"]["out_file_prefix"] == "ns"
    assert built["output"]["working_dir"] == "."
    # start_config_file replaces the mutually-exclusive start_species.
    assert built["init"]["start_config_file"] == "/remote/start.xyz"
    assert "start_species" not in built["init"]
    # dotted schedule overrides land in the nested config.
    assert built["run"]["n_live"] == 400
    assert built["run"]["n_mcmc_steps"] == 30


def test_set_dotted_nested_and_guard():
    config = {"run": {"n_live": 1}}
    _set_dotted(config, "run.n_live", 9)
    _set_dotted(config, "output.format", "extxyz")
    assert config["run"]["n_live"] == 9
    assert config["output"]["format"] == "extxyz"
    # non-mapping intermediate is an error
    with pytest.raises(WorkflowError):
        _set_dotted({"run": 5}, "run.n_live", 9)


def test_stub_produces_sampled_configs(
    mock_remote_fs, mock_cluster_config, tmp_path
):
    cfg = _write_base_config(tmp_path / "ns.yaml")
    calcdir = tmp_path / "iter_0"
    calcdir.mkdir()
    model_file = tmp_path / "model_0.pkl"
    model_file.write_bytes(b"mock")

    sampler = JaxrensNSSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        config_file=cfg,
        n_gpu_parallel=4,
        stub_n_dead=80,
        stub=True,
    )

    sampler.update(calcdir, model_file)
    sampled = sampler.get_sampled_configs()

    assert sampled.exists()
    atoms = ase.io.read(sampled, index=":")
    # Pure producer: emits ALL dead points (no subsampling). Selection of a
    # subset is a downstream selector stage's job, not the sampler's.
    assert len(atoms) == 80
    for at in atoms:
        # jaxrens trajectory info keys (NOT uncertainty -- selection is a
        # separate downstream concern).
        assert "ns_energy" in at.info
        assert "run_id" in at.info


def test_uniform_selector_thins_the_pool_downstream():
    """The thinning that used to live in the sampler is now a selector.

    The sampler emits the full pool; a ``UniformSelector`` stage reduces it.
    """
    from ase import Atoms
    from remote_handler.workflows.selectors import UniformSelector

    pool = [
        Atoms(
            "Cu2O",
            positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            cell=[6, 6, 6],
            pbc=True,
        )
        for _ in range(10)
    ]
    assert len(UniformSelector(n_configs=3).select(pool)) == 3


def test_body_template_is_shipped():
    body = template_dir() / "body_jaxrens.sh.j2"
    assert body.exists()
    assert "jaxrens run" in body.read_text()
    assert "is_finished.info" in body.read_text()


def test_pending_manifest_points_at_ns_sampler_results(
    mock_remote_fs, mock_cluster_config, tmp_path
):
    """Reattach hook (DetachableTaskMixin): <calcdir>/ns_sampler/results.manifest.json."""
    cfg = _write_base_config(tmp_path / "ns.yaml")
    sampler = JaxrensNSSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        config_file=cfg,
        stub=True,
    )
    calcdir = tmp_path / "6"
    assert sampler.pending_manifest(calcdir) == (
        calcdir / "ns_sampler" / "results.manifest.json"
    )


class _FakeExecutor:
    """Just enough of a real executor for _submit_chain's job.executor.pid."""

    def __init__(self):
        self.pid = None


class _FakeNSTask:
    """Stand-in for SingleScriptTask: records what the sampler asked it to
    do instead of touching a cluster, and fakes a finished run by writing a
    trajectory into local_savepath -- so ``get_sampled_configs`` is testable
    for both the fresh-submit and the reattached (``collect_only``) path."""

    build_calls: list = []
    from_manifest_calls: list = []
    submitted: list = []
    collected: list = []
    finish_result = True
    next_pid = 1000

    def __init__(self, *, local_savepath, _record=True, **kw):
        self.kw = kw
        self.local_savepath = Path(local_savepath)
        self.calcdir = Path(f"/remote/{self.local_savepath.name}")
        self.executor = _FakeExecutor()
        if _record:  # not for from_manifest's reconstruction -- see below
            _FakeNSTask.build_calls.append(
                {"local_savepath": local_savepath, **kw}
            )

    def copy_path(self, path):
        return f"/remote/{Path(path).name}"

    copy_file = copy_path

    def allocate_data(self):
        pass

    def submit(self, only_start=False):
        _FakeNSTask.submitted.append(self)
        _FakeNSTask.next_pid += 1
        self.executor.pid = _FakeNSTask.next_pid

    def finish_check(self):
        return _FakeNSTask.finish_result

    def collect_results(self):
        _FakeNSTask.collected.append(self)
        self.local_savepath.mkdir(parents=True, exist_ok=True)
        from ase import Atoms

        ase.io.write(
            self.local_savepath / "ns.run00.traj.extxyz",
            [Atoms("Cu", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)],
        )

    @classmethod
    def from_manifest(cls, manifest_path, remote_fs=None, **extra):
        cls.from_manifest_calls.append(str(manifest_path))
        return cls(
            local_savepath=Path(str(manifest_path)[: -len(".manifest.json")]),
            _record=False,
        )


@pytest.fixture
def fake_ns_task(monkeypatch):
    _FakeNSTask.build_calls.clear()
    _FakeNSTask.from_manifest_calls.clear()
    _FakeNSTask.submitted.clear()
    _FakeNSTask.collected.clear()
    _FakeNSTask.finish_result = True
    _FakeNSTask.next_pid = 1000
    monkeypatch.setattr("jaxrens_al.run.sampler.SingleScriptTask", _FakeNSTask)
    monkeypatch.setattr(
        "remote_handler.core.tasks.SingleScriptTask", _FakeNSTask
    )
    return _FakeNSTask


def test_fresh_run_builds_and_submits_a_task(
    mock_remote_fs, mock_cluster_config, tmp_path, fake_ns_task
):
    """The normal (not reattached) path: builds a task, ships the model,
    submits, and collects -- unlike collect_only, sampler_input IS used."""
    cfg = _write_base_config(tmp_path / "ns.yaml")
    calcdir = tmp_path / "0"
    calcdir.mkdir()
    model_file = tmp_path / "model_0.pkl"
    model_file.write_bytes(b"mock")

    sampler = JaxrensNSSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        config_file=cfg,
        out_file_prefix="ns",
    )
    sampler.update(calcdir, model_file)
    assert len(fake_ns_task.build_calls) == 1  # built, not reattached
    assert not fake_ns_task.from_manifest_calls

    sampled = sampler.get_sampled_configs()
    assert len(fake_ns_task.submitted) == 1  # actually submitted
    assert len(fake_ns_task.collected) == 1
    assert sampled.exists()


def test_collect_only_reattaches_without_resubmitting(
    mock_remote_fs, mock_cluster_config, tmp_path, fake_ns_task
):
    """collect_only=True must never touch sampler_input/ship a model/build a
    fresh task, and get_sampled_configs must collect without submitting --
    the behaviour StagedActiveLearningHandler._reattach_if_pending relies on
    to avoid duplicating an in-flight job after a driver crash."""
    cfg = _write_base_config(tmp_path / "ns.yaml")
    calcdir = tmp_path / "6"
    calcdir.mkdir()

    sampler = JaxrensNSSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        config_file=cfg,
        out_file_prefix="ns",
        collect_only=True,
    )
    # sampler_input=None -- collect_only must never read it (no model to
    # inject, no job to build).
    sampler.update(calcdir, None)
    assert not fake_ns_task.build_calls  # no fresh task built
    assert fake_ns_task.from_manifest_calls == [
        str(calcdir / "ns_sampler" / "results.manifest.json")
    ]

    sampled = sampler.get_sampled_configs()
    assert not fake_ns_task.submitted  # never resubmitted
    assert len(fake_ns_task.collected) == 1
    assert sampled.exists()


def test_collect_only_raises_if_job_not_actually_finished(
    mock_remote_fs, mock_cluster_config, tmp_path, fake_ns_task
):
    """No is_finished.info (still running, killed, or timed out) must raise
    rather than silently trusting whatever partial trajectory is on disk."""
    fake_ns_task.finish_result = False
    cfg = _write_base_config(tmp_path / "ns.yaml")
    calcdir = tmp_path / "6"
    calcdir.mkdir()

    sampler = JaxrensNSSampler(
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        config_file=cfg,
        collect_only=True,
        reattach_max_wait=0,  # fail on the first poll, no real sleeping
    )
    sampler.update(calcdir, None)
    with pytest.raises(TaskError, match="did not finish successfully"):
        sampler.get_sampled_configs()
    assert not fake_ns_task.collected


# ---------------------------------------------------------------------------
# Auto-resume chaining (max_resubmits > 1): sbatch --dependency=afternotok.
# ---------------------------------------------------------------------------
class _FakeCommandResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def chain_remote_fs():
    """A remote_fs whose run_command fakes sbatch/scancel, recording calls."""
    fs = MagicMock()
    fs.slurm_config = {"savepath": "/remote/path"}
    fs.run_command_calls = []
    next_id = {"n": 2000}

    def _run_command(command):
        fs.run_command_calls.append(command)
        if command.startswith("scancel"):
            return _FakeCommandResult(0, "", "")
        next_id["n"] += 1
        return _FakeCommandResult(0, f"Submitted batch job {next_id['n']}\n")

    fs.run_command.side_effect = _run_command
    return fs


def test_max_resubmits_rejects_less_than_one(
    mock_remote_fs, mock_cluster_config, tmp_path
):
    cfg = _write_base_config(tmp_path / "ns.yaml")
    with pytest.raises(WorkflowError, match="max_resubmits"):
        JaxrensNSSampler(
            remote_fs=mock_remote_fs,
            cluster_config=mock_cluster_config,
            config_file=cfg,
            stub=True,
            max_resubmits=0,
        )


def test_chain_submission_queues_standbys_and_cleans_up_on_success(
    mock_cluster_config, tmp_path, fake_ns_task, chain_remote_fs
):
    """max_resubmits=3: job 1 fired with only_start, 2 standby links queued
    via sbatch --dependency=afternotok, all cancelled once collected."""
    cfg = _write_base_config(tmp_path / "ns.yaml")
    calcdir = tmp_path / "0"
    calcdir.mkdir()
    model_file = tmp_path / "model_0.pkl"
    model_file.write_bytes(b"mock")

    sampler = JaxrensNSSampler(
        remote_fs=chain_remote_fs,
        cluster_config=mock_cluster_config,
        config_file=cfg,
        out_file_prefix="ns",
        max_resubmits=3,
    )
    sampler.update(calcdir, model_file)
    sampler.get_sampled_configs()

    # Only job 1 ever goes through task.submit() -- the standby links are
    # raw sbatch calls, not separate SingleScriptTask instances.
    assert len(fake_ns_task.submitted) == 1

    sbatch_calls = [
        c for c in chain_remote_fs.run_command_calls if "sbatch" in c
    ]
    assert len(sbatch_calls) == 2  # links 2 and 3
    assert "--dependency=afternotok:" in sbatch_calls[0]
    assert "--dependency=afternotok:" in sbatch_calls[1]
    # link 2 depends on job 1's id, link 3 on link 2's (chained, not all on
    # job 1).
    job1_id = str(fake_ns_task.submitted[0].executor.pid)
    assert f"afternotok:{job1_id}" in sbatch_calls[0]
    assert f"afternotok:{job1_id}" not in sbatch_calls[1]

    # Finished on the very first check (fake finish_result=True) -> every
    # chain id (including job 1's own, a harmless no-op) gets cancelled in
    # one scancel call, and the sidecar is removed.
    scancel_calls = [
        c for c in chain_remote_fs.run_command_calls if "scancel" in c
    ]
    assert len(scancel_calls) == 1
    assert job1_id in scancel_calls[0]
    assert not sampler._chain_path().exists()


def test_chain_stops_early_when_sbatch_output_is_unparsable(
    mock_cluster_config, tmp_path, fake_ns_task
):
    """A chain link whose sbatch call fails shouldn't crash the sampler --
    it just stops queuing further standbys with whatever it already has."""
    cfg = _write_base_config(tmp_path / "ns.yaml")
    calcdir = tmp_path / "0"
    calcdir.mkdir()
    model_file = tmp_path / "model_0.pkl"
    model_file.write_bytes(b"mock")

    fs = MagicMock()
    fs.run_command.return_value = _FakeCommandResult(1, "", "sbatch: error")

    sampler = JaxrensNSSampler(
        remote_fs=fs,
        cluster_config=mock_cluster_config,
        config_file=cfg,
        out_file_prefix="ns",
        max_resubmits=3,
    )
    sampler.update(calcdir, model_file)
    sampler.get_sampled_configs()  # must not raise

    # One failed attempt at link 2 (chain stops there), then one scancel
    # call for whatever it did manage to queue (just job 1's own id).
    assert fs.run_command.call_count == 2
    first_call_args = fs.run_command.call_args_list[0].args[0]
    assert "sbatch" in first_call_args
    second_call_args = fs.run_command.call_args_list[1].args[0]
    assert "scancel" in second_call_args


# ---------------------------------------------------------------------------
# Waiting on the chain as a whole (not just link 1's job id).
# ---------------------------------------------------------------------------
@pytest.fixture
def squeue_remote_fs():
    """A remote_fs faking sbatch/scancel plus a scripted ``squeue``.

    ``fs.squeue`` is a list of ``(returncode, stdout, stderr)`` answers,
    consumed one per squeue call (the last one repeats).
    """
    fs = MagicMock()
    fs.slurm_config = {"savepath": "/remote/path"}
    fs.run_command_calls = []
    fs.squeue = [(0, "RUNNING None\n", "")]
    next_id = {"n": 2000}

    def _run_command(command):
        fs.run_command_calls.append(command)
        if command.startswith("squeue"):
            answer = fs.squeue.pop(0) if len(fs.squeue) > 1 else fs.squeue[0]
            return _FakeCommandResult(*answer)
        if command.startswith("scancel"):
            return _FakeCommandResult(0, "", "")
        next_id["n"] += 1
        return _FakeCommandResult(0, f"Submitted batch job {next_id['n']}\n")

    fs.run_command.side_effect = _run_command
    return fs


def _script_finish_checks(monkeypatch, fake_ns_task, results):
    """Make finish_check return ``results`` in turn (the last repeats)."""
    results = list(results)

    def _finish_check(self):
        return results.pop(0) if len(results) > 1 else results[0]

    monkeypatch.setattr(fake_ns_task, "finish_check", _finish_check)
    monkeypatch.setattr("jaxrens_al.run.sampler.time.sleep", lambda s: None)


def _chain_sampler(fs, cluster_config, tmp_path, **kw):
    cfg = _write_base_config(tmp_path / "ns.yaml")
    calcdir = tmp_path / "5"
    calcdir.mkdir()
    model_file = tmp_path / "model_0.pkl"
    model_file.write_bytes(b"mock")
    sampler = JaxrensNSSampler(
        remote_fs=fs,
        cluster_config=cluster_config,
        config_file=cfg,
        out_file_prefix="ns",
        max_resubmits=3,
        **kw,
    )
    sampler.update(calcdir, None if kw.get("collect_only") else model_file)
    return sampler


def test_chain_survives_link_one_timing_out(
    monkeypatch,
    mock_cluster_config,
    tmp_path,
    fake_ns_task,
    squeue_remote_fs,
):
    """run32 it 5: link 1 hits its walltime and leaves the queue before
    is_finished.info exists; link 2 resumes and finishes. The sampler must
    keep waiting through that and collect, not raise when link 1 ends."""
    squeue_remote_fs.squeue = [
        (0, "RUNNING None\nPENDING Dependency\nPENDING Dependency\n", ""),
        # link 1 TIMEOUT -> gone; link 2 queued, then running
        (0, "PENDING Priority\nPENDING Dependency\n", ""),
        (0, "RUNNING None\nPENDING Dependency\n", ""),
    ]
    _script_finish_checks(monkeypatch, fake_ns_task, [False] * 3 + [True])
    sampler = _chain_sampler(squeue_remote_fs, mock_cluster_config, tmp_path)

    sampled = sampler.get_sampled_configs()

    assert sampled.exists()
    assert len(fake_ns_task.collected) == 1
    squeues = [c for c in squeue_remote_fs.run_command_calls if "squeue" in c]
    assert len(squeues) == 3
    assert any("scancel" in c for c in squeue_remote_fs.run_command_calls)
    assert not sampler._chain_path().exists()


def test_reattach_mid_chain_waits_on_the_chain(
    monkeypatch,
    mock_cluster_config,
    tmp_path,
    fake_ns_task,
    squeue_remote_fs,
):
    """A driver restarted while link 2 runs finds the chain sidecar and
    waits on the chain -- no resubmit, no failure on link 1 being gone."""
    sampler = _chain_sampler(
        squeue_remote_fs, mock_cluster_config, tmp_path, collect_only=True
    )
    sampler._chain_path().parent.mkdir(parents=True, exist_ok=True)
    sampler._chain_path().write_text('["11", "12", "13"]')
    squeue_remote_fs.squeue = [(0, "RUNNING None\nPENDING Dependency\n", "")]
    _script_finish_checks(monkeypatch, fake_ns_task, [False, False, True])

    sampler.get_sampled_configs()

    assert not fake_ns_task.submitted
    assert not any("sbatch" in c for c in squeue_remote_fs.run_command_calls)
    assert len(fake_ns_task.collected) == 1
    assert "scancel 11 12 13" in squeue_remote_fs.run_command_calls


def test_reattach_to_finished_chain_collects_immediately(
    monkeypatch,
    mock_cluster_config,
    tmp_path,
    fake_ns_task,
    squeue_remote_fs,
):
    """The run32 restart: the marker is already there, so collect without
    asking squeue at all, and cancel the never-runnable link 3."""
    sampler = _chain_sampler(
        squeue_remote_fs, mock_cluster_config, tmp_path, collect_only=True
    )
    sampler._chain_path().parent.mkdir(parents=True, exist_ok=True)
    sampler._chain_path().write_text('["11", "12", "13"]')
    _script_finish_checks(monkeypatch, fake_ns_task, [True])

    sampler.get_sampled_configs()

    assert squeue_remote_fs.run_command_calls == ["scancel 11 12 13"]
    assert len(fake_ns_task.collected) == 1


@pytest.mark.parametrize(
    "answer",
    [
        # link 2 exited 0 without the marker, so link 3 can never start
        (0, "PENDING DependencyNeverSatisfied\n", ""),
        # every link already purged from SLURM
        (1, "", "slurm_load_jobs error: Invalid job id specified\n"),
    ],
    ids=["never-satisfiable", "all-purged"],
)
def test_dead_chain_without_marker_raises(
    monkeypatch,
    mock_cluster_config,
    tmp_path,
    fake_ns_task,
    squeue_remote_fs,
    answer,
):
    squeue_remote_fs.squeue = [answer]
    _script_finish_checks(monkeypatch, fake_ns_task, [False])
    sampler = _chain_sampler(squeue_remote_fs, mock_cluster_config, tmp_path)

    with pytest.raises(TaskError, match="can still run"):
        sampler.get_sampled_configs()
    assert not fake_ns_task.collected


def test_unreachable_squeue_keeps_waiting(
    monkeypatch,
    mock_cluster_config,
    tmp_path,
    fake_ns_task,
    squeue_remote_fs,
):
    """An SSH outage (squeue exit 255) is not a dead chain."""
    squeue_remote_fs.squeue = [(255, "", "ssh: connect timed out\n")]
    _script_finish_checks(monkeypatch, fake_ns_task, [False] * 4 + [True])
    sampler = _chain_sampler(squeue_remote_fs, mock_cluster_config, tmp_path)

    sampler.get_sampled_configs()

    assert len(fake_ns_task.collected) == 1


def test_chain_wait_respects_max_wait(
    monkeypatch,
    mock_cluster_config,
    tmp_path,
    fake_ns_task,
    squeue_remote_fs,
):
    _script_finish_checks(monkeypatch, fake_ns_task, [False])
    sampler = _chain_sampler(
        squeue_remote_fs,
        mock_cluster_config,
        tmp_path,
        reattach_max_wait=0,
    )

    with pytest.raises(TaskError, match="exceeding max_wait"):
        sampler.get_sampled_configs()
