"""Tests for the project's native jaxrens Sampler (JaxrensNSSampler).

Needs the [run] extra (remote_handler); the module self-skips otherwise. The
stub-mode and config-assembly tests run with NO cluster, SSH, or jaxrens --
only remote_handler + ase + pyyaml on the host.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("remote_handler")
import ase.io  # noqa: E402
from remote_handler.core.cluster_handler import ClusterConfig  # noqa: E402
from remote_handler.exceptions import WorkflowError  # noqa: E402
from remote_handler.workflows.active_learning_base import Sampler  # noqa: E402

from jaxrens_al.sampler import JaxrensNSSampler, _set_dotted  # noqa: E402
from jaxrens_al.templates import template_dir  # noqa: E402

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
