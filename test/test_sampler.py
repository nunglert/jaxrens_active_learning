"""Stub-mode test for the project's jaxrens Sampler.

Needs the [run] extra (remote_handler); skipped otherwise. Runs entirely in
stub mode -- no cluster, no SSH, no jaxnest -- mirroring remote_handler's own
NSTrajectorySampler stub tests.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("remote_handler")
import ase.io  # noqa: E402
from remote_handler.core.cluster_handler import ClusterConfig  # noqa: E402
from remote_handler.workflows.active_learning_base import Sampler  # noqa: E402

from jaxrens_al.sampler import JaxrensNSSampler  # noqa: E402


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


@pytest.mark.integration
def test_jaxrens_sampler_is_a_sampler(
    mock_remote_fs, mock_cluster_config, tmp_path
):
    ns_input = tmp_path / "ns.inp"
    ns_input.touch()

    sampler = JaxrensNSSampler(
        sampling_script="/path/to/ns",
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        initial_input_file=ns_input,
        n_gpu_parallel=4,
        sample_size=100,
        resample_size=50,
        stub=True,
    )

    # Satisfies the orchestrator's Sampler interface...
    assert isinstance(sampler, Sampler)
    # ...and forwards attribute access to the inner method.
    assert sampler.stub is True
    assert sampler.sample_size == 100
    assert sampler.resample_size == 50


@pytest.mark.integration
def test_jaxrens_sampler_stub_produces_configs(
    mock_remote_fs, mock_cluster_config, tmp_path
):
    calcdir = tmp_path / "iter_0"
    calcdir.mkdir()
    ns_input = calcdir / "ns.inp"
    ns_input.write_text("# Mock NS input")
    model_file = tmp_path / "model_0.pkl"
    model_file.write_bytes(b"mock")

    sampler = JaxrensNSSampler(
        sampling_script="/path/to/ns",
        remote_fs=mock_remote_fs,
        cluster_config=mock_cluster_config,
        initial_input_file=ns_input,
        n_gpu_parallel=4,
        sample_size=80,
        resample_size=None,
        stub=True,
    )

    sampler.update(calcdir, model_file)
    sampled = sampler.get_sampled_configs()

    assert sampled.exists()
    atoms = ase.io.read(sampled, index=":")
    assert len(atoms) == 80
    for at in atoms:
        assert "energy_uncert" in at.info
        assert "forces_uncert" in at.info
