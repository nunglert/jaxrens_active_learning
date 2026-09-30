"""Tests for the phase-grid tool: NS-weighted sample selection and the
torch-sim staged-relaxation processor (Lennard-Jones backend, CPU)."""

import numpy as np
import pytest

pytest.importorskip("remote_handler.processing")
pytest.importorskip("jaxrens.postprocess")

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402
from remote_handler.exceptions import ScriptInputError  # noqa: E402
from remote_handler.processing.registry import (  # noqa: E402
    REGISTRY,
    validate_args,
)

from jaxrens_al.run.phase_grid import KB_EV, most_likely_frames  # noqa: E402

pytestmark = pytest.mark.integration


def _frames(iters):
    frames = []
    for i in iters:
        at = Atoms("Sn", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
        at.info["iter"] = int(i)
        frames.append(at)
    return frames


def test_most_likely_frames_moves_to_lower_enthalpy_when_cooling():
    n_live, n_dead = 50, 2000
    # enthalpy falls ever more slowly along the run, as for an NS dead-point
    # sequence approaching the ground state -> an interior peak that moves with T
    h = -2.0 * (1.0 - np.exp(-np.arange(n_dead) / 400.0))
    frames = _frames(range(0, n_dead, 10))
    idx, shares = most_likely_frames(frames, h, n_live, [800.0, 300.0, 100.0])
    iters = [frames[i].info["iter"] for i in idx]
    assert iters[0] < iters[1] < iters[2]
    assert np.all((shares > 0) & (shares <= 1))


def test_most_likely_frames_matches_the_analytic_peak():
    # linear H(i) = -a i and log X_i = i log t: weight peaks where the slopes
    # cancel, i.e. none inside unless T is right -> check the boundary cases
    n_live, n_dead = 100, 5000
    log_t = np.log(n_live - 1) - np.log(n_live)
    a = 1e-3
    h = -a * np.arange(n_dead)
    frames = _frames(range(n_dead))
    t_star = a / (-log_t * KB_EV)
    idx_hot, _ = most_likely_frames(frames, h, n_live, [2 * t_star])
    idx_cold, _ = most_likely_frames(frames, h, n_live, [0.5 * t_star])
    assert idx_hot[0] == 0
    assert idx_cold[0] == n_dead - 1


def test_most_likely_frames_skips_frames_past_the_dead_points():
    frames = _frames([0, 10, 20, 999])
    idx, _ = most_likely_frames(frames, np.linspace(0, -1, 30), 20, [50.0])
    assert frames[idx[0]].info["iter"] == 20


def test_torchsim_relax_is_registered_with_schema():
    from jaxrens_al.run.processors import TorchSimRelaxProcessor

    assert "torchsim_relax" in REGISTRY
    proc = TorchSimRelaxProcessor(
        backend="lj", device="cpu", stages=("positions", "cell")
    )
    args = proc.build_args("/remote")
    validate_args("torchsim_relax", args)
    assert proc.shared_files() == []
    assert args["model_remote"] == ""


def test_torchsim_relax_stages_a_model_file(tmp_path):
    from jaxrens_al.run.processors import TorchSimRelaxProcessor

    model = tmp_path / "m.model"
    model.write_text("x")
    proc = TorchSimRelaxProcessor(model, device="cpu")
    assert proc.shared_files() == [str(model)]
    assert proc.build_args("/remote")["model_remote"] == "/remote/m.model"


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"backend": "mace"}, "model_remote"),
        ({"backend": "lj", "stages": ["cell", "melt"]}, "stages"),
        ({"backend": "lj", "fmax": 0}, "fmax"),
        ({"backend": "nequip"}, "backend"),
    ],
)
def test_torchsim_relax_args_validation(kw, match):
    from jaxrens_al.run.processors import TorchSimRelaxArgs

    with pytest.raises(ScriptInputError, match=match):
        TorchSimRelaxArgs(**kw).validate()


def test_torchsim_relax_run_relaxes_per_pressure():
    pytest.importorskip("torch_sim")
    from jaxrens_al.run.processors import torchsim_relax

    atoms = []
    for p in (0.0, 0.0, 2.0):
        at = bulk("Cu", "fcc", a=3.4, cubic=True).repeat(2)
        at.rattle(0.05, seed=3)
        at.info["grid_P"] = p
        atoms.append(at)
    out = torchsim_relax.run(
        atoms, {"backend": "lj", "device": "cpu", "fmax": 0.01, "steps": 2000}
    )
    assert len(out) == 3
    for at, p in zip(out, (0.0, 0.0, 2.0)):
        assert at.info["is_converged"]
        assert at.info["relax_fmax"] < 0.01
        assert at.info["relax_pressure"] == pytest.approx(p, abs=0.1)
        assert at.info["grid_P"] == p
    # compressed cell is denser, and identical inputs relax identically
    assert out[2].info["relax_volume"] < out[0].info["relax_volume"]
    assert out[0].info["relax_enthalpy"] == pytest.approx(
        out[1].info["relax_enthalpy"]
    )


def test_torchsim_relax_positions_only_keeps_the_cell():
    pytest.importorskip("torch_sim")
    from jaxrens_al.run.processors import torchsim_relax

    at = bulk("Cu", "fcc", a=3.4, cubic=True).repeat(2)
    at.rattle(0.05, seed=4)
    cell = at.get_cell().copy()
    (out,) = torchsim_relax.run(
        [at],
        {
            "backend": "lj",
            "device": "cpu",
            "stages": ["positions"],
            "fmax": 0.01,
            "steps": 2000,
        },
    )
    np.testing.assert_allclose(out.get_cell(), cell)
    assert out.info["relax_fmax"] < 0.01
