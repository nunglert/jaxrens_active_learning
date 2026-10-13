"""Tests for the phase-grid tool: NS-weighted sample selection and the
torch-sim staged-relaxation processor (Lennard-Jones backend, CPU)."""

import numpy as np
import pytest

pytest.importorskip("remote_handler.processing")
pytest.importorskip("jaxrens.postprocess")

from ase import Atoms, units  # noqa: E402
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


# --- jax_cell_relax (jaxrens backend: NeuralIL / mace-jax / lj) -----------
_LJ_ARGS = {"backend": "lj", "supercell_trafo": [3, 3, 3]}


def _lj_cell(seed, a=3.4):
    at = bulk("Cu", "fcc", a=a, cubic=True).repeat(2)
    at.rattle(0.05, seed=seed)
    return at


def test_jax_cell_relax_is_registered_with_schema(tmp_path):
    from jaxrens_al.run.processors import JaxCellRelaxProcessor

    assert "jax_cell_relax" in REGISTRY
    pkl = tmp_path / "trainer_it0_0.pkl"
    pkl.write_text("x")
    proc = JaxCellRelaxProcessor(
        pkl,
        supercell_trafo=(2, 2, 2),
        max_neighbors_list=(20, 40),
        max_neighbors_offset=3,
    )
    args = proc.build_args("/remote")
    validate_args("jax_cell_relax", args)
    assert proc.shared_files() == [str(pkl)]
    assert args["model_remote"] == "/remote/trainer_it0_0.pkl"
    assert args["backend"] == "neuralil" and not args["is_archive"]
    assert args["supercell_trafo"] == [2, 2, 2]
    assert args["max_neighbors_list"] == [20, 40]


def test_jax_cell_relax_tars_a_bundle_dir(tmp_path):
    from jaxrens_al.run.processors import JaxCellRelaxProcessor

    bundle = tmp_path / "trainer_it0_0.bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    proc = JaxCellRelaxProcessor(bundle, backend="mace")
    (staged,) = proc.shared_files()
    assert staged.endswith("trainer_it0_0.bundle.tar")
    args = proc.build_args("/remote")
    assert args["is_archive"]
    assert args["model_remote"] == "/remote/trainer_it0_0.bundle.tar"


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"backend": "neuralil"}, "model_remote"),
        ({"backend": "lj", "stages": ["melt"]}, "stages"),
        ({"backend": "lj", "fmax": -1}, "fmax"),
        ({"backend": "lj", "supercell_trafo": [2, 2]}, "supercell_trafo"),
        ({"backend": "nequip"}, "backend"),
    ],
)
def test_jax_cell_relax_args_validation(kw, match):
    from jaxrens_al.run.processors import JaxCellRelaxArgs

    with pytest.raises(ScriptInputError, match=match):
        JaxCellRelaxArgs(**kw).validate()


def test_jax_cell_relax_stress_matches_finite_differences(monkeypatch):
    """The autodiff strain-derivative stress agrees with ASE's numerical
    one -- i.e. the backend's cell dependence is differentiable."""
    from jaxrens_al.run.processors import jax_cell_relax as jcr

    # The LJ stand-in has a hard cutoff, so E(strain) jumps wherever a pair
    # crosses it: keep every pair well clear. fcc a=3.5 has shells at 5.53
    # and 6.06 A; a 5.8 A cutoff, a 1% shear and a 0.01 A rattle move no
    # pair across it, even under the finite-difference strain.
    monkeypatch.setattr(jcr, "LJ_PARAMS", {**jcr.LJ_PARAMS, "cutoff": 5.8})
    at = bulk("Cu", "fcc", a=3.5, cubic=True).repeat(2)
    at.rattle(0.01, seed=1)
    at.set_cell(
        at.cell @ np.array([[1, 0.01, 0], [0, 1, 0.005], [0, 0, 1]]),
        scale_atoms=True,
    )
    backend, _ = jcr._load(_LJ_ARGS, None, ["Cu"])
    efs = jcr.make_energy_forces_stress(backend)
    types = np.zeros(len(at), dtype=int)
    at.calc = jcr._calculator(backend, efs, types, (), 4, 0)
    analytic = at.get_stress()
    numeric = at.calc.calculate_numerical_stress(at, d=1e-4)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=2e-5)
    # forces are -dE/dx of the same call. The backend computes in float32
    # (energy ulp ~1e-6 eV), so the displacement must be large enough for
    # the energy difference to resolve them: d=5e-3 A, truncation O(d^2).
    np.testing.assert_allclose(
        at.get_forces(),
        at.calc.calculate_numerical_forces(at, d=5e-3),
        rtol=1e-2,
        atol=5e-4,
    )


def test_jax_cell_relax_run_relaxes_per_pressure():
    from jaxrens_al.run.processors import jax_cell_relax

    atoms = []
    for p in (0.0, 0.0, 2.0):
        at = _lj_cell(seed=3)
        at.info["grid_P"] = p
        atoms.append(at)
    out = jax_cell_relax.run(atoms, {**_LJ_ARGS, "fmax": 0.01, "steps": 2000})
    assert len(out) == 3
    # FrechetCellFilter scales the cell gradient by cell_factor = N, so
    # fmax bounds the stress only to ~fmax * N / V (here ~0.14 GPa).
    for at, p in zip(out, (0.0, 0.0, 2.0)):
        n, v = len(at), at.info["relax_volume"] * len(at)
        tol = 0.01 * n / v / units.GPa
        assert at.info["is_converged"]
        assert at.info["relax_fmax"] < 0.01
        assert at.info["relax_pressure"] == pytest.approx(p, abs=1.2 * tol)
        assert at.info["grid_P"] == p
    assert out[2].info["relax_volume"] < out[0].info["relax_volume"]
    assert out[0].info["relax_enthalpy"] == pytest.approx(
        out[1].info["relax_enthalpy"]
    )


def test_jax_cell_relax_positions_only_keeps_the_cell():
    from jaxrens_al.run.processors import jax_cell_relax

    at = _lj_cell(seed=4)
    cell = at.get_cell().copy()
    (out,) = jax_cell_relax.run(
        [at], {**_LJ_ARGS, "stages": ["positions"], "fmax": 0.01}
    )
    np.testing.assert_allclose(out.get_cell(), cell)
    assert out.info["relax_fmax"] < 0.01 and out.info["is_converged"]


def test_jax_cell_relax_marks_a_crashing_structure_failed(monkeypatch):
    from jaxrens_al.run.processors import jax_cell_relax

    calls = {"n": 0}
    real = jax_cell_relax.relax_one

    def flaky(atoms, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("ladder exhausted")
        return real(atoms, *a, **kw)

    monkeypatch.setattr(jax_cell_relax, "relax_one", flaky)
    bad, good = _lj_cell(seed=5), _lj_cell(seed=6)
    out = jax_cell_relax.run(
        [bad, good], {**_LJ_ARGS, "stages": ["positions"], "fmax": 0.05}
    )
    assert out[0].info["relax_failed"] and not out[0].info["is_converged"]
    assert "ladder exhausted" in out[0].info["relax_error"]
    np.testing.assert_allclose(out[0].positions, bad.positions)
    assert out[1].info["is_converged"]


@pytest.mark.parametrize(
    "model, backend, expected",
    [
        ("m.model", "mace", "torchsim"),
        (None, "lj", "torchsim"),
        ("trainer_it0_0.pkl", "neuralil", "jax"),
        ("trainer_it0_0.bundle", "mace", "jax"),
    ],
)
def test_relaxer_follows_the_model(model, backend, expected):
    from jaxrens_al.run.phase_grid import relaxer_for

    assert relaxer_for(model, backend) == expected


def test_phase_grid_job_uses_jax_cell_relax_for_neuralil(
    tmp_path, monkeypatch
):
    import remote_handler.processing as processing

    from jaxrens_al.run.phase_grid import phase_grid_job

    captured = {}
    monkeypatch.setattr(
        processing,
        "RemoteTrajectoryProcessor",
        lambda **kw: captured.update(kw) or kw,
    )
    pkl = tmp_path / "trainer_it0_0.pkl"
    pkl.write_text("x")
    phase_grid_job(
        tmp_path / "picks.extxyz",
        4,
        model=pkl,
        backend="neuralil",
        cluster_config=None,
        remote_fs=None,
        remote_savepath="/remote",
        supercell_trafo=(2, 2, 2),
        max_neighbors_list=(20, 40),
    )
    relax, spacegroup = captured["processors"]
    assert relax.name == "jax_cell_relax" and spacegroup.name == "spacegroup"
    assert relax.supercell_trafo == [2, 2, 2]
    assert relax.max_neighbors_list == [20, 40]
