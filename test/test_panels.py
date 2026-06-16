"""Tests for the project panels and the data-driven dashboard assembly.

Each panel's ``compute_data`` is exercised against synthesized run artifacts
(no cluster, no real NS/training). The thermodynamics panel is tested through a
seam (``_load_monitor``) so it needs neither jaxrens nor an HDF5 checkpoint.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("alviz")
import ase.io  # noqa: E402
from alviz.panels import HeatmapData, ParityData  # noqa: E402
from ase import Atoms  # noqa: E402

from jaxrens_al.panels.sampling import NSSampledEnergyPanel  # noqa: E402
from jaxrens_al.panels.thermodynamics import (  # noqa: E402
    NSThermodynamicsHeatmap,
)
from jaxrens_al.panels.training import (  # noqa: E402
    NeuralILParityPanel,
    NeuralILTrainingCurve,
)


# --------------------------------------------------------------------------
# Synthetic run-dir helpers
# --------------------------------------------------------------------------
def _write_configs(path, n, info_keys):
    frames = []
    rng = np.random.default_rng(0)
    for i in range(n):
        at = Atoms(
            "Cu2O",
            positions=rng.uniform(0, 3, (3, 3)),
            cell=[6, 6, 6],
            pbc=True,
        )
        for k, v in info_keys(i).items():
            at.info[k] = v
        at.arrays["REF_forces"] = rng.uniform(-1, 1, (3, 3))
        frames.append(at)
    ase.io.write(path, frames)


def _make_calchome(tmp_path):
    calchome = tmp_path / "calchome"
    (calchome / "sampler_output").mkdir(parents=True)
    (calchome / "evaluator_output").mkdir(parents=True)

    progress = []
    for it in range(2):
        s = f"sampler_output/s{it}.extxyz"
        e = f"evaluator_output/e{it}.extxyz"
        _write_configs(
            calchome / s,
            5,
            lambda i, it=it: {
                "ns_energy": -10.0 - it + 0.1 * i,
                "run_id": i % 2,
            },
        )
        _write_configs(
            calchome / e,
            3,
            lambda i, it=it: {"REF_energy": -9.0 - it + 0.1 * i},
        )
        progress.append(
            {
                "steps": [0, 0, 0],
                "sampler_output": [s],
                "evaluator_output": [e],
            }
        )

        # NeuralIL training artifacts for this iteration. The NeuralIL trainer
        # (remote_handler NeuralilILTrainer) writes under <calcdir>/training.
        tdir = calchome / str(it) / "training" / "results"
        tdir.mkdir(parents=True)
        _write_npz(tdir / "val.npz")
        _write_error_log(tdir / "error_0.txt")

    (calchome / "AL_progress.json").write_text(json.dumps(progress))
    return calchome


def _write_npz(path):
    rng = np.random.default_rng(1)
    n = 8
    e_obs = rng.uniform(-30, -20, n)
    np.savez(
        path,
        e_obs=e_obs,
        e_pred=e_obs + rng.normal(0, 0.1, n),
        f_obs=rng.uniform(-1, 1, (n, 3, 3)),
        f_pred=rng.uniform(-1, 1, (n, 3, 3)),
        types=np.array([[0, 0, 1]] * n),  # 3 atoms each
    )


def _write_error_log(path):
    path.write_text(
        "i_epoch=0\n"
        "Model #1:\n"
        "F_RMSE = 0.50 eV/A. \n"
        "E_RMSE = 0.20 eV. \n"
        "Model #2:\n"
        "F_RMSE = 0.60 eV/A. \n"
        "E_RMSE = 0.30 eV. \n"
        "i_epoch=10\n"
        "Model #1:\n"
        "F_RMSE = 0.10 eV/A. \n"
        "E_RMSE = 0.05 eV. \n"
    )


# --------------------------------------------------------------------------
# Panel compute tests
# --------------------------------------------------------------------------
def test_neuralil_parity_from_npz(tmp_path):
    _write_npz(tmp_path / "val.npz")
    panel = NeuralILParityPanel(
        npz_files=[tmp_path / "val.npz"], datapath=tmp_path
    )
    data = panel.compute_data({})
    assert isinstance(data, ParityData)
    assert len(data.e_obs) == 8 and len(data.e_pred) == 8
    # per-atom normalization by the 3-atom types -> energies near -8..-7.
    assert all(-12 < e < -5 for e in data.e_obs)
    assert data.f_obs is not None and len(data.f_obs) == 8 * 3 * 3
    assert data.e_rmse >= 0


def test_training_curve_parses_error_log(tmp_path):
    _write_error_log(tmp_path / "error_0.txt")
    panel = NeuralILTrainingCurve(
        error_files=[tmp_path / "error_0.txt"], datapath=tmp_path
    )
    data = panel.compute_data({})
    assert data["epoch"] == [0, 10]
    # epoch 0 averages the two models' E_RMSE (0.20, 0.30) -> 0.25.
    assert data["e_rmse"][0] == pytest.approx(0.25)
    assert data["f_rmse"][0] == pytest.approx(0.55)
    assert data["e_rmse"][1] == pytest.approx(0.05)


def test_sampled_energy_panel(tmp_path):
    calchome = _make_calchome(tmp_path)
    (calchome / "dash").mkdir()
    panel = NSSampledEnergyPanel(calchome=calchome, datapath=calchome / "dash")
    data = panel.compute_data({})
    assert data["iterations"] == [0, 1]
    assert len(data["energies"]) == 2
    assert all(len(e) == 5 for e in data["energies"])


class _FakeMonitor:
    """Stand-in for jaxrens Monitor with the two methods the panel calls."""

    def __init__(self, n_T):
        self.dead_volumes = np.ones(7)

    def heat_capacity(self, T, k_B=1.0):
        return np.abs(np.asarray(T, dtype=float)) + 1.0

    def expectation(self, obs, T, k_B=1.0):
        return np.asarray(T, dtype=float) * 0 + float(np.mean(obs))


def test_thermodynamics_panel_with_fake_monitor(tmp_path):
    panel = NSThermodynamicsHeatmap(
        result_dirs=[tmp_path / "run0", tmp_path / "run1"],
        quantity="heat_capacity",
        n_T=12,
        datapath=tmp_path,
    )
    panel._load_monitor = lambda d: _FakeMonitor(12)
    data = panel.compute_data({})
    assert isinstance(data, HeatmapData)
    assert len(data.t) == 12
    assert len(data.obs) == 2  # one row per result dir
    assert len(data.obs[0]) == 12
    assert data.quantity == "heat_capacity"


def test_thermodynamics_panel_empty_dirs_raises(tmp_path):
    panel = NSThermodynamicsHeatmap(result_dirs=[], datapath=tmp_path)
    with pytest.raises(RuntimeError):
        panel.compute_data({})


# --------------------------------------------------------------------------
# Dashboard assembly (data-driven): registers only panels with inputs
# --------------------------------------------------------------------------
def test_build_dashboard_assembles_present_panels(tmp_path):
    from jaxrens_al.dashboard import build_dashboard

    calchome = _make_calchome(tmp_path)
    board = build_dashboard(calchome, compute=True)

    names = {e.name for e in board._entries}
    # Present inputs -> these panels registered (no NS .energies -> no thermo).
    assert {
        "dataset_growth",
        "sampled_energy_volume",
        "sampled_ns_energy",
        "neuralil_parity",
        "neuralil_training_curve",
    } <= names
    assert "ns_thermodynamics" not in names
    board.savefig(calchome / "dashboard.pdf")
    assert (calchome / "dashboard.pdf").exists()


# --------------------------------------------------------------------------
# Per-iteration (multipage) assembly: one page per AL iteration
# --------------------------------------------------------------------------
def test_build_iteration_dashboard_present_panels(tmp_path):
    from jaxrens_al.dashboard import build_iteration_dashboard

    calchome = _make_calchome(tmp_path)
    datapath = calchome / "dashboard" / "1"
    datapath.mkdir(parents=True)

    board = build_iteration_dashboard(
        calchome, 1, datapath=datapath, compute=True
    )
    names = {e.name for e in board._entries}
    # Iteration 1 has evaluated + sampled + NeuralIL training artifacts, but no
    # NS .energies -> no thermodynamics panel. Dataset growth is cumulative
    # (summary-only), so it is intentionally absent from a per-iteration page.
    assert {
        "sampled_energy_volume",
        "sampled_ns_energy",
        "neuralil_parity",
        "neuralil_training_curve",
    } == names
    assert "dataset_growth" not in names

    # The single sampled-energy box is positioned at this iteration's index.
    box_panel = next(
        e for e in board._entries if e.name == "sampled_ns_energy"
    )
    assert box_panel.data["iterations"] == [1]
