"""Tests for the project panels' computation, each in isolation.

Every panel's ``compute_data`` is exercised against minimal synthesized inputs
(no cluster, no real NS/training). The thermodynamics panel is tested through a
seam (``_load_monitor``) so it needs neither jaxrens nor an HDF5 checkpoint --
the model for keeping heavy/optional deps out of unit tests.

Dashboard *assembly* (which panels get registered, selection, layout) lives in
test_dashboard.py; this file is only about the panels themselves.
"""

import numpy as np
import pytest

pytest.importorskip("alviz")
from alviz.panels import HeatmapData, ParityData  # noqa: E402

from jaxrens_al.panels.progress import DatasetGrowthPanel  # noqa: E402
from jaxrens_al.panels.quests import (  # noqa: E402
    QuestsPCAPanel,
    ResampledGridPanel,
)
from jaxrens_al.panels.sampling import NSSampledEnergyPanel  # noqa: E402
from jaxrens_al.panels.thermodynamics import (  # noqa: E402
    NSThermodynamicsHeatmap,
)
from jaxrens_al.panels.training import (  # noqa: E402
    NeuralILParityPanel,
    NeuralILTrainingCurve,
)

pytestmark = pytest.mark.local


# --------------------------------------------------------------------------
# Dataset-growth panel: per-iteration and cumulative config counts
# --------------------------------------------------------------------------
def test_dataset_growth_counts(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2, n_evaluated=[3, 5])
    (calchome / "dash").mkdir()
    panel = DatasetGrowthPanel(calchome=calchome, datapath=calchome / "dash")
    data = panel.compute_data({})
    assert data["per_iteration"] == [3, 5]
    assert data["cumulative"] == [3, 8]


# --------------------------------------------------------------------------
# Training panels
# --------------------------------------------------------------------------
def test_neuralil_parity_from_npz(tmp_path, write_npz):
    write_npz(tmp_path / "val.npz")
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


def test_training_curve_parses_error_log(tmp_path, write_error_log):
    write_error_log(tmp_path / "error_0.txt")
    panel = NeuralILTrainingCurve(
        error_files=[tmp_path / "error_0.txt"], datapath=tmp_path
    )
    data = panel.compute_data({})
    assert data["epoch"] == [0, 10]
    # epoch 0 averages the two models' E_RMSE (0.20, 0.30) -> 0.25.
    assert data["e_rmse"][0] == pytest.approx(0.25)
    assert data["f_rmse"][0] == pytest.approx(0.55)
    assert data["e_rmse"][1] == pytest.approx(0.05)


# --------------------------------------------------------------------------
# Sampled-energy panel: cross-iteration vs single-iteration mode
# --------------------------------------------------------------------------
def test_sampled_energy_panel_cross_iteration(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2)
    (calchome / "dash").mkdir()
    panel = NSSampledEnergyPanel(calchome=calchome, datapath=calchome / "dash")
    data = panel.compute_data({})
    assert data["iterations"] == [0, 1]
    assert len(data["energies"]) == 2
    assert all(len(e) == 5 for e in data["energies"])


def test_sampled_energy_panel_single_iteration(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2)
    (calchome / "dash").mkdir()
    files = [calchome / "sampler_output" / "s1.extxyz"]
    panel = NSSampledEnergyPanel(
        config_files=files,
        iteration_index=1,
        datapath=calchome / "dash",
    )
    data = panel.compute_data({})
    # Single box, labelled with the iteration index it was built for.
    assert data["iterations"] == [1]
    assert len(data["energies"]) == 1 and len(data["energies"][0]) == 5


def test_sampled_energy_panel_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValueError):
        NSSampledEnergyPanel(datapath=tmp_path)  # neither source given
    with pytest.raises(ValueError):
        NSSampledEnergyPanel(
            calchome=tmp_path, config_files=[], datapath=tmp_path
        )  # both given


# --------------------------------------------------------------------------
# Thermodynamics panel (tested through the _load_monitor seam)
# --------------------------------------------------------------------------
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
# QUESTS resampling panels
# --------------------------------------------------------------------------
def test_quests_pca_panel_reads_structures(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=1, with_resampling=True)
    pca = calchome / "0" / "ns_sampler" / "results" / "quests_pca.json"
    panel = QuestsPCAPanel(pca_json=pca, datapath=tmp_path)
    data = panel.compute_data({})
    # 6 candidates synthesized, first 2 selected.
    assert len(data["pc1"]) == 6 and len(data["pc2"]) == 6
    assert sum(data["selected"]) == 2
    assert data["evr"] == [0.6, 0.25]


def test_resampled_grid_panel_reads_selected(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=1, with_resampling=True)
    prov = (
        calchome / "0" / "ns_sampler" / "results"
        / "resampled_provenance.json"
    )
    panel = ResampledGridPanel(provenance_json=prov, datapath=tmp_path)
    data = panel.compute_data({})
    # 2 selected: run_ids [0, 1], iters [0, 10].
    assert data["run_ids"] == [0, 1]
    assert data["iters"] == [0, 10]
