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
    MaceTrainingCurve,
    ModelParityPanel,
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


def test_model_parity_rescores_reference(tmp_path, write_configs):
    # ModelParityPanel loads no real model: patch the _evaluate seam so it
    # returns predicted (energy, forces) per config, then check it pairs them
    # against the REF_energy/REF_forces of the same configs.
    cfg = tmp_path / "db_val.extxyz"
    write_configs(
        cfg,
        4,
        lambda i: {"REF_energy": -9.0 + 0.1 * i},
        with_forces=True,
    )
    panel = ModelParityPanel(
        config_files=[cfg],
        model_file=tmp_path / "model.pkl",
        backend_type="mace",
        datapath=tmp_path,
    )
    # Predicted total = REF total + 0.3; configs are 3-atom, so per-atom the
    # prediction sits 0.1 eV/atom above the reference (a constant offset).
    panel._evaluate = lambda mf, frames: (
        [float(at.info["REF_energy"]) + 0.3 for at in frames],
        [np.asarray(at.arrays["REF_forces"]) + 0.02 for at in frames],
    )
    data = panel.compute_data({})
    assert isinstance(data, ParityData)
    assert len(data.e_obs) == 4 and len(data.e_pred) == 4
    assert data.e_pred[0] == pytest.approx(data.e_obs[0] + 0.1)
    assert data.f_obs is not None and len(data.f_obs) == 4 * 3 * 3
    assert data.e_rmse == pytest.approx(0.1, abs=1e-6)


def test_model_parity_energy_only(tmp_path, write_configs):
    cfg = tmp_path / "db_val.extxyz"
    write_configs(
        cfg, 3, lambda i: {"REF_energy": -9.0 + 0.1 * i}, with_forces=True
    )
    panel = ModelParityPanel(
        config_files=[cfg],
        model_file=tmp_path / "model.pkl",
        show_forces=False,
        datapath=tmp_path,
    )
    panel._evaluate = lambda mf, frames: (
        [float(at.info["REF_energy"]) * len(at) for at in frames],
        None,
    )
    data = panel.compute_data({})
    assert data.f_obs is None and data.f_rmse is None
    assert len(data.e_pred) == 3


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


def test_mace_training_curve_parses_eval_rows(tmp_path, write_mace_train_log):
    write_mace_train_log(tmp_path / "Model_train.txt")
    panel = MaceTrainingCurve(
        train_logs=[tmp_path / "Model_train.txt"], datapath=tmp_path
    )
    data = panel.compute_data({})
    # opt rows and the epoch=null baseline eval are skipped; two epochs remain.
    assert data["epoch"] == [0, 1]
    assert data["e_err"] == pytest.approx([12.0, 6.0])
    assert data["f_err"] == pytest.approx([2.0, 1.0])
    assert data["energy_metric"] == "rmse_e_per_atom"


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
# Thermodynamics panel (tested through the _load_monitors seam)
# --------------------------------------------------------------------------
class _FakeMonitor:
    """Stand-in for jaxrens Monitor with the two methods the panel calls."""

    def __init__(self, n_T):
        self.dead_volumes = np.ones(7)

    def heat_capacity(self, T, k_B=1.0):
        return np.abs(np.asarray(T, dtype=float)) + 1.0

    def expectation(self, obs, T, k_B=1.0):
        return np.asarray(T, dtype=float) * 0 + float(np.mean(obs))


def test_thermodynamics_panel_rows_per_replica(tmp_path):
    # Multi-run is the only loader: one heatmap row per replica, not per dir.
    panel = NSThermodynamicsHeatmap(
        result_dirs=[tmp_path / "run0"],
        quantity="heat_capacity",
        n_T=12,
        datapath=tmp_path,
    )
    panel._load_monitors = lambda d: [_FakeMonitor(12) for _ in range(3)]
    data = panel.compute_data({})
    assert isinstance(data, HeatmapData)
    assert len(data.t) == 12
    assert len(data.obs) == 3  # one row per replica
    assert len(data.obs[0]) == 12
    assert data.quantity == "heat_capacity"


def test_thermodynamics_panel_empty_dirs_raises(tmp_path):
    panel = NSThermodynamicsHeatmap(result_dirs=[], datapath=tmp_path)
    with pytest.raises(RuntimeError):
        panel.compute_data({})


def test_thermodynamics_swap_axes_orientation(tmp_path):
    import matplotlib.pyplot as plt
    from alviz.panels import HeatmapData

    data = HeatmapData(
        t=[1.0, 2.0, 3.0],
        p=[0.0, 1.0],
        obs=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        quantity="heat_capacity",
    )

    # Default: temperature on x, pressure on y.
    p0 = NSThermodynamicsHeatmap(result_dirs=["x"], datapath=tmp_path)
    p0.data = data
    fig, p0.ax = plt.subplots()
    p0.plot_data()
    assert "T" in p0.ax.get_xlabel() and "P" in p0.ax.get_ylabel()
    plt.close(fig)

    # Swapped: pressure on x, temperature on y.
    p1 = NSThermodynamicsHeatmap(
        result_dirs=["x"], datapath=tmp_path, swap_axes=True
    )
    p1.data = data
    fig, p1.ax = plt.subplots()
    p1.plot_data()
    assert "P" in p1.ax.get_xlabel() and "T" in p1.ax.get_ylabel()
    plt.close(fig)


# --------------------------------------------------------------------------
# Adaptation-trace panels + energy trace (tested through their seams)
# --------------------------------------------------------------------------
def test_step_size_panel_reads_adaptation(tmp_path, fake_adaptation_log):
    from jaxrens_al.panels.adaptation import StepSizePanel

    panel = StepSizePanel(adaptation_file="x.h5", datapath=tmp_path)
    data = panel.compute_data({})
    assert data["move_names"] == ["gmc", "swap"]
    assert data["iterations"] == [0, 10, 20]
    # (n_records, n_runs, n_moves).
    assert np.shape(data["step_sizes"]) == (3, 2, 2)
    assert data["n_runs"] == 2 and data["n_moves"] == 2


def test_adaptation_acceptance_panel_reads_adaptation(
    tmp_path, fake_adaptation_log
):
    from jaxrens_al.panels.adaptation import AdaptationAcceptancePanel

    panel = AdaptationAcceptancePanel(
        adaptation_file="x.h5", datapath=tmp_path
    )
    data = panel.compute_data({})
    assert np.shape(data["acceptance_rates"]) == (3, 2, 2)
    assert data["acceptance_rates"][0][0] == [0.5, 0.5]


def test_energy_trace_panel_per_replica(tmp_path, fake_ns_traces):
    from jaxrens_al.panels.traces import NSEnergyTracePanel

    panel = NSEnergyTracePanel(
        result_dirs=[tmp_path / "out"], prefix="ns", datapath=tmp_path
    )
    data = panel.compute_data({})
    assert len(data["traces"]) == 2  # one per replica
    tr = data["traces"][0]
    assert tr["label"] == "run0"
    assert len(tr["iterations"]) == len(tr["energies"]) == 20


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


def test_eos_panel_multi_phase(eos_reference, tmp_path):
    from jaxrens_al.panels.eos import EquationOfStatePanel, _birch_murnaghan

    # Two phases (same scan here) -> two fitted curves.
    ref_a = eos_reference(tmp_path / "phase_a.extxyz")
    ref_b = eos_reference(tmp_path / "phase_b.extxyz")
    panel = EquationOfStatePanel(
        references=[ref_a, ref_b], model_file="m.pkl", datapath=tmp_path
    )
    # Seam: model energies follow a (shifted) BM curve so the fit converges.
    panel._evaluate = lambda mf, al: [
        float(_birch_murnaghan(a.get_volume() / len(a), -3.9, 23.5, 0.55, 4.0))
        for a in al
    ]
    data = panel.compute_data({})
    assert data["labels"] == ["phase_a", "phase_b"]
    assert len(data["phases"]) == 2
    ph = data["phases"][0]
    assert len(ph["volumes"]) == 9 and len(ph["e_dft"]) == 9
    # Enthalpies are interpolated onto the shared pressure grid.
    assert len(ph["h_dft"]) == len(data["p_grid"])
    assert len(ph["h_nnff"]) == len(data["p_grid"])


def test_atomic_envs_panel_pca(
    make_calchome, tmp_path, fake_quests_descriptors
):
    from jaxrens_al.panels.environments import AtomicEnvsPanel

    calchome = make_calchome(tmp_path, n_iter=1)
    sampled = calchome / "sampler_output" / "s0.extxyz"
    panel = AtomicEnvsPanel(config_files=[sampled], datapath=tmp_path)
    data = panel.compute_data({})
    # 5 Cu2O configs -> 15 atoms -> 15 projected points, each with a species.
    assert len(data["pc1"]) == 15 and len(data["pc2"]) == 15
    assert set(data["species"]) == {"Cu", "O"}
    assert data["n_total"] == 5 and data["n_used"] == 5  # under the cap


def test_atomic_envs_panel_subsamples_large_pool(
    make_calchome, tmp_path, fake_quests_descriptors
):
    # The sampler is a pure producer: a big pool must be thinned before the
    # (expensive) descriptor step, not mapped in full.
    from jaxrens_al.panels.environments import AtomicEnvsPanel

    calchome = make_calchome(tmp_path, n_iter=1, n_sampled=40)
    sampled = calchome / "sampler_output" / "s0.extxyz"
    panel = AtomicEnvsPanel(
        config_files=[sampled], datapath=tmp_path, max_configs=10
    )
    data = panel.compute_data({})
    assert data["n_total"] == 40 and data["n_used"] == 10
    # 10 Cu2O configs kept -> 30 per-atom points.
    assert len(data["pc1"]) == 30


def _write_annotated(src, dest, l_values, seed=0):
    """Write a synthetic Steinhardt-annotated extxyz (test-only stand-in for
    the remote processor's output).

    Attaches the same per-atom ``q<l>``/``w<l>`` arrays + per-frame
    ``q<l>_mean``/``w<l>_mean`` layout the real
    :class:`jaxrens_al.processors.steinhardt.SteinhardtProcessor` produces, so
    the pure reader panels can be exercised with no jaxrens/cluster. ``q_l`` is
    drawn from a two-cluster mixture in [0, 1]; ``w_l`` is a small signed value.
    """
    import ase.io

    rng = np.random.default_rng(seed)
    frames = ase.io.read(str(src), index=":")
    for at in frames:
        n = len(at)
        for l in l_values:
            centers = rng.choice([0.2, 0.55], size=n)
            q = np.clip(centers + rng.normal(0, 0.05, n), 0.0, 1.0)
            w = rng.normal(-0.05, 0.03, n)
            at.new_array(f"q{l}", q)
            at.new_array(f"w{l}", w)
            at.info[f"q{l}_mean"] = float(q.mean())
            at.info[f"w{l}_mean"] = float(w.mean())
    ase.io.write(str(dest), frames)
    return dest


def test_steinhardt_distribution_panel_reads_annotation(
    make_calchome, tmp_path
):
    """Distribution panel reads an annotated artifact."""
    from jaxrens_al.panels.steinhardt import SteinhardtDistributionPanel

    calchome = make_calchome(tmp_path, n_iter=1)
    sampled = calchome / "sampler_output" / "s0.extxyz"
    annotated = _write_annotated(sampled, tmp_path / "ann.extxyz", (4, 6))
    panel = SteinhardtDistributionPanel(
        annotated_file=annotated,
        l_values=(4, 6),
        datapath=tmp_path,
    )
    data = panel.compute_data({})
    # 5 Cu2O configs -> 15 atoms -> one q/w value per atom per l.
    assert data["l_values"] == [4, 6]
    assert len(data["q"]["4"]) == 15 and len(data["w"]["6"]) == 15
    assert all(0.0 <= v <= 1.0 for v in data["q"]["6"])
    assert set(data["species"]) == {"Cu", "O"}


def test_steinhardt_heatmap_panel_reads_same_artifact(make_calchome, tmp_path):
    """The heatmap reads the SAME annotated file as the distribution panel.

    One processor run -> one annotated file -> handed to both panels: that is
    how a single computation is shared.
    """
    from alviz.panels import HeatmapData

    from jaxrens_al.panels.steinhardt import (
        SteinhardtDistributionPanel,
        SteinhardtHeatmapPanel,
    )

    calchome = make_calchome(tmp_path, n_iter=1)
    sampled = calchome / "sampler_output" / "s0.extxyz"
    annotated = _write_annotated(sampled, tmp_path / "ann.extxyz", (6,))

    dist = SteinhardtDistributionPanel(
        annotated_file=annotated, l_values=(6,), datapath=tmp_path
    )
    dist.compute_data({})

    heat = SteinhardtHeatmapPanel(
        annotated_file=annotated,
        l=6,
        x_key="volume_per_atom",
        y_key="ns_energy",
        n_bins=4,
        datapath=tmp_path,
    )
    data = heat.compute_data({})
    assert isinstance(data, HeatmapData)
    assert data.quantity == "<q6>"
    assert len(data.obs) == len(data.p) and len(data.obs[0]) == len(data.t)
    assert np.isfinite(np.asarray(data.obs, float)).any()


def test_steinhardt_distribution_joint_scatter_renders(
    make_calchome, tmp_path
):
    import matplotlib.pyplot as plt

    from jaxrens_al.panels.steinhardt import SteinhardtDistributionPanel

    calchome = make_calchome(tmp_path, n_iter=1)
    sampled = calchome / "sampler_output" / "s0.extxyz"
    annotated = _write_annotated(sampled, tmp_path / "ann.extxyz", (4, 6))
    panel = SteinhardtDistributionPanel(
        annotated_file=annotated,
        l_values=(4, 6),
        datapath=tmp_path,
    )
    panel.data = panel.compute_data({})

    fig = plt.figure()
    panel.subspec = fig.add_gridspec(1, 1)[0, 0]
    panel.allocate_panels(fig)
    panel.plot_data()
    assert panel.colspan == 1 and len(panel.axes) == 1
    assert "q_{4}" in panel.axes[0].get_xlabel()
    assert "q_{6}" in panel.axes[0].get_ylabel()
    plt.close(fig)


def test_steinhardt_panel_missing_array_errors(make_calchome, tmp_path):
    """A plain (un-annotated) trajectory trips the helpful read-back error."""
    from jaxrens_al.panels.steinhardt import SteinhardtDistributionPanel

    calchome = make_calchome(tmp_path, n_iter=1)
    sampled = calchome / "sampler_output" / "s0.extxyz"
    panel = SteinhardtDistributionPanel(
        annotated_file=sampled, l_values=(6,), datapath=tmp_path
    )
    with pytest.raises(ValueError, match="'q6'"):
        panel.compute_data({})


def test_move_acceptance_panel_reads_log(tmp_path, fake_acc_log):
    from jaxrens_al.panels.acceptance import MoveAcceptancePanel

    panel = MoveAcceptancePanel(acc_rates_file="x.h5", datapath=tmp_path)
    data = panel.compute_data({})
    assert data["move_names"] == ["gmc", "swap"]
    assert data["iterations"] == [0, 10, 20]
    # 3 iterations x 2 moves, averaged over the single run.
    assert np.shape(data["rates"]) == (3, 2)
    assert data["rates"][0] == [0.8, 0.5]


def test_re_swap_panel_reads_log(tmp_path, fake_re_log):
    from jaxrens_al.panels.acceptance import RESwapAcceptancePanel

    panel = RESwapAcceptancePanel(re_stats_file="x.h5", datapath=tmp_path)
    data = panel.compute_data({})
    assert data["n_pairs"] == 2
    assert data["iterations"] == [0, 100, 200]
    assert np.shape(data["rates"]) == (3, 2)
    assert data["flavor"] == "pressure"


def test_resampled_grid_panel_reads_selected(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=1, with_resampling=True)
    prov = (
        calchome / "0" / "ns_sampler" / "results" / "resampled_provenance.json"
    )
    panel = ResampledGridPanel(provenance_json=prov, datapath=tmp_path)
    data = panel.compute_data({})
    # 2 selected: run_ids [0, 1], iters [0, 10].
    assert data["run_ids"] == [0, 1]
    assert data["iters"] == [0, 10]
