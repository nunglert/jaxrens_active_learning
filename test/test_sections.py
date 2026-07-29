"""Tests for the shared raw-directory section builders (jaxrens_al.dashboard.sections).

These are the single source of truth for the per-artifact panel wiring the
standalone ``scripts/{ns,dft,training}_dashboard.py`` and the global CLI's
``--sections`` mode both use. The tests assert *which panels a section wires*
for a given on-disk artifact set (by ``.name``) -- construction only, no
render -- so they need no jaxrens: an empty stub file is enough to be
"present". Rendering is covered by the panel/CLI suites.
"""

import numpy as np
import pytest

from jaxrens_al.dashboard.sections import (
    dft_section,
    ns_section,
    training_section,
)

pytestmark = pytest.mark.local


def _names(entries):
    return {getattr(e, "name", None) for e in entries}


def _make_ns_dir(tmp_path, *, prefix="ns", n_runs=2, trajs=True):
    """A minimal multi-run NS output dir (stub files; discovery only)."""
    d = tmp_path / "output"
    d.mkdir()
    for r in range(n_runs):
        (d / f"{prefix}.run{r}.energies").write_text("# stub\n")
        if trajs:
            (d / f"{prefix}.run{r}.traj.extxyz").write_text("")
    return d


# ---------------------------------------------------------------------------
# ns_section
# ---------------------------------------------------------------------------
def test_ns_section_empty_without_energies(tmp_path):
    (tmp_path / "output").mkdir()
    assert ns_section(tmp_path / "output", datapath=tmp_path) == []


def test_ns_section_core_panels(tmp_path):
    d = _make_ns_dir(tmp_path)
    entries = ns_section(
        d, temperatures=np.linspace(0.1, 1.0, 5), datapath=tmp_path
    )
    names = _names(entries)
    # Always present when energies exist.
    assert {"ns_energy_trace", "ns_thermodynamics"} <= names
    # sampled_ns_energy is opt-in: one NS dir is a single box, so it is not
    # part of the curated NS set.
    assert "sampled_ns_energy" not in names
    # No adaptation/re_stats stubs -> those panels absent.
    assert "step_sizes" not in names
    assert "re_swap_rates" not in names
    # Steinhardt is opt-in.
    assert "steinhardt_expectation" not in names


def test_ns_section_discovers_optional_logs(tmp_path):
    d = _make_ns_dir(tmp_path)
    (d / "ns.adaptation.h5").write_bytes(b"")
    (d / "ns.re_stats.h5").write_bytes(b"")
    names = _names(ns_section(d, datapath=tmp_path))
    assert {"step_sizes", "adaptation_acceptance", "re_swap_rates"} <= names


def test_ns_section_no_steinhardt_by_default(tmp_path):
    # Steinhardt is opt-in via a config `add:` -- never in the curated set.
    d = _make_ns_dir(tmp_path)
    entries = ns_section(d, datapath=tmp_path)
    assert not any(n.startswith("steinhardt") for n in _names(entries))


def test_ns_section_steinhardt_opt_in_via_config(tmp_path):
    from jaxrens_al.dashboard.panel_specs import SteinhardtSpec

    d = _make_ns_dir(tmp_path)
    entries = ns_section(
        d,
        datapath=tmp_path,
        panel_config={"add": [SteinhardtSpec(r_cut=3.5, l=6)]},
    )
    # Param-keyed name (so several Steinhardts can coexist).
    assert "steinhardt_q6_rc3.5" in _names(entries)


def test_ns_section_multiple_steinhardts_via_config(tmp_path):
    from jaxrens_al.dashboard.panel_specs import SteinhardtSpec

    d = _make_ns_dir(tmp_path)
    entries = ns_section(
        d,
        datapath=tmp_path,
        panel_config={
            "add": [
                SteinhardtSpec(r_cut=3.5, l=6),
                SteinhardtSpec(r_cut=3.5, l=4),
            ]
        },
    )
    names = _names(entries)
    assert "steinhardt_q6_rc3.5" in names and "steinhardt_q4_rc3.5" in names


# ---------------------------------------------------------------------------
# dft_section
# ---------------------------------------------------------------------------
def test_dft_section_basic(tmp_path, write_configs):
    f = write_configs(
        tmp_path / "e.extxyz",
        4,
        lambda i: {"REF_energy": -9.0 + 0.1 * i},
        with_forces=True,
    )
    names = _names(dft_section([f], datapath=tmp_path))
    assert names == {"sampled_energy_volume", "atomic_envs"}


def test_dft_section_adds_surrogate_parity(tmp_path, write_configs):
    f = write_configs(
        tmp_path / "e.extxyz",
        4,
        lambda i: {"REF_energy": -9.0 + 0.1 * i, "ns_energy": -8.9 + 0.1 * i},
        with_forces=True,
    )
    names = _names(dft_section([f], datapath=tmp_path))
    assert {"parity", "energy_error"} <= names


def test_dft_section_surrogate_suppressed(tmp_path, write_configs):
    f = write_configs(
        tmp_path / "e.extxyz",
        4,
        lambda i: {"REF_energy": -9.0 + 0.1 * i, "ns_energy": -8.9 + 0.1 * i},
        with_forces=True,
    )
    names = _names(dft_section([f], pred_energy_key=None, datapath=tmp_path))
    assert "parity" not in names and "energy_error" not in names


def test_dft_section_empty_files(tmp_path):
    assert dft_section([], datapath=tmp_path) == []


def test_dft_section_panel_config_disable(tmp_path, write_configs):
    # dft is on the registry now, so panel_config layers like ns/AL.
    f = write_configs(
        tmp_path / "e.extxyz",
        4,
        lambda i: {"REF_energy": -9.0 + 0.1 * i},
        with_forces=True,
    )
    names = _names(
        dft_section(
            [f], datapath=tmp_path, panel_config={"disable": ["atomic_envs"]}
        )
    )
    assert names == {"sampled_energy_volume"}


# ---------------------------------------------------------------------------
# training_section
# ---------------------------------------------------------------------------
def test_training_section_neuralil(tmp_path, write_npz, write_error_log):
    res = tmp_path / "results"
    res.mkdir()
    write_npz(res / "val.npz")
    write_error_log(res / "error_0.txt")
    names = _names(training_section(res, datapath=tmp_path))
    # NeuralIL: curve + parity from the .npz (no model re-score needed).
    assert names == {"neuralil_training_curve", "neuralil_parity"}


def test_training_section_mace_rescore(
    tmp_path, write_mace_train_log, write_configs
):
    res = tmp_path / "results"
    res.mkdir()
    write_mace_train_log(res / "mace_run-0_train.txt")
    # A model bundle to re-score + a labelled split to score it on (MACE
    # writes no parity .npz, so the section re-scores a db_val set).
    (res / "model-jax.pkl").write_bytes(b"")
    write_configs(
        res / "db_val.extxyz",
        4,
        lambda i: {"REF_energy": -9.0 + 0.1 * i},
        with_forces=True,
    )
    names = _names(training_section(res, datapath=tmp_path))
    assert {"mace_training_curve", "model_parity"} <= names


def test_training_section_empty_without_logs(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    assert training_section(res, datapath=tmp_path) == []


def test_training_section_panel_config_disable(
    tmp_path, write_npz, write_error_log, write_configs
):
    # training is on the registry now, so panel_config layers like ns/dft/AL.
    res = tmp_path / "results"
    res.mkdir()
    write_npz(res / "val.npz")
    write_error_log(res / "error_0.txt")
    write_configs(
        res / "db_train.extxyz",
        4,
        lambda i: {"REF_energy": -9.0 + 0.1 * i},
        with_forces=True,
    )
    names = _names(
        training_section(
            res,
            datapath=tmp_path,
            panel_config={"disable": ["neuralil_parity", "atomic_envs"]},
        )
    )
    assert "neuralil_parity" not in names and "atomic_envs" not in names
    # The rest of the curated training set still resolves.
    assert {"neuralil_training_curve", "sampled_energy_volume"} <= names
