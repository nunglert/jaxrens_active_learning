"""Tests for dashboard *assembly* (jaxrens_al.dashboard).

This layer decides which panels appear (presence-driven, from a RunContext),
in the summary vs per-iteration view, lays them out, and supports name-based
selection. Individual panel computation lives in test_panels.py; here panels
are treated as opaque -- assertions are about the assembled set, sizing, and a
saved PDF, not about plotted numbers.
"""

import pytest

pytest.importorskip("alviz")
pytest.importorskip("ase")

from jaxrens_al.dashboard import (  # noqa: E402
    assemble,
    build_dashboard,
    build_iteration_dashboard,
    iteration_context,
    summary_context,
)

pytestmark = pytest.mark.local


# --------------------------------------------------------------------------
# Summary view: panels registered only when their inputs exist
# --------------------------------------------------------------------------
def test_summary_assembles_present_panels(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2, with_training=True)
    board = build_dashboard(calchome, compute=True)

    names = {e.name for e in board.entries}
    # Present inputs -> these registered; no NS .energies -> no thermo.
    assert {
        "dataset_growth",
        "sampled_energy_volume",
        "sampled_ns_energy",
        "neuralil_parity",
        "neuralil_training_curve",
    } <= names
    assert "ns_thermodynamics" not in names


def test_summary_assembles_all_panels(make_calchome, tmp_path):
    # Every artifact present -> every summary-eligible panel registers.
    calchome = make_calchome(
        tmp_path,
        n_iter=2,
        with_training=True,
        with_resampling=True,
        with_ns=True,
    )
    board = build_dashboard(calchome, compute=False)
    names = {e.name for e in board.entries}
    # quests_pca / resampled_grid are per-iteration views, never in summary.
    assert names == {
        "dataset_growth",
        "sampled_energy_volume",
        "sampled_ns_energy",
        "neuralil_parity",
        "neuralil_training_curve",
        "ns_thermodynamics",
    }


def test_iteration_assembles_all_panels(
    make_calchome, eos_reference, tmp_path
):
    calchome = make_calchome(
        tmp_path,
        n_iter=2,
        with_training=True,
        with_resampling=True,
        with_ns=True,
        with_acc_rates=True,
        with_re_stats=True,
        with_model=True,
    )
    ref = eos_reference(tmp_path / "eos.extxyz")
    dp = calchome / "dashboard" / "1"
    dp.mkdir(parents=True)
    board = build_iteration_dashboard(
        calchome, 1, datapath=dp, compute=False, eos_reference=[ref]
    )
    names = {e.name for e in board.entries}
    # dataset_growth is cumulative (summary-only); everything else appears.
    assert names == {
        "sampled_energy_volume",
        "sampled_ns_energy",
        "quests_pca",
        "resampled_grid",
        "atomic_envs",
        "eos",
        "neuralil_parity",
        "neuralil_training_curve",
        "ns_thermodynamics",
        "acceptance_rates",
        "re_swap_rates",
    }


def test_summary_end_to_end_writes_pdf_and_caches(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2)
    board = build_dashboard(calchome)

    out = calchome / "dashboard.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0
    # Panel caches were written under calchome/dashboard.
    assert (calchome / "dashboard" / "sampled_energy_volume.json").exists()
    assert (calchome / "dashboard" / "dataset_growth.json").exists()


def test_grid_auto_sizes_to_hold_all_panels(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2, with_training=True)
    board = build_dashboard(calchome, ncols=2, compute=False)
    # Panels may span multiple cells (colspan/rowspan), so assert the grid is
    # two columns wide and has enough cells for their total area. Exact packing
    # is alviz's concern (tested there); here we just check it fits.
    total_cells = sum(
        getattr(e, "rowspan", 1) * getattr(e, "colspan", 1)
        for e in board.entries
    )
    assert board._ncols == 2
    assert board._nrows >= 1
    assert board._nrows * board._ncols >= total_cells


# --------------------------------------------------------------------------
# Per-iteration view
# --------------------------------------------------------------------------
def test_iteration_assembles_present_panels(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2, with_training=True)
    datapath = calchome / "dashboard" / "1"
    datapath.mkdir(parents=True)

    board = build_iteration_dashboard(calchome, 1, datapath=datapath)
    names = {e.name for e in board.entries}
    # Iteration 1 has evaluated + sampled + training, no NS .energies. Dataset
    # growth is cumulative (summary-only), so it is absent per-iteration.
    # atomic_envs registers whenever sampled configs exist.
    assert names == {
        "sampled_energy_volume",
        "sampled_ns_energy",
        "atomic_envs",
        "neuralil_parity",
        "neuralil_training_curve",
    }
    assert "dataset_growth" not in names

    box = next(e for e in board.entries if e.name == "sampled_ns_energy")
    assert box.data["iterations"] == [1]  # box positioned at this iteration


def test_iteration_includes_quests_panels_when_present(
    make_calchome, tmp_path
):
    calchome = make_calchome(tmp_path, n_iter=1, with_resampling=True)
    datapath = calchome / "dashboard" / "0"
    datapath.mkdir(parents=True)
    board = build_iteration_dashboard(calchome, 0, datapath=datapath)
    names = {e.name for e in board.entries}
    assert {"quests_pca", "resampled_grid"} <= names


def test_summary_excludes_quests_panels(make_calchome, tmp_path):
    # The QUESTS panels are per-iteration (per RENS run); never in the summary.
    calchome = make_calchome(tmp_path, n_iter=1, with_resampling=True)
    board = build_dashboard(calchome, compute=False)
    names = {e.name for e in board.entries}
    assert "quests_pca" not in names and "resampled_grid" not in names


def test_iteration_with_no_inputs_returns_none(make_calchome, tmp_path):
    # An iteration that produced nothing -> no page.
    calchome = make_calchome(tmp_path, n_iter=1, n_evaluated=0, n_sampled=0)
    datapath = calchome / "dashboard" / "0"
    datapath.mkdir(parents=True)
    assert build_iteration_dashboard(calchome, 0, datapath=datapath) is None


# --------------------------------------------------------------------------
# assemble(): the project's panel set + order. Name-based selection itself is
# alviz's job (tested there); here we only check --only/--exclude passthrough.
# --------------------------------------------------------------------------
def test_assemble_summary_panel_order(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2, with_training=True)
    datapath = calchome / "dashboard"
    datapath.mkdir(parents=True)
    names = [e.name for e in assemble(summary_context(calchome, datapath))]
    # Stable registration order, growth first; no NS .energies -> no thermo.
    assert names == [
        "dataset_growth",
        "sampled_energy_volume",
        "sampled_ns_energy",
        "neuralil_parity",
        "neuralil_training_curve",
    ]


def test_build_dashboard_exclude_kwarg(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2)
    board = build_dashboard(calchome, compute=True, exclude=["dataset_growth"])
    names = {e.name for e in board.entries}
    assert "dataset_growth" not in names
    assert "sampled_energy_volume" in names


def test_context_factories_set_view_flag(make_calchome, tmp_path):
    # Guard on the seam: the two factories set the summary/iteration flag.
    calchome = make_calchome(tmp_path, n_iter=2)
    assert summary_context(calchome, tmp_path).is_summary
    assert not iteration_context(calchome, tmp_path, 1).is_summary


# --------------------------------------------------------------------------
# Layout render for visual inspection.
#
# Layout (grid balance, titles, axis overlap) is invariant to the synthetic
# data and can't be asserted meaningfully -- so render the fullest summary and
# one per-iteration page to the git-ignored test/data/temp for eyeballing. The
# assertions only confirm a non-empty PDF was produced; *look at* the files.
# --------------------------------------------------------------------------
def test_render_layouts_for_inspection(
    make_calchome,
    eos_reference,
    artifact_dir,
    tmp_path,
    fake_ns_monitor,
    fake_acc_log,
    fake_re_log,
    fake_quests_descriptors,
    fake_eos_eval,
):
    # Every artifact present (+ fakes for the panels whose compute needs
    # jaxrens), so the rendered PDFs exercise every panel.
    calchome = make_calchome(
        tmp_path,
        n_iter=2,
        with_training=True,
        with_resampling=True,
        with_ns=True,
        with_acc_rates=True,
        with_re_stats=True,
        with_model=True,
    )
    ref = eos_reference(tmp_path / "eos.extxyz")

    summary = build_dashboard(calchome)
    summary_pdf = artifact_dir / "summary.pdf"
    summary.savefig(summary_pdf)

    it_datapath = tmp_path / "it1"
    it_datapath.mkdir()
    page = build_iteration_dashboard(
        calchome, 1, datapath=it_datapath, eos_reference=[ref]
    )
    iteration_pdf = artifact_dir / "iteration_1.pdf"
    page.savefig(iteration_pdf)

    for pdf in (summary_pdf, iteration_pdf):
        assert pdf.exists() and pdf.stat().st_size > 0
