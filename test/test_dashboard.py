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
from jaxrens_al.dashboard.panel_specs import make_spec  # noqa: E402

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
        with_adaptation=True,
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
        "quests_pca",
        "resampled_grid",
        "eos",
        "neuralil_parity",
        "neuralil_training_curve",
        "ns_thermodynamics",
        "acceptance_rates",
        "step_sizes",
        "adaptation_acceptance",
        "ns_energy_trace",
        "re_swap_rates",
    }


# --------------------------------------------------------------------------
# Combination of ns/dft/train dashboards: the panels the standalone scripts
# show but which only register given the right artifacts.
# --------------------------------------------------------------------------
def test_sampling_parity_registers_on_iteration_with_proposing_model(
    make_calchome, tmp_path
):
    # sampling_parity re-scores the evaluated configs with the model that
    # PROPOSED them -- the previous iteration's recorded trainer output. So it
    # registers on iteration >= 1 and points at that model.
    calchome = make_calchome(tmp_path, n_iter=2, with_trainer_output=True)
    dp = tmp_path / "dp"
    dp.mkdir()
    ctx = iteration_context(calchome, dp, 1)
    assert ctx.sampling_model_file == calchome / "trainer_output" / "t0.bundle"
    (panel,) = [e for e in assemble(ctx) if e.name == "sampling_parity"]
    assert str(panel._model_file) == str(ctx.sampling_model_file)


def test_sampling_parity_absent_without_proposing_model(
    make_calchome, tmp_path
):
    # Iteration 0's proposing model (the foundation seed) is not recorded, so
    # there is no sampling_parity there; likewise when no trainer output exists.
    dp = tmp_path / "dp"
    dp.mkdir()
    calchome = make_calchome(tmp_path, n_iter=2, with_trainer_output=True)
    ctx0 = iteration_context(calchome, dp, 0)
    assert ctx0.sampling_model_file is None
    assert "sampling_parity" not in {e.name for e in assemble(ctx0)}

    plain = make_calchome(tmp_path / "b", n_iter=2)  # no trainer_output
    ctx1 = iteration_context(plain, dp, 1)
    assert ctx1.sampling_model_file is None
    assert "sampling_parity" not in {e.name for e in assemble(ctx1)}


def test_mace_training_curve_registers_when_log_present(
    make_calchome, tmp_path
):
    calchome = make_calchome(tmp_path, n_iter=2, with_mace_training=True)
    names = {e.name for e in build_dashboard(calchome, compute=False).entries}
    assert "mace_training_curve" in names


def test_model_parity_registers_for_mace_iteration(make_calchome, tmp_path):
    # A MACE-style iteration: a model bundle + evaluated configs but no .npz
    # -> the model-rescored parity panel registers (per-iteration only).
    calchome = make_calchome(
        tmp_path, n_iter=2, with_mace_training=True, with_model=True
    )
    dp = calchome / "dashboard" / "1"
    dp.mkdir(parents=True)
    board = build_iteration_dashboard(calchome, 1, datapath=dp, compute=False)
    assert "model_parity" in {e.name for e in board.entries}


def test_model_parity_absent_when_npz_present(make_calchome, tmp_path):
    # NeuralIL writes a .npz parity bundle, so the re-scoring panel is skipped
    # in favour of the cheaper neuralil_parity reader.
    calchome = make_calchome(
        tmp_path, n_iter=2, with_training=True, with_model=True
    )
    dp = calchome / "dashboard" / "1"
    dp.mkdir(parents=True)
    board = build_iteration_dashboard(calchome, 1, datapath=dp, compute=False)
    names = {e.name for e in board.entries}
    assert "model_parity" not in names and "neuralil_parity" in names


def test_model_parity_renders_via_seam(
    make_calchome, tmp_path, fake_model_parity_eval
):
    # Compute+plot the MACE model-parity panel end to end (model load faked).
    calchome = make_calchome(
        tmp_path, n_iter=2, with_mace_training=True, with_model=True
    )
    dp = calchome / "dashboard" / "1"
    dp.mkdir(parents=True)
    board = build_iteration_dashboard(calchome, 1, datapath=dp, compute=True)
    assert "model_parity" in {e.name for e in board.entries}
    assert (dp / "model_parity.json").exists()


def test_iteration_backend_autodetected_from_mace_logs(
    make_calchome, tmp_path
):
    # A MACE iteration (its *_train.txt log present) -> backend auto-detects
    # 'mace' with no eos_backend passed, so model_parity/EOS load correctly.
    calchome = make_calchome(
        tmp_path, n_iter=2, with_mace_training=True, with_model=True
    )
    ctx = iteration_context(calchome, tmp_path / "dp", 1)
    assert ctx.eos_backend == "mace"


def test_iteration_backend_autodetected_neuralil(make_calchome, tmp_path):
    # A NeuralIL iteration (no MACE logs, a .pkl model) -> 'neuralil'.
    calchome = make_calchome(
        tmp_path, n_iter=2, with_training=True, with_model=True
    )
    ctx = iteration_context(calchome, tmp_path / "dp", 1)
    assert ctx.eos_backend == "neuralil"


def test_iteration_backend_explicit_override_wins(make_calchome, tmp_path):
    # An explicit eos_backend overrides auto-detection.
    calchome = make_calchome(
        tmp_path, n_iter=2, with_mace_training=True, with_model=True
    )
    ctx = iteration_context(
        calchome, tmp_path / "dp", 1, eos_backend="neuralil"
    )
    assert ctx.eos_backend == "neuralil"


def test_summary_backend_autodetected_from_mace_logs(make_calchome, tmp_path):
    calchome = make_calchome(
        tmp_path, n_iter=2, with_mace_training=True, with_model=True
    )
    ctx = summary_context(calchome, tmp_path / "dp")
    assert ctx.eos_backend == "mace"


def test_summary_end_to_end_writes_pdf_and_caches(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2)
    board = build_dashboard(calchome)

    out = calchome / "dashboard.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0
    # Panel caches were written under calchome/dashboard.
    assert (calchome / "dashboard" / "sampled_energy_volume.json").exists()
    assert (calchome / "dashboard" / "dataset_growth.json").exists()


def test_recompute_refreshes_only_named_panels(make_calchome, tmp_path):
    import json

    calchome = make_calchome(tmp_path, n_iter=2)
    build_dashboard(calchome)  # first build writes the JSON caches
    dp = calchome / "dashboard"
    named = dp / "sampled_energy_volume.json"
    other = dp / "dataset_growth.json"
    assert named.exists() and other.exists()

    # Tamper both caches with a marker; a cache *hit* leaves the file as-is,
    # a recompute overwrites it.
    for f in (named, other):
        d = json.loads(f.read_text())
        d["_tampered"] = True
        f.write_text(json.dumps(d))

    # Recompute only sampled_energy_volume; dataset_growth stays cached.
    build_dashboard(calchome, recompute=["sampled_energy_volume"])
    assert "_tampered" not in json.loads(named.read_text())  # recomputed
    assert json.loads(other.read_text()).get("_tampered") is True  # cache hit


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
    # atomic_envs is opt-in now (needs a PCA reference), so it is not here.
    assert names == {
        "sampled_energy_volume",
        "neuralil_parity",
        "neuralil_training_curve",
    }
    assert "dataset_growth" not in names
    # Dropped from the curated set: a single-iteration page would show one box.
    assert "sampled_ns_energy" not in names


def test_sampled_ns_energy_is_opt_in(make_calchome, tmp_path):
    """Not a default, but an ``add:`` entry brings the box back."""
    calchome = make_calchome(tmp_path, n_iter=2, with_training=True)
    datapath = calchome / "dashboard" / "1"
    datapath.mkdir(parents=True)

    cfg = {"add": [make_spec("sampled_ns_energy")]}
    board = build_iteration_dashboard(
        calchome, 1, datapath=datapath, panel_config=cfg
    )
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
    fake_ns_traces,
    fake_acc_log,
    fake_adaptation_log,
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
        with_adaptation=True,
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


# --------------------------------------------------------------------------
# Selector stage: the selection map, and a second thermodynamic quantity.
# --------------------------------------------------------------------------
def test_selection_map_registers_only_with_selector_output(
    make_calchome, tmp_path
):
    """The panel is a curated default but self-gates on a selector stage."""
    plain = make_calchome(tmp_path / "plain", n_iter=1)
    dp = plain / "dashboard"
    dp.mkdir(parents=True, exist_ok=True)
    ctx = iteration_context(plain, dp, 0)
    assert "selection_map" not in {e.name for e in assemble(ctx)}

    staged = make_calchome(
        tmp_path / "staged", n_iter=1, n_selected=2, staged=True
    )
    dp2 = staged / "dashboard"
    dp2.mkdir(parents=True, exist_ok=True)
    ctx2 = iteration_context(staged, dp2, 0)
    assert ctx2.selected_files
    panels = {e.name: e for e in assemble(ctx2)}
    assert "selection_map" in panels

    # It reads the selected configs against the full sampler pool.
    data = panels["selection_map"].compute_data({})
    assert data["groups"][0]["iter"] == [0, 2]
    assert data["n_pool"] == 5  # the iteration's sampled configs


def test_selection_map_summary_groups_by_iteration(make_calchome, tmp_path):
    calchome = make_calchome(tmp_path, n_iter=2, n_selected=2, staged=True)
    dp = calchome / "dashboard"
    dp.mkdir(parents=True, exist_ok=True)
    ctx = summary_context(calchome, dp)
    panel = {e.name: e for e in assemble(ctx)}["selection_map"]
    data = panel.compute_data({})
    # One series per AL iteration; the pool background is dropped (it would
    # be every iteration's full NS pool).
    assert [g["label"] for g in data["groups"]] == ["0", "1"]
    assert data["n_pool"] == 0


def test_volume_thermodynamics_coexists_with_heat_capacity(
    make_calchome, tmp_path
):
    """A second ns_thermodynamics quantity gets its own (quantity-keyed) name,
    so it does not collide with the curated heat-capacity panel."""
    from jaxrens_al.dashboard.panel_specs import parse_panel_config

    calchome = make_calchome(tmp_path, n_iter=1, with_ns=True)
    dp = calchome / "dashboard"
    dp.mkdir(parents=True, exist_ok=True)
    ctx = iteration_context(calchome, dp, 0)
    cfg = parse_panel_config(
        {"add": [{"type": "ns_thermodynamics", "quantity": "volume"}]}
    )
    names = {e.name for e in assemble(ctx, panel_config=cfg)}
    assert {"ns_thermodynamics", "ns_volume"} <= names
