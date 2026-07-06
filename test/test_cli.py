"""Tests for the standalone dashboard CLI (jaxrens_al.cli).

CLI concerns only: exit-code guard paths, the skip-if-exists behaviour, and the
multipage build (one page + one file per iteration, merged). The guard paths
return before any alviz import, so they run without the dashboard stack; the
multipage test imports it (and pypdf) and self-skips if unavailable.
"""

import json

import pytest

from jaxrens_al.cli import main

pytestmark = pytest.mark.local


def test_missing_path_returns_error(capsys):
    rc = main(["/no/such/path"])
    assert rc == 1
    assert "does not exist" in capsys.readouterr().err.lower() or rc == 1


def test_not_a_calchome_returns_error(tmp_path):
    # Dir exists but has no AL_progress.json.
    rc = main([str(tmp_path)])
    assert rc == 1


def test_existing_output_skips_without_force(tmp_path):
    (tmp_path / "AL_progress.json").write_text(json.dumps([]))
    out = tmp_path / "dashboard.pdf"
    out.write_bytes(b"%PDF-stub")
    # Output exists + no --force -> returns 0 without importing/plotting.
    rc = main([str(tmp_path)])
    assert rc == 0
    assert out.read_bytes() == b"%PDF-stub"  # untouched


def test_multipage_build_one_page_per_iteration(make_calchome, tmp_path):
    """Full build: one page (and one file) per iteration, then merged."""
    pytest.importorskip("alviz")
    pdf = pytest.importorskip("pypdf")

    calchome = make_calchome(tmp_path, n_iter=3)

    rc = main([str(calchome), "--force"])
    assert rc == 0

    # One individual page file per iteration ...
    for it in range(3):
        assert (calchome / "plots" / f"Iteration{it}.pdf").exists()

    # ... merged into a single combined multipage PDF.
    combined = calchome / "dashboard.pdf"
    assert combined.exists()
    assert len(pdf.PdfReader(str(combined)).pages) == 3


def test_multipage_build_reads_staged_schema(make_calchome, tmp_path):
    """The CLI builds the same multipage report from the staged schema."""
    pytest.importorskip("alviz")
    pdf = pytest.importorskip("pypdf")

    calchome = make_calchome(tmp_path, n_iter=3, staged=True)

    rc = main([str(calchome), "--force"])
    assert rc == 0
    for it in range(3):
        assert (calchome / "plots" / f"Iteration{it}.pdf").exists()
    combined = calchome / "dashboard.pdf"
    assert len(pdf.PdfReader(str(combined)).pages) == 3


def test_only_flag_restricts_panels(make_calchome, tmp_path):
    """--only keeps just the named panel; an empty iteration is skipped."""
    pytest.importorskip("alviz")
    pdf = pytest.importorskip("pypdf")

    # Iteration 0 has no sampler output, so --only sampled_ns_energy leaves it
    # empty -> 2 pages (iters 1, 2), not 3.
    calchome = make_calchome(tmp_path, n_iter=3, n_sampled=[0, 5, 5])

    rc = main([str(calchome), "--force", "--only", "sampled_ns_energy"])
    assert rc == 0
    assert not (calchome / "plots" / "Iteration0.pdf").exists()
    assert len(pdf.PdfReader(str(calchome / "dashboard.pdf")).pages) == 2


def test_sections_flag_emits_per_iteration_section_pages(
    make_calchome, tmp_path
):
    """--sections adds raw-dir NS/DFT/training pages alongside the AL pages."""
    pytest.importorskip("alviz")
    pdf = pytest.importorskip("pypdf")

    # Training artifacts on disk per iteration -> a Training section page;
    # evaluated configs -> a DFT section page. (NS needs run*.energies, which
    # this synthetic run doesn't write, so no NS section here.)
    calchome = make_calchome(tmp_path, n_iter=2, with_training=True)

    rc = main([str(calchome), "--force", "--sections"])
    assert rc == 0

    plots = calchome / "plots"
    for it in range(2):
        # Only the AL iteration page persists in plots/ ...
        assert (plots / f"Iteration{it}.pdf").exists()
        # ... section pages are ephemeral (merged, not kept on disk).
        assert not (plots / f"Iteration{it}_Training.pdf").exists()
        assert not (plots / f"Iteration{it}_DFT.pdf").exists()

    # ... but they ARE in the combined PDF: (AL + Training + DFT) x 2 = 6.
    combined = calchome / "dashboard.pdf"
    assert len(pdf.PdfReader(str(combined)).pages) == 6


def test_sections_flag_renders_unrecorded_iteration(make_calchome, tmp_path):
    """--sections renders an iteration dir AL_progress.json never recorded.

    The crashed-loop case the standalone scripts exist for: iteration 2's
    training results are on disk but not in the progress file (which stops at
    iteration 1). The AL pages ignore it; the section pass still renders it.
    """
    pytest.importorskip("alviz")
    pdf = pytest.importorskip("pypdf")

    calchome = make_calchome(tmp_path, n_iter=2, with_training=True)
    # An unrecorded iteration 2 with training artifacts on disk.
    res = calchome / "2" / "train" / "results"
    res.mkdir(parents=True)
    (res / "error_0.txt").write_text(
        "i_epoch=0\nModel #1:\nF_RMSE = 0.5 eV/A. \nE_RMSE = 0.2 eV. \n"
    )

    # Without --sections: iters 0,1 -> AL + Training + DFT = 6 pages; the
    # unrecorded iter 2 contributes nothing.
    rc = main([str(calchome), "--force", "--sections"])
    assert rc == 0

    plots = calchome / "plots"
    # No AL page for iter 2 (not in AL_progress.json), and section pages never
    # persist to plots/ ...
    assert not (plots / "Iteration2.pdf").exists()
    assert not (plots / "Iteration2_Training.pdf").exists()
    # ... but iter 2's section IS merged into the combined PDF: the two
    # recorded iters give 6 pages, iter 2 adds its Training section -> 7.
    combined = calchome / "dashboard.pdf"
    assert len(pdf.PdfReader(str(combined)).pages) == 7


def test_sections_autodetects_mace_prefix_and_train_dir(
    make_calchome, tmp_path, caplog
):
    """--sections finds NS/training pages for a MACE-style staged run.

    The NS files use a 'mace_sn16' prefix (not 'ns') and the trainer dir is
    'train' (not 'training') -- both are auto-resolved, so the sections are
    discovered without any --prefix flag.
    """
    pytest.importorskip("alviz")

    calchome = make_calchome(tmp_path, n_iter=1, with_training=False)
    it = calchome / "0"
    out = it / "ns_sampler" / "results" / "output"
    out.mkdir(parents=True)
    for r in (0, 1):
        (out / f"mace_sn16.run0{r}.energies").write_text("# stub\n")
        (out / f"mace_sn16.run0{r}.traj.extxyz").write_text("")
    res = it / "train" / "results"
    res.mkdir(parents=True)
    (res / "Model_run-1_train.txt").write_text(
        '{"loss": 1.0, "rmse_e_per_atom": 6.0, "rmse_f": 1.0, '
        '"mode": "eval", "epoch": 0, "head": "Default"}\n'
    )

    import logging

    with caplog.at_level(logging.INFO, logger="jaxrens_al.cli"):
        rc = main([str(calchome), "--dry-run", "--sections"])
    assert rc == 0
    log = caplog.text
    assert "'mace_sn16' (auto-detected)" in log
    assert "iteration 0 -- NS" in log
    assert "iteration 0 -- Training" in log


def test_log_file_writes_run_log(make_calchome, tmp_path):
    """--log-file tees logs to <calchome>/dashboard.log (bare form)."""
    pytest.importorskip("alviz")
    pytest.importorskip("pypdf")

    calchome = make_calchome(tmp_path, n_iter=1)
    rc = main([str(calchome), "--force", "--log-file"])
    assert rc == 0

    log = calchome / "dashboard.log"
    assert log.exists()
    text = log.read_text()
    assert "Wrote combined" in text  # the full render was captured
    # No leaked handler after the run resolves (idempotent clear on next call).
    main([str(calchome), "--force"])  # a run without --log-file
    import logging

    root = logging.getLogger()
    assert not any(
        getattr(h, "_jaxrens_dashboard_log", False) for h in root.handlers
    )
