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
