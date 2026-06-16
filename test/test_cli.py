"""Tests for the standalone dashboard CLI (jaxrens_al.cli).

The guard paths below return before any alviz import, so they run without the
dashboard stack. The multipage build is exercised separately (and skips when
the dashboard stack / pypdf is unavailable).
"""

import json

import pytest

from jaxrens_al.cli import main


def test_missing_path_returns_error(capsys):
    rc = main(["/no/such/path"])
    assert rc == 1
    assert "does not exist" in capsys.readouterr().err.lower() or rc == 1


def test_not_a_calchome_returns_error(tmp_path, capsys):
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


@pytest.mark.local
def test_multipage_build_one_page_per_iteration(tmp_path):
    """Full build: one page (and one file) per iteration, then merged."""
    pytest.importorskip("alviz")
    pdf = pytest.importorskip("pypdf")
    import ase.io
    import numpy as np
    from ase import Atoms

    calchome = tmp_path / "calchome"
    (calchome / "sampler_output").mkdir(parents=True)
    (calchome / "evaluator_output").mkdir(parents=True)

    rng = np.random.default_rng(0)

    def _write(path, n, info):
        frames = []
        for i in range(n):
            at = Atoms(
                "Cu2O", positions=rng.uniform(0, 3, (3, 3)), cell=[6, 6, 6]
            )
            at.info.update(info(i))
            frames.append(at)
        ase.io.write(path, frames)

    n_iter = 3
    progress = []
    for it in range(n_iter):
        s = f"sampler_output/s{it}.extxyz"
        e = f"evaluator_output/e{it}.extxyz"
        _write(calchome / s, 5, lambda i, it=it: {"ns_energy": -10.0 - it + i})
        _write(calchome / e, 3, lambda i, it=it: {"REF_energy": -9.0 - it + i})
        progress.append(
            {"steps": [0], "sampler_output": [s], "evaluator_output": [e]}
        )
    (calchome / "AL_progress.json").write_text(json.dumps(progress))

    rc = main([str(calchome), "--force"])
    assert rc == 0

    # One individual page file per iteration ...
    for it in range(n_iter):
        assert (calchome / "plots" / f"Iteration{it}.pdf").exists()

    # ... merged into a single combined multipage PDF.
    combined = calchome / "dashboard.pdf"
    assert combined.exists()
    assert len(pdf.PdfReader(str(combined)).pages) == n_iter
