"""Tests for the section-dashboard CLI subcommands (ns/dft/training).

The ``jaxrens-al-dashboard`` CLI dispatches its ``ns`` / ``dft`` / ``training``
first-token subcommands to :mod:`jaxrens_al.section_dashboards`, which the thin
``scripts/{ns,dft,training}_dashboard.py`` wrappers also call. These tests
cover the dispatch + guard paths (which return before any alviz import, so they
run without the dashboard stack) and the config-file mechanism; a single render
is exercised to prove the whole path wires up.
"""

import pytest

from jaxrens_al.cli import main

pytestmark = pytest.mark.local


def _make_ns_dir(tmp_path, *, prefix="mace_sn16", n_runs=2):
    """A minimal multi-run NS output dir (stub files; discovery only)."""
    d = tmp_path / "output"
    d.mkdir()
    for r in range(n_runs):
        (d / f"{prefix}.run{r}.energies").write_text("# stub\n")
        (d / f"{prefix}.run{r}.traj.extxyz").write_text("")
    return d


# ---------------------------------------------------------------------------
# Dispatch + guard paths (no alviz import needed -- return early).
# ---------------------------------------------------------------------------
def test_ns_missing_dir_returns_error(capsys):
    rc = main(["ns", "/no/such/dir"])
    assert rc == 1
    assert "not a directory" in capsys.readouterr().err.lower()


def test_ns_no_energies_returns_error(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    rc = main(["ns", str(tmp_path / "empty")])
    assert rc == 1
    assert "run*.energies" in capsys.readouterr().err


def test_dft_missing_file_returns_error(tmp_path, capsys):
    rc = main(["dft", str(tmp_path / "nope.extxyz")])
    assert rc == 1
    assert "no such file" in capsys.readouterr().err.lower()


def test_training_missing_dir_returns_error(capsys):
    rc = main(["training", "/no/such/dir"])
    assert rc == 1
    assert "not a directory" in capsys.readouterr().err.lower()


def test_unknown_first_token_is_treated_as_calchome(tmp_path):
    # A path that is not a known subcommand falls through to the AL flow, which
    # errors on the missing AL_progress.json rather than being a "subcommand".
    rc = main([str(tmp_path)])  # dir exists, no AL_progress.json
    assert rc == 1


# ---------------------------------------------------------------------------
# Config file: auto-discovery beside the input + CLI-flag precedence.
# ---------------------------------------------------------------------------
def test_ns_reads_prefix_from_autodiscovered_config(tmp_path, capsys):
    out = _make_ns_dir(tmp_path, prefix="mace_sn16")
    # No --prefix on the CLI; the config beside the dir supplies it. Without it
    # the default prefix 'ns' would find no energies -> error.
    (out / "dashboard.yaml").write_text("prefix: mace_sn16\n")
    pytest.importorskip("yaml")
    pytest.importorskip("alviz")

    rc = main(["ns", str(out), "--n-T", "3", "--force"])
    assert rc == 0
    assert (out / "ns_dashboard.pdf").exists()


def test_ns_cli_prefix_overrides_config(tmp_path, capsys):
    out = _make_ns_dir(tmp_path, prefix="mace_sn16")
    (out / "dashboard.yaml").write_text("prefix: mace_sn16\n")
    pytest.importorskip("yaml")

    # CLI --prefix ns wins over the config's mace_sn16 -> no ns.run*.energies.
    rc = main(["ns", str(out), "--prefix", "ns"])
    assert rc == 1
    assert "ns.run*.energies" in capsys.readouterr().err
