"""Tests for the section-dashboard CLI subcommands (ns/dft/training).

The ``jaxrens-al dashboard`` CLI dispatches its ``ns`` / ``dft`` / ``training``
first-token subcommands to :mod:`jaxrens_al.dashboard.section_dashboards`, which the thin
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
    # No --prefix on the CLI; a config beside the input dir supplies it.
    (out / "dashboard.yaml").write_text("prefix: mace_sn16\n")
    pytest.importorskip("yaml")
    pytest.importorskip("alviz")

    rc = main(["ns", str(out), "--n-T", "3", "--force"])
    assert rc == 0
    # PDF defaults one directory above the NS output dir.
    assert (out.parent / "ns_dashboard.pdf").exists()
    assert not (out / "ns_dashboard.pdf").exists()


def test_ns_reads_config_from_parent_dir(tmp_path, capsys):
    """A config beside the *output* (the input dir's parent) is discovered."""
    out = _make_ns_dir(tmp_path, prefix="mace_sn16")
    # Config one level up, where the PDF lands -- not in the input dir itself.
    (out.parent / "dashboard.yaml").write_text("prefix: mace_sn16\n")
    pytest.importorskip("yaml")
    pytest.importorskip("alviz")

    rc = main(["ns", str(out), "--n-T", "3", "--force"])
    assert rc == 0
    assert (out.parent / "ns_dashboard.pdf").exists()


def test_ns_cli_prefix_overrides_config(tmp_path, capsys):
    out = _make_ns_dir(tmp_path, prefix="mace_sn16")
    (out / "dashboard.yaml").write_text("prefix: mace_sn16\n")
    pytest.importorskip("yaml")

    # CLI --prefix ns wins over the config's mace_sn16 -> no ns.run*.energies.
    rc = main(["ns", str(out), "--prefix", "ns"])
    assert rc == 1
    assert "ns.run*.energies" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# init-config: scaffold a starter config from each parser flavor.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flavor", ["al", "ns", "dft", "training"])
def test_init_config_stdout_round_trips(flavor, tmp_path, capsys):
    pytest.importorskip("yaml")
    from jaxrens_al.cli_args import load_config

    rc = main(["init-config", flavor])
    assert rc == 0
    text = capsys.readouterr().out
    assert text.startswith("#")  # header comment

    p = tmp_path / "dashboard.yaml"
    p.write_text(text)
    cfg = load_config(p)
    # Every dumped key is a real option of that flavor's parser (drift guard),
    # so apply_config would accept the file without warning.
    from jaxrens_al.cli import _build_parser
    from jaxrens_al.dashboard.section_dashboards import (
        build_dft_parser,
        build_ns_parser,
        build_training_parser,
    )

    parser = {
        "al": _build_parser,
        "ns": build_ns_parser,
        "dft": build_dft_parser,
        "training": build_training_parser,
    }[flavor]()
    assert set(cfg) <= {a.dest for a in parser._actions}


def test_init_config_writes_file_and_guards_overwrite(tmp_path, capsys):
    dest = tmp_path / "dashboard.yaml"
    assert main(["init-config", "ns", "-o", str(dest)]) == 0
    assert dest.is_file() and dest.read_text().startswith("#")

    # Existing file + no --force -> error, original untouched.
    original = dest.read_text()
    rc = main(["init-config", "ns", "-o", str(dest)])
    assert rc == 1
    assert dest.read_text() == original
    # --force overwrites.
    assert main(["init-config", "ns", "-o", str(dest), "--force"]) == 0


def test_init_config_bad_flavor_errors():
    with pytest.raises(SystemExit):  # argparse choices rejects it
        main(["init-config", "bogus"])
