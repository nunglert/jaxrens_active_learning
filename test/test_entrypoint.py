"""Tests for the top-level ``jaxrens-al`` dispatcher (jaxrens_al.__main__).

Routing only -- that ``--version`` reports the package version and that the
``dashboard`` / ``refdb`` subcommands hand off to the right ``main``. The
handoffs are stubbed, so no dashboard/refdb machinery is imported here.
"""

import pytest

import jaxrens_al
from jaxrens_al.__main__ import main

pytestmark = pytest.mark.local


def test_version_prints_package_version(capsys):
    rc = main(["--version"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == jaxrens_al.__version__


def test_no_args_shows_usage(capsys):
    rc = main([])
    assert rc == 0
    assert "usage: jaxrens-al" in capsys.readouterr().out


def test_unknown_command_errors(capsys):
    rc = main(["bogus"])
    assert rc == 2
    assert "unknown command" in capsys.readouterr().err


def test_dashboard_dispatch(monkeypatch):
    seen = {}

    def stub(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("jaxrens_al.cli.main", stub)
    rc = main(["dashboard", "some/calchome", "--force"])
    assert rc == 0
    assert seen["argv"] == ["some/calchome", "--force"]


def test_refdb_dispatch(monkeypatch):
    seen = {}

    def stub(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("jaxrens_al.run.reference_db.main", stub)
    rc = main(["refdb", "Cu", "out.extxyz"])
    assert rc == 0
    assert seen["argv"] == ["Cu", "out.extxyz"]
