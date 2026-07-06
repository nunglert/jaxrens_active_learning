"""Tests for the shared argparse groups (jaxrens_al.cli_args).

These helpers back both the standalone scripts and the global CLI, so the
subtle part is that each consumer keeps its *own* defaults (the scripts scan
Kelvin for eV models and default the pressure/temperature swap on; the AL CLI
uses reduced units and swap off). Lock that in, plus the ``steinhardt_params``
shape the section builders expect.
"""

import argparse

import pytest

from jaxrens_al.cli_args import (
    add_output_args,
    add_ref_key_args,
    add_steinhardt_args,
    add_thermo_args,
    apply_config,
    steinhardt_params,
)

pytestmark = pytest.mark.local


def _parse(*adders, argv=()):
    parser = argparse.ArgumentParser()
    for add in adders:
        add(parser)
    return parser.parse_args(list(argv))


def test_output_args_defaults_and_force_help():
    parser = argparse.ArgumentParser()
    add_output_args(parser, force_help="Replot even if it exists.")
    args = parser.parse_args([])
    assert args.output is None and args.datapath is None
    assert args.force is False
    assert parser.parse_args(["--force"]).force is True


def test_ref_key_args_defaults():
    args = _parse(add_ref_key_args)
    assert args.energy_key == "REF_energy"
    assert args.forces_key == "REF_forces"


def test_thermo_defaults_are_parameterized():
    # AL-CLI flavour: reduced units, swap off.
    al = _parse(add_thermo_args)
    assert (al.k_B, al.tmin, al.tmax, al.n_T, al.swap_pt) == (
        1.0,
        0.05,
        2.0,
        50,
        False,
    )
    # Script flavour: Kelvin defaults, swap on.
    parser = argparse.ArgumentParser()
    add_thermo_args(
        parser,
        k_B=8.617e-5,
        tmin=100.0,
        tmax=3000.0,
        n_T=60,
        swap_default=True,
    )
    ns = parser.parse_args([])
    assert (ns.tmin, ns.tmax, ns.n_T, ns.swap_pt) == (100.0, 3000.0, 60, True)
    # BooleanOptionalAction gives the negation either way.
    assert parser.parse_args(["--no-swap-pt"]).swap_pt is False


def test_steinhardt_params_none_unless_requested():
    args = _parse(add_steinhardt_args)
    assert args.cluster == "local"
    assert steinhardt_params(args) is None  # --steinhardt-rcut omitted


def test_steinhardt_params_shape_when_requested():
    args = _parse(
        add_steinhardt_args,
        argv=["--steinhardt-rcut", "3.5", "--steinhardt-l", "4"],
    )
    assert steinhardt_params(args) == {
        "r_cut": 3.5,
        "r_cut_min": 0.0,
        "l": 4,
    }


# ---------------------------------------------------------------------------
# Config file mechanism
# ---------------------------------------------------------------------------
def test_find_config_priority(tmp_path):
    from jaxrens_al.cli_args import find_config

    assert find_config(tmp_path) is None
    (tmp_path / "dashboard.json").write_text("{}")
    (tmp_path / "dashboard.yaml").write_text("sections: true\n")
    # .yaml outranks .json.
    assert find_config(tmp_path).name == "dashboard.yaml"


def test_load_config_yaml_normalizes_hyphen_keys(tmp_path):
    from jaxrens_al.cli_args import load_config

    p = tmp_path / "dashboard.yaml"
    p.write_text("swap-pt: true\nsteinhardt-rcut: 3.5\nonly:\n  - eos\n")
    cfg = load_config(p)
    assert cfg == {"swap_pt": True, "steinhardt_rcut": 3.5, "only": ["eos"]}


def test_load_config_json(tmp_path):
    from jaxrens_al.cli_args import load_config

    p = tmp_path / "dashboard.json"
    p.write_text('{"width": 6.0, "eos-backend": "mace"}')
    assert load_config(p) == {"width": 6.0, "eos_backend": "mace"}


def test_load_config_rejects_non_mapping(tmp_path):
    from jaxrens_al.cli_args import load_config

    p = tmp_path / "dashboard.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        load_config(p)


def test_apply_config_cli_overrides_and_drops_unknown(tmp_path, caplog):
    import logging

    from jaxrens_al.cli import _build_parser

    (tmp_path / "AL_progress.json").write_text("[]")
    (tmp_path / "dashboard.yaml").write_text(
        "width: 6.0\nsections: true\nbogus: 1\n"
    )
    parser = _build_parser()
    argv = [str(tmp_path), "--width", "9.0"]  # CLI width beats config
    args = parser.parse_args(argv)
    with caplog.at_level(logging.WARNING):
        args = apply_config(parser, argv, args, logging.getLogger("t"))
    assert args.width == 9.0  # CLI wins
    assert args.sections is True  # from config
    assert "bogus" in caplog.text  # unknown key warned
    assert not hasattr(args, "bogus")


def test_apply_config_noop_without_file(tmp_path):
    import logging

    from jaxrens_al.cli import _build_parser

    parser = _build_parser()
    argv = [str(tmp_path)]
    args = parser.parse_args(argv)
    # No dashboard.* present -> args returned unchanged.
    assert apply_config(parser, argv, args, logging.getLogger("t")) is args
