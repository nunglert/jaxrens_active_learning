"""Tests for the shared argparse groups (jaxrens_al.cli_args).

These helpers back both the standalone scripts and the global CLI, so the
subtle part is that each consumer keeps its *own* defaults (the scripts scan
Kelvin for eV models and default the pressure/temperature swap on; the AL CLI
uses reduced units and swap off). Lock that in.
"""

import argparse

import pytest

from jaxrens_al.cli_args import (
    add_config_arg,
    add_output_args,
    add_processing_args,
    add_ref_key_args,
    add_thermo_args,
    apply_config,
    dump_config,
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


def test_processing_args_defaults():
    # The processing/job knobs are globals; there are no per-panel (Steinhardt)
    # CLI flags -- a Steinhardt heatmap is a config `add:` entry.
    args = _parse(add_processing_args)
    assert args.cluster == "local"
    assert args.shard_size == 2000
    assert args.num_parallel == 1
    assert args.n_processors == 1
    assert args.relax_batch_size == 32
    assert args.relax_shard_size is None
    assert not hasattr(args, "steinhardt_rcut")


def test_relax_shard_size_falls_back_to_global():
    from jaxrens_al.cli_args import relax_shard_size

    # Unset override -> the global --shard-size.
    args = _parse(add_processing_args, argv=["--shard-size", "500"])
    assert relax_shard_size(args) == 500
    # Set override -> wins, leaving --shard-size for the other resolvers.
    args = _parse(
        add_processing_args,
        argv=["--shard-size", "500", "--relax-shard-size", "64"],
    )
    assert args.shard_size == 500
    assert relax_shard_size(args) == 64


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
    p.write_text("swap-pt: true\nshard-size: 5000\nonly:\n  - eos\n")
    cfg = load_config(p)
    assert cfg == {"swap_pt": True, "shard_size": 5000, "only": ["eos"]}


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


def test_apply_config_extracts_panel_keys(tmp_path, caplog):
    """add/disable/global are handled specially, not warned as unknown."""
    import logging

    from jaxrens_al.cli import _build_parser

    (tmp_path / "AL_progress.json").write_text("[]")
    (tmp_path / "dashboard.yaml").write_text(
        "global:\n  width: 6.0\n"
        "disable: [ns_thermodynamics]\n"
        "add:\n"
        "  - {type: steinhardt_expectation, r_cut: 3.5, l: 6}\n"
    )
    parser = _build_parser()
    argv = [str(tmp_path)]
    args = parser.parse_args(argv)
    with caplog.at_level(logging.WARNING):
        args = apply_config(parser, argv, args, logging.getLogger("t"))
    # global block merged into a flat key.
    assert args.width == 6.0
    # panel keys stashed raw, not warned about.
    assert "add" not in caplog.text and "disable" not in caplog.text
    raw = args.panel_config_raw
    assert raw["disable"] == ["ns_thermodynamics"]
    assert raw["add"][0]["l"] == 6


def test_apply_config_custom_positional_and_run_dir(tmp_path):
    """Section subcommands discover a config beside a non-'path' positional."""
    import argparse
    import logging

    out = tmp_path / "output"
    out.mkdir()
    (out / "dashboard.yaml").write_text("prefix: mace_sn16\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--prefix", default="ns")
    add_config_arg(parser)

    argv = [str(out)]
    args = parser.parse_args(argv)
    args = apply_config(
        parser,
        argv,
        args,
        logging.getLogger("t"),
        run_dir=out,
        positional="output_dir",
    )
    # Config beside the input dir supplies the prefix; the positional itself is
    # never overridden from the config.
    assert args.prefix == "mace_sn16"
    assert args.output_dir == str(out)


# ---------------------------------------------------------------------------
# dump_config: parser-derived config template
# ---------------------------------------------------------------------------
def test_dump_config_round_trips_and_keys_are_valid_dests(tmp_path):
    """A dumped template loads cleanly and every active key is a real dest.

    This is the drift guard: because the template is derived from the parser's
    own actions, a new flag appears here for free -- and load_config -> dests
    proves no dumped key would be silently dropped by apply_config.
    """
    pytest.importorskip("yaml")
    from jaxrens_al.cli_args import load_config

    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    add_output_args(parser)
    add_thermo_args(parser, k_B=8.617e-5, tmin=100.0, swap_default=True)
    add_processing_args(parser)

    text = dump_config(parser, title="test flavor")
    p = tmp_path / "dashboard.yaml"
    p.write_text(text)
    cfg = load_config(p)  # parses only the active (non-None-default) keys

    dests = {a.dest for a in parser._actions}
    assert set(cfg) <= dests  # no key would be dropped as "unknown"
    # Active defaults are reproduced; a None-default flag is commented out.
    assert cfg["cluster"] == "local"
    assert cfg["swap_pt"] is True
    # The positional and --config are never templated.
    assert "config" not in cfg
    # A None-default flag (--output) is listed commented for discovery.
    assert "# output:" in text
