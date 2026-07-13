"""Tests for the registry-driven panel wiring (jaxrens_al.dashboard.panel_specs).

These lock the behaviour of ``dashboard.assemble`` after it was rebuilt on
``alviz``'s spec/registry machinery: the curated default set + gating must match
the old hand-wired list, and a ``panel_config`` (disable / add / authoritative
panels) must layer on top -- including several Steinhardt heatmaps at once.
"""

import pytest

from jaxrens_al.dashboard import assemble, iteration_context, summary_context
from jaxrens_al.dashboard.panel_specs import SteinhardtSpec, default_specs

pytestmark = pytest.mark.local


def _names(panels):
    return [p.name for p in panels]


@pytest.fixture
def rich(make_calchome, tmp_path):
    """A fully-featured run + its dashboard datapath."""
    calchome = make_calchome(
        tmp_path,
        n_iter=2,
        eval_with_surrogate=True,
        with_training=True,
        with_mace_training=True,
        with_resampling=True,
        with_ns=True,
        with_acc_rates=True,
        with_adaptation=True,
        with_re_stats=True,
        with_model=True,
    )
    dp = calchome / "dashboard"
    dp.mkdir(exist_ok=True)
    return calchome, dp


# ---------------------------------------------------------------------------
# Characterization: the curated set + gating is unchanged.
# ---------------------------------------------------------------------------
def test_summary_panels_match(rich):
    calchome, dp = rich
    assert _names(assemble(summary_context(calchome, dp))) == [
        "dataset_growth",
        "sampled_energy_volume",
        "sampled_ns_energy",
        "neuralil_parity",
        "neuralil_training_curve",
        "mace_training_curve",
        "ns_thermodynamics",
    ]


def test_iteration_panels_match(rich):
    calchome, dp = rich
    ctx = iteration_context(calchome, dp, 0)
    cfg = {"add": [SteinhardtSpec(r_cut=3.5, l=6)]}  # opt-in via config
    assert _names(assemble(ctx, panel_config=cfg)) == [
        "sampled_energy_volume",
        "sampled_ns_energy",
        "quests_pca",
        "resampled_grid",
        "acceptance_rates",
        "step_sizes",
        "adaptation_acceptance",
        "ns_energy_trace",
        "re_swap_rates",
        "neuralil_parity",
        "neuralil_training_curve",
        "mace_training_curve",
        "ns_thermodynamics",
        "steinhardt_q6_rc3.5",  # opt-in, param-keyed name
    ]


def test_default_specs_are_static_and_ordered():
    names = [s.type for s in default_specs()]
    assert names[0] == "dataset_growth"
    assert "steinhardt_expectation" not in names  # opt-in, never a default


# ---------------------------------------------------------------------------
# Config layering: disable / add / authoritative panels.
# ---------------------------------------------------------------------------
def test_disable_drops_a_default(rich):
    calchome, dp = rich
    ctx = summary_context(calchome, dp)
    names = _names(
        assemble(ctx, panel_config={"disable": ["ns_thermodynamics"]})
    )
    assert "ns_thermodynamics" not in names
    assert "dataset_growth" in names  # the rest survive


def test_add_multiple_steinhardts(rich):
    calchome, dp = rich
    ctx = iteration_context(calchome, dp, 0)  # no ctx.steinhardt
    cfg = {
        "add": [
            SteinhardtSpec(r_cut=3.5, l=6),
            SteinhardtSpec(r_cut=3.5, l=4),
            SteinhardtSpec(r_cut=3.0, l=6),
        ]
    }
    names = _names(assemble(ctx, panel_config=cfg))
    # Three distinct, param-keyed Steinhardt panels coexist.
    assert names[-3:] == [
        "steinhardt_q6_rc3.5",
        "steinhardt_q4_rc3.5",
        "steinhardt_q6_rc3",
    ]


def test_duplicate_steinhardt_without_name_raises(rich):
    calchome, dp = rich
    ctx = iteration_context(calchome, dp, 0)
    cfg = {
        "add": [SteinhardtSpec(r_cut=3.5, l=6), SteinhardtSpec(r_cut=3.5, l=6)]
    }
    with pytest.raises(ValueError, match="same name"):
        assemble(ctx, panel_config=cfg)


def test_authoritative_panels_replace_defaults(rich):
    calchome, dp = rich
    ctx = iteration_context(calchome, dp, 0)
    cfg = {"panels": [SteinhardtSpec(r_cut=3.5, l=6, name="only_one")]}
    assert _names(assemble(ctx, panel_config=cfg)) == ["only_one"]


# ---------------------------------------------------------------------------
# parse_panel_config: raw config dicts -> specs
# ---------------------------------------------------------------------------
def test_parse_panel_config_builds_specs():
    from jaxrens_al.dashboard.panel_specs import parse_panel_config

    raw = {
        "disable": ["ns_thermodynamics"],
        "add": [
            {"type": "steinhardt_expectation", "r_cut": 3.5, "l": 6},
            {"type": "steinhardt_expectation", "r_cut": 3.5, "l": 4},
        ],
        "panels": None,
    }
    cfg = parse_panel_config(raw)
    assert cfg["disable"] == ["ns_thermodynamics"]
    assert [s.l for s in cfg["add"]] == [6, 4]
    assert "panels" not in cfg


def test_parse_panel_config_none_when_empty():
    from jaxrens_al.dashboard.panel_specs import parse_panel_config

    assert parse_panel_config(None) is None
    assert parse_panel_config({"add": None, "disable": None}) is None


def test_parse_panel_config_unknown_type_raises():
    from jaxrens_al.dashboard.panel_specs import parse_panel_config

    with pytest.raises(ValueError, match="Unknown panel type"):
        parse_panel_config({"add": [{"type": "bogus"}]})


# ---------------------------------------------------------------------------
# Global inheritance: a spec inherits unset globals; explicit values override.
# ---------------------------------------------------------------------------
def test_spec_inherits_and_overrides_global_swap_axes(rich):
    calchome, dp = rich
    ctx = iteration_context(
        calchome, dp, 0, ns_swap_axes=True
    )  # global swap on
    cfg = {
        "add": [
            SteinhardtSpec(r_cut=3.5, l=6),  # inherits global True
            SteinhardtSpec(r_cut=3.5, l=4, swap_axes=False),  # overrides
        ]
    }
    panels = {p.name: p for p in assemble(ctx, panel_config=cfg)}
    assert panels["steinhardt_q6_rc3.5"]._swap_axes is True  # inherited
    assert panels["steinhardt_q4_rc3.5"]._swap_axes is False  # overridden
    # The curated thermodynamics heatmap also inherits the global.
    assert panels["ns_thermodynamics"]._swap_axes is True
