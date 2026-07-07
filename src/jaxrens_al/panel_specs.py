"""Composition of the AL dashboard's panel set (registry glue only).

Each panel registers its own :class:`alviz.PanelSpec` + builder **in its own
module** (``jaxrens_al.panels.*``; the generic Extxyz panels register in
``panels.dft_config``), so the panel-specific knowledge lives with the panel.
Importing :mod:`jaxrens_al.panels` runs those registrations -- this module just
imports it and then *composes*: the curated default order, the NS-section
subset, and the ``dashboard.*`` config -> spec plumbing. No panel-specific
wiring lives here.
"""

from __future__ import annotations

from typing import Optional

from alviz import PanelSpec

# Importing the panels package runs every panel module's ``register_panel``
# call, populating the alviz panel registry this module composes over.
import jaxrens_al.panels  # noqa: F401

# Re-exported for callers that construct one directly (sections, tests). The
# authoritative definition lives with the panel, in panels.steinhardt.
from jaxrens_al.panels.steinhardt import SteinhardtSpec

__all__ = [
    "SteinhardtSpec",
    "make_spec",
    "default_specs",
    "ns_default_specs",
    "apply_ctx_globals",
    "parse_panel_config",
]


def make_spec(type_name: str, **params) -> PanelSpec:
    """Construct a registered panel spec by ``type`` name (+ its params)."""
    from alviz.spec import PANEL_REGISTRY

    return PANEL_REGISTRY[type_name].spec_model(**params)


# Historical render order of the curated AL page (every builder self-gates, so
# absent-input / wrong-view panels drop out). Steinhardt is intentionally *not*
# here -- it is opt-in via a config ``add:`` entry (no CLI flag).
_DEFAULT_ORDER = [
    "dataset_growth",
    "sampled_energy_volume",
    "parity",
    "energy_error",
    "sampled_ns_energy",
    "quests_pca",
    "resampled_grid",
    "acceptance_rates",
    "step_sizes",
    "adaptation_acceptance",
    "ns_energy_trace",
    "re_swap_rates",
    "atomic_envs",
    "eos",
    "neuralil_parity",
    "model_parity",
    "neuralil_training_curve",
    "mace_training_curve",
    "ns_thermodynamics",
]


def default_specs(ctx=None) -> list:
    """The curated default panel specs, in historical render order.

    ``ctx`` is accepted for symmetry / future context-dependent defaults; the
    current set is static (builders do all the gating).
    """
    return [make_spec(t) for t in _DEFAULT_ORDER]


def ns_default_specs(*, per_run: bool = False) -> list:
    """The NS section's curated panels (a subset of the AL page), in order.

    One definition of "what an NS dashboard shows", shared by the standalone NS
    dashboard and any future consumer.
    """
    return [
        make_spec("ns_energy_trace"),
        make_spec("ns_thermodynamics"),
        make_spec("step_sizes", per_run=per_run),
        make_spec("adaptation_acceptance", per_run=per_run),
        make_spec("re_swap_rates"),
        make_spec("sampled_ns_energy"),
    ]


def apply_ctx_globals(specs, ctx) -> list:
    """Fill each spec's *unset* NS-global fields from the run context.

    Global settings (``swap_axes`` / ``k_B`` / ``temperatures``) live on the
    context; a spec inherits them automatically unless it set the field itself
    (a per-panel override). A panel opts in simply by declaring the field --
    param-less specs are returned untouched.
    """
    from alviz import apply_globals

    t = getattr(ctx, "ns_temperatures", None)
    g = {
        "swap_axes": ctx.ns_swap_axes,
        "k_B": ctx.ns_k_B,
        "temperatures": list(t) if t is not None else None,
    }
    return [apply_globals(s, g) for s in specs]


def parse_panel_config(raw) -> Optional[dict]:
    """Turn a config's raw panel mapping into an ``assemble`` ``panel_config``.

    ``raw`` is ``{"add": [...], "disable": [...], "panels": [...]}`` of plain
    dicts (from a ``dashboard.*`` file, via ``cli_args.apply_config``). The
    ``add`` / ``panels`` lists are validated into :class:`~alviz.PanelSpec`s
    against the panel registry (an unknown ``type`` or field raises). Returns
    ``None`` when there is nothing to layer.
    """
    if not raw:
        return None
    from alviz import parse_specs

    out = {}
    if raw.get("disable"):
        out["disable"] = list(raw["disable"])
    if raw.get("add"):
        out["add"] = parse_specs(raw["add"])
    if raw.get("panels"):
        out["panels"] = parse_specs(raw["panels"])
    return out or None
