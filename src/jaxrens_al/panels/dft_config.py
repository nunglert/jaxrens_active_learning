"""Panel specs for the DFT-labelled / evaluated-config panels.

These three panels are the generic ``alviz.panels.Extxyz*`` readers (alviz owns
them, and stays free of any run-layout knowledge), so their jaxrens ctx-adapters
cannot live in the panel class -- they live here, the one jaxrens home for
"wire an Extxyz panel from a RunContext". Everything else co-locates its spec
with its panel class in the sibling modules.
"""

from __future__ import annotations

from alviz import register_simple
from alviz.panels import (
    ExtxyzEnergyErrorPanel,
    ExtxyzParityPanel,
    ExtxyzSampledEnergyVolume,
)

from jaxrens_al.keys import REF_ENERGY_KEY, REF_FORCES_KEY, first_has_info_key


def _build_sampled_energy_volume(ctx, c):
    if not ctx.eval_files:
        return None
    return ExtxyzSampledEnergyVolume(
        config_files=ctx.eval_files,
        energy_key=REF_ENERGY_KEY,
        forces_key=REF_FORCES_KEY,
        **c,
    )


def _build_parity(ctx, c):
    if not first_has_info_key(ctx.eval_files, ctx.surrogate_energy_key):
        return None
    return ExtxyzParityPanel(
        config_files=ctx.eval_files,
        energy_ref_key=REF_ENERGY_KEY,
        energy_pred_key=ctx.surrogate_energy_key,
        show_forces=False,
        **c,
    )


def _build_energy_error(ctx, c):
    if not first_has_info_key(ctx.eval_files, ctx.surrogate_energy_key):
        return None
    return ExtxyzEnergyErrorPanel(
        config_files=ctx.eval_files,
        energy_ref_key=REF_ENERGY_KEY,
        energy_pred_key=ctx.surrogate_energy_key,
        **c,
    )


register_simple("sampled_energy_volume", _build_sampled_energy_volume)
register_simple("parity", _build_parity)
register_simple("energy_error", _build_energy_error)
