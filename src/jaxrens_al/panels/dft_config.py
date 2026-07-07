"""Panel specs for the DFT-labelled / evaluated-config panels.

These three panels are the generic ``alviz.panels.Extxyz*`` readers (alviz owns
them, and stays free of any run-layout knowledge), so their jaxrens ctx-adapters
cannot live in the panel class -- they live here, the one jaxrens home for
"wire an Extxyz panel from a RunContext". Everything else co-locates its spec
with its panel class in the sibling modules.
"""

from __future__ import annotations

from alviz import PanelSpec, register_panel, register_simple
from alviz.panels import (
    ExtxyzEnergyErrorPanel,
    ExtxyzParityPanel,
    ExtxyzSampledEnergyVolume,
)

from jaxrens_al.keys import first_has_info_key


class SampledEnergyVolumeSpec(PanelSpec):
    """Energy/volume spread of one config set.

    ``source`` names the :class:`RunContext` attribute holding the configs (the
    evaluated set by default; a training set in the training section), so one
    panel type serves every consumer.
    """

    type: str = "sampled_energy_volume"
    source: str = "eval_files"


def _build_sampled_energy_volume(spec, ctx, c):
    files = getattr(ctx, spec.source, None)
    if not files:
        return None
    return ExtxyzSampledEnergyVolume(
        config_files=files,
        energy_key=ctx.energy_key,
        forces_key=ctx.forces_key,
        **c,
    )


def _build_parity(ctx, c):
    if not first_has_info_key(ctx.eval_files, ctx.surrogate_energy_key):
        return None
    return ExtxyzParityPanel(
        config_files=ctx.eval_files,
        energy_ref_key=ctx.energy_key,
        energy_pred_key=ctx.surrogate_energy_key,
        show_forces=False,
        **c,
    )


def _build_energy_error(ctx, c):
    if not first_has_info_key(ctx.eval_files, ctx.surrogate_energy_key):
        return None
    return ExtxyzEnergyErrorPanel(
        config_files=ctx.eval_files,
        energy_ref_key=ctx.energy_key,
        energy_pred_key=ctx.surrogate_energy_key,
        **c,
    )


register_panel(
    "sampled_energy_volume",
    SampledEnergyVolumeSpec,
    _build_sampled_energy_volume,
)
register_simple("parity", _build_parity)
register_simple("energy_error", _build_energy_error)
