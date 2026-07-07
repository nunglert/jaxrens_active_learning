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

_REF_ENERGY_KEY = "REF_energy"
_REF_FORCES_KEY = "REF_forces"


def _first_has_info_key(files, key) -> bool:
    """True if the first config of ``files`` carries ``key`` in its info.

    A cheap header-peek so the surrogate-vs-DFT parity panels are only built
    when the surrogate key is present (reading a missing key would raise at
    compute time). Any read error -> False.
    """
    if not files:
        return False
    try:
        from ase.io import read

        return key in read(str(files[0]), index=0).info
    except Exception:
        return False


def _build_sampled_energy_volume(ctx, c):
    if not ctx.eval_files:
        return None
    return ExtxyzSampledEnergyVolume(
        config_files=ctx.eval_files,
        energy_key=_REF_ENERGY_KEY,
        forces_key=_REF_FORCES_KEY,
        **c,
    )


def _build_parity(ctx, c):
    if not _first_has_info_key(ctx.eval_files, ctx.surrogate_energy_key):
        return None
    return ExtxyzParityPanel(
        config_files=ctx.eval_files,
        energy_ref_key=_REF_ENERGY_KEY,
        energy_pred_key=ctx.surrogate_energy_key,
        show_forces=False,
        **c,
    )


def _build_energy_error(ctx, c):
    if not _first_has_info_key(ctx.eval_files, ctx.surrogate_energy_key):
        return None
    return ExtxyzEnergyErrorPanel(
        config_files=ctx.eval_files,
        energy_ref_key=_REF_ENERGY_KEY,
        energy_pred_key=ctx.surrogate_energy_key,
        **c,
    )


register_simple("sampled_energy_volume", _build_sampled_energy_volume)
register_simple("parity", _build_parity)
register_simple("energy_error", _build_energy_error)
