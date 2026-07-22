"""The ``spacegroup`` processor -- assign a space group to each structure.

A downstream processor for ``remote_handler``'s trajectory-processing framework
(mirroring its built-in ``rdf`` processor). On the cluster it runs
``spglib`` on each frame and writes the space-group number/symbol into
``atoms.info`` -- the per-sample *phase label* the phase-diagram panel groups by.

Meant to run **after** the ``relax`` step (space groups are only meaningful for
structures relaxed to their local minimum), as the second step of the phase
pipeline; it composes with any producer, though. ``spglib`` is imported lazily
so importing this module host-side is cheap; the cluster analysis env needs
``spglib`` (a light, pure-ish dep) alongside ``jaxrens_al``.

Writes ``atoms.info["spacegroup"]`` (int number, 0 if undetermined) and
``atoms.info["spacegroup_symbol"]`` (Hermann-Mauguin, ``"unknown"`` if none).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing.base import Processor
from remote_handler.processing.registry import register
from remote_handler.scripts.input_models import ScriptInput

# The info keys both ends agree on live in the shared (pure) keys module.
from jaxrens_al.keys import SPACEGROUP_KEY, SPACEGROUP_SYMBOL_KEY


# --- arg schema ------------------------------------------------------------
@dataclass
class SpaceGroupArgs(ScriptInput):
    """Args for the ``spacegroup`` processor.

    ``symprec`` is the spglib distance tolerance (Angstrom) and
    ``angle_tolerance`` its angle tolerance (degrees, ``-1`` = spglib default).
    """

    symprec: float = 1e-3
    angle_tolerance: float = -1.0

    def validate(self, root: Optional[Path] = None) -> None:
        if not isinstance(self.symprec, (int, float)) or self.symprec <= 0:
            raise ScriptInputError(
                "spacegroup: 'symprec' must be a positive number"
            )


# --- cluster-side compute --------------------------------------------------
def run(atoms_list, args, calcpath=None):
    """Annotate each structure with its spglib space group."""
    import spglib

    symprec = float(args.get("symprec", 1e-3))
    angle_tolerance = float(args.get("angle_tolerance", -1.0))

    for atoms in atoms_list:
        number, symbol = 0, "unknown"
        try:
            cell = (
                atoms.get_cell()[:],
                atoms.get_scaled_positions(),
                atoms.get_atomic_numbers(),
            )
            sg = spglib.get_spacegroup(
                cell, symprec=symprec, angle_tolerance=angle_tolerance
            )
            if sg:  # e.g. "Fd-3m (227)"
                symbol = sg.split(" (")[0]
                number = int(sg.rsplit("(", 1)[-1].rstrip(")"))
        except Exception:  # a non-periodic / degenerate cell -> undetermined
            number, symbol = 0, "unknown"
        atoms.info[SPACEGROUP_KEY] = number
        atoms.info[SPACEGROUP_SYMBOL_KEY] = symbol
    return atoms_list


# --- host-side spec --------------------------------------------------------
class SpaceGroupProcessor(Processor):
    """Annotate structures with their spglib space group (number + symbol).

    A pure host-side spec (no staged files); the compute is :func:`run`.
    Pair it after ``relax`` so the space group is that of the relaxed basin.

    Args:
        symprec: spglib distance tolerance in Angstrom (default 1e-3).
        angle_tolerance: spglib angle tolerance in degrees (default -1 =
            spglib's own default).
    """

    name = "spacegroup"

    def __init__(self, symprec: float = 1e-3, angle_tolerance: float = -1.0):
        self.symprec = float(symprec)
        self.angle_tolerance = float(angle_tolerance)

    def build_args(self, savepath: str) -> dict:
        return SpaceGroupArgs(
            symprec=self.symprec, angle_tolerance=self.angle_tolerance
        ).to_dict()


register("spacegroup", run, schema=SpaceGroupArgs)
