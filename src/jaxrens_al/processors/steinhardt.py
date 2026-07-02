"""The ``steinhardt`` processor -- one module, both ends of the contract.

A downstream processor for ``remote_handler``'s trajectory-processing framework
(:mod:`remote_handler.processing`), mirroring the built-in ``rdf`` /
``optimize`` modules. It holds, in one place:

* :class:`SteinhardtArgs` -- the arg schema (single source of truth), used by
  the host to emit the payload and by the driver to validate it;
* :func:`run` -- the cluster-side compute (jaxrens Steinhardt annotation;
  ``jaxrens`` imported lazily so importing this module host-side is cheap);
* :class:`SteinhardtProcessor` -- the host-side spec;
* the module-level :func:`register` call wiring ``run`` / ``SteinhardtArgs``
  into the registry.

Steinhardt needs ``jaxrens``, which ``remote_handler`` deliberately does not
depend on -- so it lives here, not upstream. The stock shard driver imports
this module on the cluster (``SteinhardtProcessor.registration_module``
defaults to this module, via ``Processor.__init_subclass__``) before resolving
steps, so its ``register`` runs and ``"steinhardt"`` resolves -- no custom
driver.
Requires ``jaxrens_al`` installed in the cluster analysis env.

Use it like any processor, e.g. on its own or after a relaxation::

    from remote_handler.processing import RemoteTrajectoryProcessor
    from jaxrens_al.processors import SteinhardtProcessor

    RemoteTrajectoryProcessor(
        traj_files, ..., processors=[SteinhardtProcessor(r_cut=3.5)]
    )

The annotated trajectory (per-atom ``q<l>`` / ``w<l>`` arrays + per-frame
``q<l>_mean`` / ``w<l>_mean`` in ``atoms.info``) is read back by the pure
:mod:`jaxrens_al.panels.steinhardt` reader panels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing.base import Processor
from remote_handler.processing.registry import register
from remote_handler.scripts.input_models import ScriptInput


# --- arg schema ------------------------------------------------------------
@dataclass
class SteinhardtArgs(ScriptInput):
    """Args for the ``steinhardt`` processor.

    ``l_values`` are the bond-order degrees (even in practice, e.g.
    ``[4, 6]``); ``r_cut`` is the neighbour cutoff in Angstrom and
    ``r_cut_min`` an optional lower cutoff that drops near neighbours.
    """

    l_values: list[int] = field(default_factory=list)
    r_cut: float = 0.0
    r_cut_min: float = 0.0

    def validate(self, root: Optional[Path] = None) -> None:
        if not self.l_values:
            raise ScriptInputError(
                "steinhardt: 'l_values' must be a non-empty list"
            )
        if not all(isinstance(v, int) for v in self.l_values):
            raise ScriptInputError("steinhardt: 'l_values' must be ints")
        if not isinstance(self.r_cut, (int, float)) or self.r_cut <= 0:
            raise ScriptInputError(
                "steinhardt: 'r_cut' must be a positive number"
            )
        if not isinstance(self.r_cut_min, (int, float)) or self.r_cut_min < 0:
            raise ScriptInputError(
                "steinhardt: 'r_cut_min' must be a non-negative number"
            )
        if self.r_cut_min >= self.r_cut:
            raise ScriptInputError(
                "steinhardt: 'r_cut_min' must be smaller than 'r_cut'"
            )


# --- cluster-side compute --------------------------------------------------
def run(atoms_list, args, calcpath=None):
    """Annotate each structure with per-atom Steinhardt ``q_l`` / ``w_l``.

    Attaches per-atom arrays ``q<l>`` / ``w<l>`` and per-frame
    ``q<l>_mean`` / ``w<l>_mean`` (``atoms.info``), matching the layout the
    ``jaxrens annotate-steinhardt`` CLI produces and the reader panels expect.
    """
    import numpy as np
    from jaxrens.postprocess.steinhardt import (
        build_solidr_table,
        build_wigner_triples,
        neighbour_edges,
        qw_from_edges,
    )

    ls = [int(v) for v in args["l_values"]]
    r_cut = float(args["r_cut"])
    r_cut_min = float(args.get("r_cut_min", 0.0))

    # Static per-l tables are input-independent -- build them once and reuse
    # across every frame (jaxrens' ``calc_qw`` would rebuild them per call).
    tables = {v: (build_solidr_table(v), build_wigner_triples(v)) for v in ls}

    for atoms in atoms_list:
        n_atoms = len(atoms)
        for v in ls:
            table, triples = tables[v]
            centers, vectors = neighbour_edges(atoms, r_cut, r_cut_min)
            if len(centers) == 0:
                q = np.zeros(n_atoms)
                w = np.zeros(n_atoms)
            else:
                q, w = qw_from_edges(
                    v, vectors, centers, n_atoms, table, triples
                )
                q = np.asarray(q)
                w = np.asarray(w)
            atoms.new_array(f"q{v}", q)
            atoms.new_array(f"w{v}", w)
            atoms.info[f"q{v}_mean"] = float(q.mean())
            atoms.info[f"w{v}_mean"] = float(w.mean())
    return atoms_list


# --- host-side spec --------------------------------------------------------
class SteinhardtProcessor(Processor):
    """Annotate structures with per-atom Steinhardt order parameters.

    A pure host-side spec: it contributes only its remote args (degrees +
    cutoffs); the compute is :func:`run`, resolved on the cluster by the
    matching registry name. ``registration_module`` defaults to this module,
    so the stock driver imports it and ``register`` runs -- nothing to declare.

    Args:
        r_cut: Neighbour cutoff in Angstrom (required).
        l_values: Bond-order degrees (even in practice; default ``(4, 6)``).
        r_cut_min: Lower neighbour cutoff in Angstrom (default 0).
    """

    name = "steinhardt"

    def __init__(
        self,
        r_cut: float,
        *,
        l_values: Sequence[int] = (4, 6),
        r_cut_min: float = 0.0,
    ):
        self.l_values = [int(v) for v in l_values]
        if not self.l_values:
            raise ValueError("SteinhardtProcessor needs at least one l value.")
        self.r_cut = float(r_cut)
        self.r_cut_min = float(r_cut_min)

    def build_args(self, savepath: str) -> dict:
        # Fill the shared schema so field names come from one place; the
        # cluster driver validates the payload against this same schema.
        return SteinhardtArgs(
            l_values=list(self.l_values),
            r_cut=self.r_cut,
            r_cut_min=self.r_cut_min,
        ).to_dict()


register("steinhardt", run, schema=SteinhardtArgs)
