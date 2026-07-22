"""Remote MLIP evaluation needs -- plain data, resolved run-side.

Some panels show a *trained model's* predictions on a config set: the EOS
panel's model E(V) curve and ``ModelParityPanel``'s re-scored parity both load
the model and evaluate it. A MACE-jax bundle pulls in ``mace_jax`` /
``cuequivariance`` (a CUDA build), so it cannot run in a CPU dashboard process
-- it must run on a GPU cluster.

This is the seam that moves that evaluation off the dashboard: a panel
*declares* a :class:`ModelEvalNeed` (which configs, which model, which backend,
whether it also needs forces) and later *receives* the annotated files back.
The run-side :func:`jaxrens_al.run.processing_resolver.resolve_model_evaluations`
collects the needs, runs one cluster job per distinct model (the
``model_eval`` processor writes each frame's model energy/forces into per-model
``info``/``arrays`` keys), and delivers ``{input -> annotated}`` back. Both
halves here are pure data / attribute assignment -- no ``remote_handler``, no
cluster -- so the panels stay light, dashboard-side readers.

Mirrors :mod:`jaxrens_al.dashboard.panels.processing` (the Steinhardt seam); a separate
method (``model_eval_needs``) keeps the two resolvers from crossing wires.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEvalNeed:
    """A panel's request to have configs scored by a trained model.

    Purely declarative and hashable, so identical needs (e.g. the EOS and
    model-parity panels of one iteration sharing a model) deduplicate.

    Attributes:
        inputs: Config file(s) to score -- the grouping key.
        model: Path to the trained-model bundle (a mace-jax bundle dir or a
            neuralil ``.pkl``); the identity the per-model key derives from.
        backend: jaxrens backend to load the model with (``"mace"`` / etc.).
        want_forces: Also annotate per-atom model forces (parity needs them;
            the EOS energy curve does not).
        max_neighbors: Static neighbour-buffer size for the backend call.
        min_image_cutoff: When set (Å), each frame is scored on the smallest
            supercell hosting the cutoff (the annotated energy is scaled back to
            the input cell's atom count). Lets the EOS panel keep a unit-cell
            DFT reference; ``None`` (parity) scores frames as-is.
    """

    inputs: tuple
    model: str
    backend: str
    want_forces: bool = False
    max_neighbors: int = 50
    min_image_cutoff: float | None = None

    @classmethod
    def make(
        cls,
        inputs,
        model,
        backend: str,
        *,
        want_forces: bool = False,
        max_neighbors: int = 50,
        min_image_cutoff: float | None = None,
    ) -> "ModelEvalNeed":
        return cls(
            tuple(str(i) for i in inputs),
            str(model),
            backend,
            want_forces,
            int(max_neighbors),
            None if min_image_cutoff is None else float(min_image_cutoff),
        )


def score_atoms(
    model_file,
    backend_type,
    atoms_list,
    *,
    want_forces: bool = False,
    max_neighbors: int = 50,
    min_image_cutoff: float | None = None,
):
    """Score configs with a trained model through the jaxrens backend.

    Loads the model (a mace-jax bundle *dir* via ``model_path=`` or a neuralil
    ``.pkl`` via ``pickle_file=``), builds the symbol map, and evaluates each
    structure. Returns ``(energies, forces)``: ``energies`` a list of total
    energies (eV); ``forces`` a list of ``(n_atoms, 3)`` arrays when
    ``want_forces`` else ``None``.

    ``min_image_cutoff`` (Å): when set, each structure is first replicated into
    the smallest supercell hosting the cutoff (jaxrens' neighbour list is
    minimum-image, valid only when the cell hosts it). The returned energy is
    scaled back to the *input* cell's atom count (``E_super / replication``), so
    a caller that divides by the input ``len(atoms)`` still gets the per-atom
    energy -- this is what lets an EOS reference stay on cheap unit cells while
    the model is scored on a cutoff-hosting cell. Forces are per-atom already,
    so expanding then mapping them back is ambiguous; ``min_image_cutoff`` with
    ``want_forces`` is rejected.

    The single home for the backend-load + walker-eval loop shared by the EOS
    and model-parity reader panels (local re-score) and the ``model_eval``
    cluster processor. jaxrens is imported lazily, so importing this module
    stays cheap on the dashboard path.
    """
    import numpy as np
    from ase.data import chemical_symbols
    from jaxrens.backends.base import eval_energy_and_forces
    from jaxrens.backends.loader import load_backend
    from jaxrens.io.formats import ase_atoms_to_walker

    if min_image_cutoff is not None and want_forces:
        raise ValueError(
            "score_atoms: min_image_cutoff is not supported with want_forces "
            "(per-atom forces cannot be mapped back from a supercell)."
        )

    if backend_type == "mace":
        backend = load_backend("mace", model_path=model_file)
    else:
        backend = load_backend(backend_type, pickle_file=model_file)

    symbol_map = getattr(backend, "symbol_map", None)
    if symbol_map is None:
        symbol_map = {
            chemical_symbols[int(z)]: i
            for i, z in enumerate(backend.atomic_numbers)
        }

    energies = []
    forces = [] if want_forces else None
    for atoms in atoms_list:
        scale = 1
        if min_image_cutoff is not None:
            from jaxrens_al.supercell import min_image_supercell

            n_orig = len(atoms)
            atoms = min_image_supercell(atoms, min_image_cutoff)
            scale = len(atoms) // n_orig
        w = ase_atoms_to_walker(atoms, symbol_map)
        if want_forces:
            res = eval_energy_and_forces(
                backend, w.positions, w.types, w.cell, max_neighbors
            )
            energies.append(float(res.energy))
            forces.append(np.asarray(res.forces, dtype=float))
        else:
            res = backend(w.positions, w.types, w.cell, max_neighbors)
            energies.append(float(res[0]) / scale)
    return energies, forces


def model_eval_keys(model) -> tuple:
    """The ``(energy_key, forces_key)`` a model's predictions are written under.

    Derived from the model path so a panel and the resolver agree without
    coordination, and two panels sharing a model read the same annotation. The
    hash keeps distinct iterations' models from colliding on one file.
    """
    h = hashlib.sha1(str(model).encode()).hexdigest()[:10]
    return f"model_energy_{h}", f"model_forces_{h}"


class ModelEvalConsumer:
    """Mixin for a panel whose model predictions are computed on the cluster.

    Override :meth:`model_eval_needs` to declare what scoring the panel needs
    and :meth:`receive_model_eval` to pick up the resolved annotated path(s). A
    panel that evaluates locally (e.g. a CPU NeuralIL model) declares nothing.
    """

    def model_eval_needs(self) -> list:
        """The model evaluation this panel needs (default: none)."""
        return []

    def receive_model_eval(self, mapping: dict) -> None:
        """Receive ``{input_file -> annotated_file}`` from the resolver."""
