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

import functools
import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


def _pick_bucket(true_max, ladder, offset: int, above: int = -1) -> int:
    """Smallest ladder entry fitting ``true_max + offset``, above ``above``.

    Mirrors jaxrens' ``bucket_manager._pick_next_bucket``: restricting the
    neighbour-buffer size to a fixed ladder bounds the number of distinct XLA
    compilations to ``len(ladder)``, where sizing each batch exactly would
    recompile on nearly every batch. ``offset`` is headroom so a small
    geometry change does not immediately trip an overflow retry.
    """
    target = int(true_max) + int(offset)
    for b in ladder:
        if b >= target and b > above:
            return int(b)
    raise RuntimeError(
        f"score_atoms: neighbour bucket ladder exhausted -- observed max "
        f"neighbour count {int(true_max)} (+ offset {offset}) needs a bucket "
        f">= {target} and > {above}, but max_neighbors_list={list(ladder)} "
        f"has no such entry. Extend the ladder."
    )


def _score_batched(
    backend,
    walkers,
    scales,
    *,
    want_forces: bool,
    max_neighbors: int,
    batch_size: int,
    ladder,
    offset: int,
):
    """``jax.vmap`` the backend over stacked configs, one bucket per batch.

    The backend contract is explicitly per-walker ("Operates on a SINGLE
    walker. Batched via vmap/pmap by the sampler wrapper" -- ``jaxrens.base``),
    and ``BackendResult.overflow`` is already typed with a batch axis, so this
    is the intended way to evaluate many configs at once.

    Configs are grouped by atom count (a batch has to stack to ``(B, N, 3)``),
    chunked to ``batch_size``, and each chunk is padded up to the full batch so
    a ragged tail does not trigger a second compilation. When ``ladder`` is set
    the bucket is sized per chunk from geometry alone via the backend's
    ``max_neighbors_for`` (no GNN forward pass), and an overflow reported by
    the forward pass grows the bucket and retries -- the same
    escalate-and-retry the NS loop does.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaxrens.backends.base import eval_energy_and_forces

    energies = [0.0] * len(walkers)
    forces = [None] * len(walkers) if want_forces else None

    groups: dict[int, list[int]] = {}
    for i, w in enumerate(walkers):
        groups.setdefault(int(np.asarray(w.positions).shape[0]), []).append(i)

    sizer = getattr(backend, "max_neighbors_for", None)
    v_sizer = (
        jax.vmap(lambda p, c: sizer(p, c))
        if (ladder and sizer is not None)
        else None
    )
    if ladder and sizer is None:
        logger.warning(
            "score_atoms: backend has no max_neighbors_for(); falling back to "
            "the fixed max_neighbors=%d instead of the bucket ladder.",
            max_neighbors,
        )

    @functools.lru_cache(maxsize=None)
    def _vmapped(bucket: int):
        # Cached per bucket so revisiting a bucket reuses the JAX compilation.
        def one(p, t, c):
            if want_forces:
                r = eval_energy_and_forces(backend, p, t, c, bucket)
                f = r.forces
            else:
                r = backend(p, t, c, bucket)
                f = jnp.zeros_like(p)
            return (
                r.energy,
                f,
                jnp.asarray(r.overflow),
                jnp.asarray(r.max_neighbor_count),
            )

        return jax.vmap(one)

    for idx in groups.values():
        for start in range(0, len(idx), batch_size):
            chunk = idx[start : start + batch_size]
            # Pad to a full batch by repeating the last config; the duplicates
            # are discarded below and cannot change the bucket sizing.
            sel = chunk + [chunk[-1]] * (batch_size - len(chunk))
            pos = jnp.stack([jnp.asarray(walkers[i].positions) for i in sel])
            typ = jnp.stack([jnp.asarray(walkers[i].types) for i in sel])
            cel = jnp.stack([jnp.asarray(walkers[i].cell) for i in sel])

            bucket = int(max_neighbors)
            if v_sizer is not None:
                bucket = _pick_bucket(
                    int(jnp.max(v_sizer(pos, cel))), ladder, offset
                )

            while True:
                e, f, over, seen = _vmapped(bucket)(pos, typ, cel)
                if not bool(jnp.any(over)):
                    break
                if v_sizer is None:
                    logger.warning(
                        "score_atoms: neighbour overflow at max_neighbors=%d "
                        "and no ladder configured -- results are truncated. "
                        "Pass max_neighbors_list= to size the bucket.",
                        bucket,
                    )
                    break
                grown = _pick_bucket(
                    int(jnp.max(seen)), ladder, offset, above=bucket
                )
                logger.info(
                    "score_atoms: neighbour overflow at bucket %d -> %d",
                    bucket,
                    grown,
                )
                bucket = grown

            for k, i in enumerate(chunk):
                energies[i] = float(e[k]) / scales[i]
                if want_forces:
                    forces[i] = np.asarray(f[k], dtype=float)
    return energies, forces


def score_atoms(
    model_file,
    backend_type,
    atoms_list,
    *,
    want_forces: bool = False,
    max_neighbors: int = 50,
    min_image_cutoff: float | None = None,
    batch_size: int | None = None,
    max_neighbors_list=None,
    max_neighbors_offset: int = 4,
):
    """Score configs with a trained model through the jaxrens backend.

    Loads the model (a mace-jax bundle *dir* via ``model_path=`` or a neuralil
    ``.pkl`` via ``pickle_file=``), builds the symbol map, and evaluates each
    structure. Returns ``(energies, forces)``: ``energies`` a list of total
    energies (eV); ``forces`` a list of ``(n_atoms, 3)`` arrays when
    ``want_forces`` else ``None``.

    ``batch_size`` (configs per ``jax.vmap`` call): unset evaluates one config
    per backend call, which leaves a GPU almost idle on small cells. Setting it
    stacks configs into ``(B, N, 3)`` batches -- configs are grouped by atom
    count first, since a batch must stack. Order of the returned lists always
    follows ``atoms_list``.

    ``max_neighbors_list`` (a bucket ladder, e.g. ``[20, 30, 40, 50, 60]``):
    unset pins the neighbour buffer to ``max_neighbors`` for every config --
    simple, but oversized for dense cells and silently truncating for sparse
    ones. Set it and the buffer is sized per batch from the geometry alone
    (the backend's ``max_neighbors_for``, no forward pass), snapped up to the
    smallest ladder entry with ``max_neighbors_offset`` slots of headroom, and
    grown on an overflow report. Restricting sizes to the ladder is what keeps
    the number of XLA compilations bounded; sizing each batch exactly would
    recompile constantly. Only used with ``batch_size``.

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

    walkers, scales = [], []
    for atoms in atoms_list:
        scale = 1
        if min_image_cutoff is not None:
            from jaxrens_al.supercell import min_image_supercell

            n_orig = len(atoms)
            atoms = min_image_supercell(atoms, min_image_cutoff)
            scale = len(atoms) // n_orig
        walkers.append(ase_atoms_to_walker(atoms, symbol_map))
        scales.append(scale)

    if batch_size is not None and int(batch_size) > 1:
        return _score_batched(
            backend,
            walkers,
            scales,
            want_forces=want_forces,
            max_neighbors=int(max_neighbors),
            batch_size=int(batch_size),
            ladder=(
                tuple(sorted(int(b) for b in max_neighbors_list))
                if max_neighbors_list
                else None
            ),
            offset=int(max_neighbors_offset),
        )

    energies = []
    forces = [] if want_forces else None
    for w, scale in zip(walkers, scales):
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
