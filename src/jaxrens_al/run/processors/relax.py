"""The ``relax`` processor -- relax each structure with a trained MLIP remotely.

A downstream processor for ``remote_handler``'s trajectory-processing framework
(mirroring :mod:`jaxrens_al.run.processors.model_eval`). On the cluster it loads a
trained model through the jaxrens backend and relaxes each frame to its local
minimum -- the basin representative the ``spacegroup`` step then labels. Unlike
``remote_handler``'s built-in ``optimize`` (a NeuralIL-pickle / LJ calculator),
this drives the **mace-jax** bundle a jaxrens AL run trains, so a sample is
relaxed with the very model that generated it.

The relaxation is done **entirely in jax**: the whole optimizer loop is one
compiled ``lax.while_loop`` (no Python-per-step, no host round-trips), and frames
are ``vmap``-batched by atom count so a GPU relaxes a whole bucket in one kernel
launch rather than one structure at a time. Two optimizers share that harness --
``method="fire"`` (jax-md's FIRE) or ``method="lbfgs"`` (optax's L-BFGS with a
zoom line search); both stop on the same force criterion ``max|dE/dx| < fmax``.

Writes the relaxed positions in place (cell unchanged -- positions only),
``atoms.info["is_converged"]`` and ``atoms.info["relax_energy"]`` (the model
potential energy at the minimum, eV). The model bundle is staged as a single
``.tar`` (a mace-jax bundle is a *directory*), untarred cluster-side. ``jaxrens`` / ``jax``
are imported lazily so importing this module host-side is cheap; ``jaxrens_al``
(and ``jax``/``jax_md``/``optax``) must be installed in the cluster analysis env.
"""

from __future__ import annotations

import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing.base import Processor
from remote_handler.processing.registry import register
from remote_handler.scripts.input_models import ScriptInput


# --- arg schema ------------------------------------------------------------
@dataclass
class RelaxArgs(ScriptInput):
    """Args for the ``relax`` processor.

    ``model_remote`` is the staged model on the cluster -- a ``.tar`` of a
    mace-jax bundle dir when ``is_archive`` (untarred before loading) or a plain
    file (a neuralil ``.pkl``). ``fmax`` (eV/A) and ``steps`` are the optimizer
    stopping criteria.

    ``batch_size`` caps how many structures are relaxed in one ``vmap`` kernel
    (GPU-memory bound, independent of the shard size): a shard's frames are
    relaxed in sub-batches of at most ``batch_size``. ~20-50 suits a MACE model.

    The neighbor-buffer size is chosen **dynamically per atom-count bucket**,
    exactly as jaxrens does: the backend's geometry-only ``max_neighbors_for``
    probe gives each structure's true neighbor count, and the smallest entry of
    ``max_neighbors_list`` that covers ``peak + max_neighbors_offset`` is used;
    if the *relaxed* geometry densifies past that buffer the bucket grows one
    ladder rung and the relaxation reruns. ``max_neighbors`` is only the
    fallback buffer for backends without the probe.
    """

    model_remote: str = ""
    backend: str = "mace"
    is_archive: bool = True
    method: str = "lbfgs"
    fmax: float = 0.05
    steps: int = 200
    batch_size: int = 32
    max_neighbors: int = 50
    max_neighbors_list: tuple = (30, 35, 40, 45, 50)
    max_neighbors_offset: int = 5

    def validate(self, root: Optional[Path] = None) -> None:
        if not self.model_remote:
            raise ScriptInputError("relax: 'model_remote' is required")
        if not self.backend:
            raise ScriptInputError("relax: 'backend' is required")
        if self.method not in ("fire", "lbfgs"):
            raise ScriptInputError(
                "relax: 'method' must be 'fire' or 'lbfgs', "
                f"got {self.method!r}"
            )
        if not isinstance(self.fmax, (int, float)) or self.fmax <= 0:
            raise ScriptInputError("relax: 'fmax' must be a positive number")
        if not isinstance(self.steps, int) or self.steps <= 0:
            raise ScriptInputError("relax: 'steps' must be a positive int")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ScriptInputError(
                "relax: 'batch_size' must be a positive int"
            )
        if not self.max_neighbors_list:
            raise ScriptInputError(
                "relax: 'max_neighbors_list' (bucket ladder) must be non-empty"
            )


# --- cluster-side compute --------------------------------------------------
def _materialise_model(args, calcpath):
    """Return a local model path, untarring a staged bundle archive first."""
    remote = args["model_remote"]
    if not args.get("is_archive", True):
        return remote
    dest = Path(calcpath or ".") / "model_bundle"
    with tarfile.open(remote, "r") as tar:
        tar.extractall(dest)
    entries = [p for p in dest.iterdir() if p.is_dir()]
    return str(entries[0] if len(entries) == 1 else dest)


def _load_backend(model_path, backend_type):
    """Load a jaxrens energy backend and its ``{symbol: type}`` map."""
    from ase.data import chemical_symbols
    from jaxrens.backends.loader import load_backend

    if backend_type == "mace":
        backend = load_backend("mace", model_path=model_path)
    else:
        backend = load_backend(backend_type, pickle_file=model_path)

    symbol_map = getattr(backend, "symbol_map", None)
    if symbol_map is None:
        symbol_map = {
            chemical_symbols[int(z)]: i
            for i, z in enumerate(backend.atomic_numbers)
        }
    return backend, symbol_map


def _pick_next_bucket(true_max, current, ladder, offset):
    """Smallest ladder entry that covers ``true_max + offset`` and beats
    ``current`` -- mirrors jaxrens' bucket-growth rule
    (``jaxrens.sampling.bucket_manager._pick_next_bucket``), inlined to avoid
    depending on a private symbol. Raises when the ladder can't accommodate the
    need (user-actionable: extend the ladder).
    """
    target = int(true_max) + int(offset)
    for b in ladder:
        if b >= target and b > current:
            return b
    raise RuntimeError(
        f"relax: no neighbor bucket in {list(ladder)} covers "
        f"{int(true_max)} + offset {int(offset)} = {target} (and beats "
        f"current {int(current)}). Extend 'max_neighbors_list'."
    )


def _probe_peak(backend, positions, cell):
    """True max neighbor count over a batch (geometry only), or ``None``.

    Uses the backend's ``max_neighbors_for`` probe (mace / neuralil) so the
    bucket is sized to the neighbors the forward pass will actually see, without
    a GNN evaluation. ``None`` when the backend has no probe (fixed-buffer or
    neighbor-free backends) -- callers fall back to a static ``max_neighbors``.
    """
    import jax
    import jax.numpy as jnp

    probe = getattr(backend, "max_neighbors_for", None)
    if probe is None:
        return None
    counts = jax.vmap(probe)(positions, cell)  # (B, N) per-atom counts
    return int(jnp.max(counts))


def _make_relax_batch(backend, fmax, steps, method):
    """Compile a ``vmap``-batched relaxation.

    Returns a jitted ``(pos, types, cell, max_neighbors) -> (relaxed_pos,
    converged, energy)``, where ``energy`` is the model potential energy at the
    local minimum. ``max_neighbors`` (the neighbor-buffer size) is a
    ``static_argnums`` argument, so jax's own compilation cache keys on
    ``(max_neighbors, input shapes)`` -- one compile per (buffer, atom-count)
    pair, reused automatically. No hand-rolled cache needed.

    The whole optimizer loop is one ``lax.while_loop`` (compiled, no Python per
    step), ``vmap``-ed over the leading batch axis so a bucket of same-``N``
    frames relaxes in a single kernel launch. ``method`` picks FIRE (jax-md) or
    L-BFGS (optax); both stop at ``max|dE/dx| < fmax`` or ``steps`` iterations.
    Positions only -- the cell is held fixed.
    """
    import jax
    import jax.numpy as jnp

    def relax_one(positions, types, cell, max_neighbors):
        def energy(pos):
            return backend(pos, types, cell, max_neighbors).energy

        if method == "fire":
            from jax_md.minimize import fire_descent

            init, apply = fire_descent(energy, lambda R, dR, **kw: R + dR)

            def cond(carry):
                i, state = carry
                return (i < steps) & (jnp.max(jnp.abs(state.force)) > fmax)

            def body(carry):
                i, state = carry
                return i + 1, apply(state)

            _, state = jax.lax.while_loop(cond, body, (0, init(positions)))
            final_pos = state.position
            converged = jnp.max(jnp.abs(state.force)) <= fmax
        else:  # lbfgs
            import optax

            opt = optax.lbfgs()
            value_and_grad = optax.value_and_grad_from_state(energy)

            def cond(carry):
                i, pos, _ = carry
                grad = jax.grad(energy)(pos)
                return (i < steps) & (jnp.max(jnp.abs(grad)) > fmax)

            def body(carry):
                i, pos, state = carry
                value, grad = value_and_grad(pos, state=state)
                updates, state = opt.update(
                    grad,
                    state,
                    pos,
                    value=value,
                    grad=grad,
                    value_fn=energy,
                )
                pos = optax.apply_updates(pos, updates)
                return i + 1, pos, state

            _, final_pos, _ = jax.lax.while_loop(
                cond, body, (0, positions, opt.init(positions))
            )
            converged = jnp.max(jnp.abs(jax.grad(energy)(final_pos))) <= fmax

        return final_pos, converged, energy(final_pos)

    # vmap over the batch axis; max_neighbors stays unbatched + static so jax
    # caches one compiled kernel per (buffer, atom-count) pair.
    batched = jax.vmap(relax_one, in_axes=(0, 0, 0, None))
    return jax.jit(batched, static_argnums=(3,))


def run(atoms_list, args, calcpath=None):
    """Relax each structure with the trained model in jax; set ``is_converged``.

    Loads the backend lazily, so a shard whose frames are all already converged
    (``is_converged`` in info) never touches jax. Frames are bucketed by atom
    count (fixed ``vmap`` shape) and each bucket is relaxed in sub-batches of at
    most ``batch_size`` -- so the ``vmap`` width (GPU memory) is bounded
    independently of the shard size. The neighbor buffer for a sub-batch is
    sized dynamically from its structures via the backend probe (jaxrens' ladder
    + offset), and grown one rung with a rerun if the relaxed geometry densifies
    past it -- the relaxer is compiled once per (atom-count, buffer) pair (jax's
    own cache), so distinct compilations are bounded by the ladder length.
    """
    import numpy as np

    from jaxrens_al.keys import RELAX_ENERGY_KEY

    fmax = float(args["fmax"])
    steps = int(args["steps"])
    method = str(args.get("method", "lbfgs"))
    batch_size = int(args.get("batch_size", 32))
    fallback_nb = int(args.get("max_neighbors", 50))
    ladder = tuple(int(x) for x in args.get("max_neighbors_list", ()) or ())
    offset = int(args.get("max_neighbors_offset", 5))

    def _log(msg):
        # Cluster-side progress -> the array job's stdout (SLURM .out), like
        # the processing driver's own print()s. flush so lines stream live.
        print(f"[relax] {msg}", flush=True)

    todo = [
        i for i, at in enumerate(atoms_list) if "is_converged" not in at.info
    ]
    out = list(atoms_list)
    skipped = len(atoms_list) - len(todo)
    if not todo:
        _log(
            f"all {len(atoms_list)} structure(s) already converged; nothing "
            "to do"
        )
        return out

    import jax.numpy as jnp
    from jaxrens.io.formats import ase_atoms_to_walker

    _log(
        f"{len(todo)} structure(s) to relax ({skipped} already converged); "
        f"method={method}, fmax={fmax}, steps={steps}, "
        f"batch_size={batch_size}"
    )

    backend, symbol_map = _load_backend(
        _materialise_model(args, calcpath), args["backend"]
    )

    # One jitted relaxer; the neighbor buffer is a static arg, so jax compiles
    # (and caches) one kernel per (buffer, atom-count) pair on its own.
    relax_batch = _make_relax_batch(backend, fmax, steps, method)

    walkers = {i: ase_atoms_to_walker(atoms_list[i], symbol_map) for i in todo}

    def _relax_chunk(chunk):
        """Relax one sub-batch (same atom count); grow buffer on overflow."""
        pos = jnp.stack([walkers[i].positions for i in chunk])
        types = jnp.stack([walkers[i].types for i in chunk])
        cell = jnp.stack([walkers[i].cell for i in chunk])

        # Size the neighbor buffer to this sub-batch (jaxrens rule); no probe
        # (fixed-buffer backend) -> the static fallback.
        peak = _probe_peak(backend, pos, cell) if ladder else None
        buffer = (
            _pick_next_bucket(peak, 0, ladder, offset)
            if peak is not None
            else fallback_nb
        )
        while True:
            relaxed_pos, converged, energy = relax_batch(
                pos, types, cell, buffer
            )
            if peak is None:
                break
            # Relaxation may densify a basin past the buffer -> grow one rung
            # and rerun from the original geometry (the overflowed run's forces
            # were truncated and can't be trusted).
            final_peak = _probe_peak(backend, relaxed_pos, cell)
            if final_peak <= buffer:
                break
            grown = _pick_next_bucket(final_peak, buffer, ladder, offset)
            _log(
                f"  neighbor overflow (peak {final_peak} > buffer {buffer}) "
                f"-> grew to {grown}, rerunning sub-batch"
            )
            buffer = grown
        return (
            np.asarray(relaxed_pos),
            np.asarray(converged),
            np.asarray(energy),
            buffer,
        )

    # Bucket by atom count (fixed vmap shape), then sub-batch by batch_size.
    buckets: dict = {}
    for i in todo:
        buckets.setdefault(len(atoms_list[i]), []).append(i)

    n_chunks = sum(
        (len(v) + batch_size - 1) // batch_size for v in buckets.values()
    )
    done = 0
    n_converged = 0
    for n_atoms, idxs in buckets.items():
        for j in range(0, len(idxs), batch_size):
            chunk = idxs[j : j + batch_size]
            done += 1
            _log(
                f"sub-batch {done}/{n_chunks}: {len(chunk)} struct x "
                f"{n_atoms} atoms"
            )
            relaxed_pos, converged, energy, buffer = _relax_chunk(chunk)
            n_ok = int(np.count_nonzero(converged))
            n_converged += n_ok
            _log(f"  done: {n_ok}/{len(chunk)} converged (buffer {buffer})")
            for k, i in enumerate(chunk):
                at = atoms_list[i].copy()
                at.set_positions(relaxed_pos[k])
                at.info["is_converged"] = bool(converged[k])
                at.info[RELAX_ENERGY_KEY] = float(energy[k])
                out[i] = at
    _log(
        f"finished: {n_converged}/{len(todo)} converged across {n_chunks} "
        "sub-batch(es)"
    )
    return out


# --- host-side spec --------------------------------------------------------
class RelaxProcessor(Processor):
    """Relax structures with one trained model (mace-jax bundle) on the cluster.

    Stages the model (a mace-jax bundle dir is tarred; a plain file as-is) and
    contributes the LBFGS args; the compute is :func:`run`.

    Args:
        model: Trained-model bundle -- a mace-jax bundle *directory* or a
            neuralil ``.pkl`` file.
        backend: jaxrens backend to load it with (``"mace"`` / ``"neuralil"``).
        method: jax optimizer -- ``"fire"`` (jax-md) or ``"lbfgs"`` (optax).
        fmax / steps: force tolerance (eV/A) and max iterations.
        batch_size: Max structures per ``vmap`` kernel (~20-50 for MACE); caps
            GPU memory independently of the shard size.
        max_neighbors_list / max_neighbors_offset: neighbor-bucket ladder and
            headroom -- the buffer is sized dynamically per atom-count bucket
            (jaxrens' rule), growing a rung if the relaxed basin densifies.
        max_neighbors: Fallback static buffer for backends without the probe.
    """

    name = "relax"

    def __init__(
        self,
        model,
        *,
        backend: str = "mace",
        method: str = "lbfgs",
        fmax: float = 0.05,
        steps: int = 200,
        batch_size: int = 32,
        max_neighbors: int = 50,
        max_neighbors_list=(30, 35, 40, 45, 50),
        max_neighbors_offset: int = 5,
    ):
        self.model = Path(model)
        self.backend = backend
        self.method = method
        self.fmax = float(fmax)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.max_neighbors = int(max_neighbors)
        self.max_neighbors_list = tuple(int(x) for x in max_neighbors_list)
        self.max_neighbors_offset = int(max_neighbors_offset)
        self._is_archive = self.model.is_dir()
        self._staged: Optional[Path] = None

    def _archive(self) -> Path:
        if self._staged is None:
            tmp = Path(tempfile.mkdtemp(prefix="relax_"))
            archive = tmp / (self.model.name + ".tar")
            with tarfile.open(archive, "w") as tar:
                tar.add(self.model, arcname=self.model.name)
            self._staged = archive
        return self._staged

    def shared_files(self) -> list:
        if self._is_archive:
            return [str(self._archive())]
        return [str(self.model)]

    def build_args(self, savepath: str) -> dict:
        staged = self._archive() if self._is_archive else self.model
        remote = str(Path(savepath) / Path(staged).name)
        return RelaxArgs(
            model_remote=remote,
            backend=self.backend,
            is_archive=self._is_archive,
            method=self.method,
            fmax=self.fmax,
            steps=self.steps,
            batch_size=self.batch_size,
            max_neighbors=self.max_neighbors,
            max_neighbors_list=self.max_neighbors_list,
            max_neighbors_offset=self.max_neighbors_offset,
        ).to_dict()


register("relax", run, schema=RelaxArgs)
