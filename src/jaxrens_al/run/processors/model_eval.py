"""The ``model_eval`` processor -- score configs with a trained MLIP remotely.

A downstream processor for ``remote_handler``'s trajectory-processing framework
(mirroring its built-in ``rdf`` processor). On the cluster it loads a
trained model through the jaxrens backend and annotates each frame with the
model's energy (and optionally per-atom forces), so a GPU-only model (a MACE-jax
bundle needing ``cuequivariance``) is evaluated on a GPU node instead of in the
CPU dashboard process.

Both the EOS panel (energy curve) and
:class:`~jaxrens_al.dashboard.panels.training.ModelParityPanel` (re-scored parity) read
the result back -- one processor serves both. The predictions land under
per-model ``info``/``arrays`` keys (:func:`jaxrens_al.dashboard.panels.model_eval.
model_eval_keys`) so distinct iterations' models never collide on a shared
config file.

The model bundle is staged as a single ``.tar`` (a mace-jax bundle is a
*directory*; ``shared_files`` copies plain files), untarred cluster-side before
the load. ``jaxrens`` is imported lazily so importing this module host-side is
cheap; ``jaxrens_al`` must be installed in the cluster analysis env.
"""

from __future__ import annotations

import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing.base import Processor
from remote_handler.processing.registry import register
from remote_handler.scripts.input_models import ScriptInput


# --- arg schema ------------------------------------------------------------
@dataclass
class ModelEvalArgs(ScriptInput):
    """Args for the ``model_eval`` processor.

    ``model_remote`` is the staged model on the cluster -- a ``.tar`` of a
    mace-jax bundle dir when ``is_archive`` (untarred before loading) or a plain
    file (a neuralil ``.pkl``) otherwise. ``backend`` selects the jaxrens
    backend; predictions are written under ``energy_key`` (and ``forces_key``
    when ``want_forces``).
    """

    model_remote: str = ""
    backend: str = "mace"
    is_archive: bool = True
    energy_key: str = "model_energy"
    forces_key: str = "model_forces"
    want_forces: bool = False
    max_neighbors: int = 50
    # 0.0 -> off; >0 -> score each frame on the smallest supercell hosting this
    # cutoff (Å), scaling the energy back to the input cell's atom count.
    min_image_cutoff: float = 0.0
    # Forwarded to score_atoms (vmapped batching + the neighbour-bucket
    # ladder); an empty ladder keeps the buffer fixed at max_neighbors.
    batch_size: Optional[int] = None
    max_neighbors_list: list = field(default_factory=list)
    max_neighbors_offset: int = 4

    def validate(self, root: Optional[Path] = None) -> None:
        if not self.model_remote:
            raise ScriptInputError("model_eval: 'model_remote' is required")
        if not self.backend:
            raise ScriptInputError("model_eval: 'backend' is required")
        if not self.energy_key:
            raise ScriptInputError("model_eval: 'energy_key' is required")
        if self.want_forces and not self.forces_key:
            raise ScriptInputError(
                "model_eval: 'forces_key' is required when want_forces"
            )
        if not isinstance(self.max_neighbors, int) or self.max_neighbors <= 0:
            raise ScriptInputError(
                "model_eval: 'max_neighbors' must be a positive int"
            )
        if self.batch_size is not None and (
            not isinstance(self.batch_size, int) or self.batch_size <= 0
        ):
            raise ScriptInputError(
                "model_eval: 'batch_size' must be a positive int or null"
            )
        ladder = list(self.max_neighbors_list)
        if ladder and (
            not all(isinstance(b, int) and b > 0 for b in ladder)
            or ladder != sorted(ladder)
        ):
            raise ScriptInputError(
                "model_eval: 'max_neighbors_list' must be positive ints "
                "in ascending order"
            )


# --- cluster-side compute --------------------------------------------------
def run(atoms_list, args, calcpath=None):
    """Annotate each structure with the trained model's energy (+ forces).

    Writes ``atoms.info[energy_key]`` (total energy, eV) and, when
    ``want_forces``, ``atoms.arrays[forces_key]`` (``(n_atoms, 3)``) -- the keys
    the EOS / model-parity reader panels expect.
    """
    from jaxrens_al.dashboard.panels.model_eval import score_atoms

    backend_type = args["backend"]
    model_path = _materialise_model(args, calcpath)
    energy_key = args["energy_key"]
    forces_key = args.get("forces_key", "model_forces")
    want_forces = bool(args.get("want_forces", False))
    max_neighbors = int(args.get("max_neighbors", 50))
    cutoff = float(args.get("min_image_cutoff", 0.0) or 0.0)
    batch_size = args.get("batch_size")
    ladder = list(args.get("max_neighbors_list") or [])
    offset = int(args.get("max_neighbors_offset", 4))

    energies, forces = score_atoms(
        model_path,
        backend_type,
        atoms_list,
        want_forces=want_forces,
        max_neighbors=max_neighbors,
        min_image_cutoff=cutoff if cutoff > 0 else None,
        batch_size=batch_size,
        max_neighbors_list=ladder or None,
        max_neighbors_offset=offset,
    )
    for i, atoms in enumerate(atoms_list):
        atoms.info[energy_key] = energies[i]
        if want_forces:
            atoms.new_array(forces_key, forces[i])
    return atoms_list


def _materialise_model(args, calcpath):
    """Return a local model path, untarring a staged bundle archive first."""
    remote = args["model_remote"]
    if not args.get("is_archive", True):
        return remote
    dest = Path(calcpath or ".") / "model_bundle"
    with tarfile.open(remote, "r") as tar:
        tar.extractall(dest)
    # The archive holds a single top-level dir (the bundle); descend into it.
    entries = [p for p in dest.iterdir() if p.is_dir()]
    return str(entries[0] if len(entries) == 1 else dest)


# --- host-side spec --------------------------------------------------------
class ModelEvalProcessor(Processor):
    """Score structures with one trained model on the cluster.

    A pure host-side spec: it stages the model (a mace-jax bundle dir is tarred;
    a plain file is staged as-is) and contributes the remote args; the compute
    is :func:`run`. ``registration_module`` defaults to this module, so the
    stock driver imports it and ``register`` runs.

    Args:
        model: Trained-model bundle -- a mace-jax bundle *directory* or a
            neuralil ``.pkl`` file.
        backend: jaxrens backend to load it with (``"mace"`` / ``"neuralil"``).
        energy_key / forces_key: ``info`` / ``arrays`` keys to write under.
        want_forces: Also compute per-atom forces (parity needs them).
        max_neighbors: Static neighbour-buffer size for the backend call
            (the starting bucket when ``max_neighbors_list`` is given).
        min_image_cutoff: When set (Å), score each frame on the smallest
            supercell hosting the cutoff, scaling the energy back to the input
            cell's atom count (for a unit-cell EOS reference); ``None`` scores
            as-is.
        batch_size: Configs per vmapped backend call (``None`` = one per call).
        max_neighbors_list: Neighbour-bucket ladder -- sizes the buffer from
            the geometry instead of pinning it at ``max_neighbors``, growing it
            on overflow, with the ladder bounding the compile count.
        max_neighbors_offset: Headroom above the observed neighbour count.
    """

    name = "model_eval"

    def __init__(
        self,
        model,
        *,
        backend: str = "mace",
        energy_key: str = "model_energy",
        forces_key: str = "model_forces",
        want_forces: bool = False,
        max_neighbors: int = 50,
        min_image_cutoff: float | None = None,
        batch_size: int | None = None,
        max_neighbors_list=None,
        max_neighbors_offset: int = 4,
    ):
        self.model = Path(model)
        self.backend = backend
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.want_forces = bool(want_forces)
        self.max_neighbors = int(max_neighbors)
        self.min_image_cutoff = (
            0.0 if min_image_cutoff is None else float(min_image_cutoff)
        )
        self.batch_size = None if batch_size is None else int(batch_size)
        self.max_neighbors_list = (
            sorted(int(b) for b in max_neighbors_list)
            if max_neighbors_list
            else []
        )
        self.max_neighbors_offset = int(max_neighbors_offset)
        self._is_archive = self.model.is_dir()
        self._staged: Optional[Path] = None

    def _archive(self) -> Path:
        """Tar the bundle directory once; cache the archive path."""
        if self._staged is None:
            tmp = Path(tempfile.mkdtemp(prefix="model_eval_"))
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
        return ModelEvalArgs(
            model_remote=remote,
            backend=self.backend,
            is_archive=self._is_archive,
            energy_key=self.energy_key,
            forces_key=self.forces_key,
            want_forces=self.want_forces,
            max_neighbors=self.max_neighbors,
            min_image_cutoff=self.min_image_cutoff,
            batch_size=self.batch_size,
            max_neighbors_list=self.max_neighbors_list,
            max_neighbors_offset=self.max_neighbors_offset,
        ).to_dict()


register("model_eval", run, schema=ModelEvalArgs)
