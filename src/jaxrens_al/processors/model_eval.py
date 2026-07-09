"""The ``model_eval`` processor -- score configs with a trained MLIP remotely.

A downstream processor for ``remote_handler``'s trajectory-processing framework
(mirroring :mod:`jaxrens_al.processors.steinhardt`). On the cluster it loads a
trained model through the jaxrens backend and annotates each frame with the
model's energy (and optionally per-atom forces), so a GPU-only model (a MACE-jax
bundle needing ``cuequivariance``) is evaluated on a GPU node instead of in the
CPU dashboard process.

Both the EOS panel (energy curve) and
:class:`~jaxrens_al.panels.training.ModelParityPanel` (re-scored parity) read
the result back -- one processor serves both. The predictions land under
per-model ``info``/``arrays`` keys (:func:`jaxrens_al.panels.model_eval.
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
from dataclasses import dataclass
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


# --- cluster-side compute --------------------------------------------------
def run(atoms_list, args, calcpath=None):
    """Annotate each structure with the trained model's energy (+ forces).

    Writes ``atoms.info[energy_key]`` (total energy, eV) and, when
    ``want_forces``, ``atoms.arrays[forces_key]`` (``(n_atoms, 3)``) -- the keys
    the EOS / model-parity reader panels expect.
    """
    import numpy as np
    from ase.data import chemical_symbols
    from jaxrens.backends.base import eval_energy_and_forces
    from jaxrens.backends.loader import load_backend
    from jaxrens.io.formats import ase_atoms_to_walker

    backend_type = args["backend"]
    model_path = _materialise_model(args, calcpath)
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

    energy_key = args["energy_key"]
    forces_key = args.get("forces_key", "model_forces")
    want_forces = bool(args.get("want_forces", False))
    max_neighbors = int(args.get("max_neighbors", 50))

    for atoms in atoms_list:
        w = ase_atoms_to_walker(atoms, symbol_map)
        if want_forces:
            res = eval_energy_and_forces(
                backend, w.positions, w.types, w.cell, max_neighbors
            )
            atoms.info[energy_key] = float(res.energy)
            atoms.new_array(forces_key, np.asarray(res.forces, dtype=float))
        else:
            res = backend(w.positions, w.types, w.cell, max_neighbors)
            atoms.info[energy_key] = float(res[0])
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
        max_neighbors: Static neighbour-buffer size for the backend call.
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
    ):
        self.model = Path(model)
        self.backend = backend
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.want_forces = bool(want_forces)
        self.max_neighbors = int(max_neighbors)
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
        ).to_dict()


register("model_eval", run, schema=ModelEvalArgs)
