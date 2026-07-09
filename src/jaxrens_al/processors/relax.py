"""The ``relax`` processor -- relax each structure with a trained MLIP remotely.

A downstream processor for ``remote_handler``'s trajectory-processing framework
(mirroring :mod:`jaxrens_al.processors.model_eval`). On the cluster it loads a
trained model through the jaxrens backend, wraps it in a minimal ASE calculator,
and LBFGS-relaxes each frame to its local minimum -- the basin representative the
``spacegroup`` step then labels. Unlike ``remote_handler``'s built-in
``optimize`` (a NeuralIL-pickle / LJ calculator), this drives the **mace-jax**
bundle a jaxrens AL run trains, so a sample is relaxed with the very model that
generated it.

Writes the relaxed positions in place and ``atoms.info["is_converged"]``. The
model bundle is staged as a single ``.tar`` (a mace-jax bundle is a *directory*),
untarred cluster-side. ``jaxrens`` is imported lazily so importing this module
host-side is cheap; ``jaxrens_al`` must be installed in the cluster analysis env.
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
    file (a neuralil ``.pkl``). ``fmax`` (eV/A) and ``steps`` are the LBFGS
    stopping criteria.
    """

    model_remote: str = ""
    backend: str = "mace"
    is_archive: bool = True
    fmax: float = 0.05
    steps: int = 200
    max_neighbors: int = 50

    def validate(self, root: Optional[Path] = None) -> None:
        if not self.model_remote:
            raise ScriptInputError("relax: 'model_remote' is required")
        if not self.backend:
            raise ScriptInputError("relax: 'backend' is required")
        if not isinstance(self.fmax, (int, float)) or self.fmax <= 0:
            raise ScriptInputError("relax: 'fmax' must be a positive number")
        if not isinstance(self.steps, int) or self.steps <= 0:
            raise ScriptInputError("relax: 'steps' must be a positive int")


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


def _jaxrens_calculator(model_path, backend_type, max_neighbors):
    """A minimal ASE calculator backed by a jaxrens energy/force backend."""
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes
    from ase.data import chemical_symbols
    from jaxrens.backends.base import eval_energy_and_forces
    from jaxrens.backends.loader import load_backend
    from jaxrens.io.formats import ase_atoms_to_walker

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

    class _JaxrensCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self, atoms=None, properties=None, system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            w = ase_atoms_to_walker(atoms, symbol_map)
            res = eval_energy_and_forces(
                backend, w.positions, w.types, w.cell, max_neighbors
            )
            self.results = {
                "energy": float(res.energy),
                "forces": np.asarray(res.forces, dtype=float),
            }

    return _JaxrensCalculator()


def run(atoms_list, args, calcpath=None):
    """LBFGS-relax each structure with the trained model; set ``is_converged``.

    Builds the calculator lazily on first use, so a shard whose frames are all
    already converged (``is_converged`` in info) never loads the model.
    """
    from ase.optimize import LBFGS

    fmax = float(args["fmax"])
    steps = int(args["steps"])
    calc = None
    out = []
    for at in atoms_list:
        if "is_converged" in at.info:
            out.append(at)
            continue
        if calc is None:
            model_path = _materialise_model(args, calcpath)
            calc = _jaxrens_calculator(
                model_path, args["backend"], int(args.get("max_neighbors", 50))
            )
        relaxed = at.copy()
        relaxed.calc = calc
        converged = LBFGS(relaxed, logfile=None).run(fmax=fmax, steps=steps)
        relaxed.info["is_converged"] = bool(converged)
        out.append(relaxed)
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
        fmax / steps: LBFGS force tolerance (eV/A) and max iterations.
        max_neighbors: Static neighbour-buffer size for the backend call.
    """

    name = "relax"

    def __init__(
        self,
        model,
        *,
        backend: str = "mace",
        fmax: float = 0.05,
        steps: int = 200,
        max_neighbors: int = 50,
    ):
        self.model = Path(model)
        self.backend = backend
        self.fmax = float(fmax)
        self.steps = int(steps)
        self.max_neighbors = int(max_neighbors)
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
            fmax=self.fmax,
            steps=self.steps,
            max_neighbors=self.max_neighbors,
        ).to_dict()


register("relax", run, schema=RelaxArgs)
