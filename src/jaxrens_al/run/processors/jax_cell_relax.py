"""The ``jax_cell_relax`` processor -- staged relaxation via a jaxrens backend.

The jax counterpart of :mod:`jaxrens_al.run.processors.torchsim_relax`, for
models torch-sim cannot drive -- a NeuralIL ``.pkl`` above all, but any
jaxrens backend works (a mace-jax bundle, ``lj`` for tests). Each structure is
relaxed in the same *stages* by default:

1. ``"positions"`` -- atomic positions at fixed cell (FIRE);
2. ``"cell"`` -- cell and positions together under the structure's pressure
   (FIRE + ASE's ``FrechetCellFilter``, the filter torch-sim's is a port of).

jaxrens backends return no stress, so it is taken by autodiff: every backend
is a jax function of positions and cell, and applying a symmetric strain
``eps`` to both gives ``stress = (1/V) dE/d eps`` at ``eps = 0`` (the forces
are ``-dE/dx`` of the same call). A small ASE calculator wraps that one jitted
energy/forces/stress function, so ASE's optimiser and cell filter do the rest,
one structure at a time.

The neighbour buffer follows the jaxrens rule: the backend's geometry-only
``max_neighbors_for`` probe sizes it from the ``max_neighbors_list`` ladder (+
``max_neighbors_offset`` headroom), and an overflow reported by the forward
pass grows it one rung and re-evaluates, so a basin that densifies during the
cell stage never sees a truncated neighbour list. ``supercell_trafo`` is the
backend's periodic-image expansion and must match the NS run's
(``backend.supercell_trafo`` -- e.g. ``(2, 2, 2)`` for 32-atom NeuralIL cells).

The pressure is read per structure from ``atoms.info[pressure_key]`` (GPa, 0
if absent). Writes the relaxed geometry back and annotates ``info`` with the
same keys as ``torchsim_relax`` -- ``relax_energy`` (eV), ``relax_enthalpy``
(eV/atom, ``(E + P V) / N``), ``relax_volume`` (A^3/atom), ``relax_fmax``
(eV/A), ``relax_pressure`` (GPa, from the final stress) and ``is_converged``
-- so the ``spacegroup`` step and the phase-grid panel read either. A
structure whose relaxation crashes is kept unrelaxed with ``relax_failed`` /
``relax_error`` set and ``is_converged`` False instead of failing the shard.

``jax`` / ``jaxrens`` are imported lazily, so importing this module host-side
is cheap; ``jaxrens_al`` (with the model's jaxrens backend deps) must be
installed in the cluster analysis env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing.base import Processor
from remote_handler.processing.registry import register
from remote_handler.scripts.input_models import ScriptInput

from jaxrens_al.keys import RELAX_ENERGY_KEY, RELAX_ERROR_KEY, RELAX_FAILED_KEY
from jaxrens_al.run.processors.relax import (
    _materialise_model,
    _pick_next_bucket,
)
from jaxrens_al.run.processors.torchsim_relax import (
    CONVERGED_KEY,
    RELAX_ENTHALPY_KEY,
    RELAX_FMAX_KEY,
    RELAX_PRESSURE_KEY,
    RELAX_VOLUME_KEY,
    STAGES,
)

BACKENDS = ("neuralil", "mace", "lj")
# The Lennard-Jones stand-in for tests (no model file), as torchsim_relax's.
LJ_PARAMS = {"sigma": 2.3, "epsilon": 0.05, "cutoff": 6.0}


# --- arg schema ------------------------------------------------------------
@dataclass
class JaxCellRelaxArgs(ScriptInput):
    """Args for the ``jax_cell_relax`` processor.

    ``model_remote`` is the staged model on the cluster -- a plain file (a
    NeuralIL ``.pkl``), or a ``.tar`` of a mace-jax bundle dir when
    ``is_archive``; the ``"lj"`` backend needs none. ``supercell_trafo`` is
    the backend's image expansion (empty = the backend default).
    """

    model_remote: str = ""
    backend: str = "neuralil"
    is_archive: bool = False
    fmax: float = 0.02
    steps: int = 1000
    stages: list = field(default_factory=lambda: list(STAGES))
    pressure_key: str = "grid_P"
    supercell_trafo: list = field(default_factory=list)
    max_neighbors: int = 50
    max_neighbors_list: list = field(default_factory=list)
    max_neighbors_offset: int = 4

    def validate(self, root: Optional[Path] = None) -> None:
        if self.backend not in BACKENDS:
            raise ScriptInputError(
                f"jax_cell_relax: 'backend' must be one of {BACKENDS}, got "
                f"{self.backend!r}"
            )
        if self.backend != "lj" and not self.model_remote:
            raise ScriptInputError(
                "jax_cell_relax: 'model_remote' is required"
            )
        if not self.stages or any(s not in STAGES for s in self.stages):
            raise ScriptInputError(
                f"jax_cell_relax: 'stages' must be a non-empty list of "
                f"{STAGES}, got {self.stages!r}"
            )
        if not isinstance(self.fmax, (int, float)) or self.fmax <= 0:
            raise ScriptInputError(
                "jax_cell_relax: 'fmax' must be a positive number"
            )
        if not isinstance(self.steps, int) or self.steps <= 0:
            raise ScriptInputError(
                "jax_cell_relax: 'steps' must be a positive int"
            )
        if self.supercell_trafo and (
            len(self.supercell_trafo) != 3
            or any(int(s) < 1 for s in self.supercell_trafo)
        ):
            raise ScriptInputError(
                "jax_cell_relax: 'supercell_trafo' must be 3 positive ints, "
                f"got {self.supercell_trafo!r}"
            )


# --- cluster-side compute --------------------------------------------------
def _load(args, calcpath, symbols):
    """The jaxrens backend and its ``{symbol: type}`` map."""
    from ase.data import chemical_symbols
    from jaxrens.backends.loader import load_backend

    backend_type = args.get("backend", "neuralil")
    extra = {}
    if args.get("supercell_trafo"):
        extra["supercell_trafo"] = tuple(
            int(s) for s in args["supercell_trafo"]
        )
    if backend_type == "lj":
        backend = load_backend("lj", **LJ_PARAMS, **extra)
        return backend, {s: 0 for s in symbols}
    path = _materialise_model(args, calcpath)
    if backend_type == "mace":
        backend = load_backend("mace", model_path=path, **extra)
    else:
        backend = load_backend(backend_type, pickle_file=path, **extra)
    symbol_map = getattr(backend, "symbol_map", None)
    if symbol_map is None:
        symbol_map = {
            chemical_symbols[int(z)]: i
            for i, z in enumerate(backend.atomic_numbers)
        }
    return backend, symbol_map


def make_energy_forces_stress(backend):
    """Jitted ``(pos, types, cell, buffer) -> (E, F, stress, ovf, count)``.

    ``stress`` is ``(1/V) dE/d eps`` for a symmetric strain ``eps`` applied
    to positions and cell alike (ASE's sign convention: positive = tensile),
    ``F = -dE/dx`` from the same evaluation. ``buffer`` (the neighbour
    buffer) is static: one compilation per (atom count, buffer).
    """
    import functools

    import jax
    import jax.numpy as jnp

    @functools.partial(jax.jit, static_argnums=3)
    def efs(pos, types, cell, buffer):
        def energy(p, eps):
            deform = jnp.eye(3) + 0.5 * (eps + eps.T)
            res = backend(p @ deform.T, types, cell @ deform.T, buffer)
            return res.energy, res

        (e, res), (g_pos, g_eps) = jax.value_and_grad(
            energy, argnums=(0, 1), has_aux=True
        )(pos, jnp.zeros((3, 3), dtype=pos.dtype))
        volume = jnp.abs(jnp.linalg.det(cell))
        return (
            e,
            -g_pos,
            g_eps / volume,
            jnp.any(jnp.asarray(res.overflow)),
            jnp.max(jnp.asarray(res.max_neighbor_count)),
        )

    return efs


def _calculator(backend, efs, types, ladder, offset, fallback):
    """An ASE calculator over ``efs`` that grows its buffer on overflow."""
    import jax.numpy as jnp
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes

    probe = getattr(backend, "max_neighbors_for", None)

    class _JaxCalc(Calculator):
        implemented_properties = ["energy", "free_energy", "forces", "stress"]

        def __init__(self):
            super().__init__()
            self.buffer = None  # sized on the first call, then only grows

        def calculate(
            self, atoms=None, properties=None, system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            pos = jnp.asarray(self.atoms.get_positions())
            cell = jnp.asarray(self.atoms.get_cell()[:])
            if self.buffer is None:
                # The probe gives per-atom counts; size to the busiest atom.
                self.buffer = (
                    _pick_next_bucket(
                        int(jnp.max(probe(pos, cell))), 0, ladder, offset
                    )
                    if ladder and probe is not None
                    else fallback
                )
            while True:
                e, f, s, ovf, count = efs(pos, types, cell, self.buffer)
                if not bool(ovf):
                    break
                if not ladder:
                    raise RuntimeError(
                        f"neighbour overflow at max_neighbors={self.buffer} "
                        f"(saw {int(count)}) and no max_neighbors_list ladder "
                        "to grow into"
                    )
                self.buffer = _pick_next_bucket(
                    int(count), self.buffer, ladder, offset
                )
            stress = np.asarray(s, dtype=float)
            self.results = {
                "energy": float(e),
                "free_energy": float(e),
                "forces": np.asarray(f, dtype=float),
                "stress": stress[
                    [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]
                ],  # Voigt xx yy zz yz xz xy
            }

    return _JaxCalc()


def relax_one(atoms, calc, stages, pressure_gpa, fmax, steps):
    """Relax ``atoms`` (in place) stage by stage; True if every stage
    converged."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from ase.units import GPa

    atoms.calc = calc
    converged = True
    for stage in stages:
        target = (
            atoms
            if stage == "positions"
            else FrechetCellFilter(atoms, scalar_pressure=pressure_gpa * GPa)
        )
        opt = FIRE(target, logfile=None)
        converged &= bool(opt.run(fmax=fmax, steps=steps))
    return converged


def run(atoms_list, args, calcpath=None):
    """Relax every structure stage by stage at its own pressure."""
    import jax.numpy as jnp
    import numpy as np
    from ase.units import GPa

    fmax = float(args.get("fmax", 0.02))
    steps = int(args.get("steps", 1000))
    stages = list(args.get("stages", STAGES))
    key = args.get("pressure_key", "grid_P")
    ladder = tuple(sorted(int(b) for b in args.get("max_neighbors_list", [])))
    offset = int(args.get("max_neighbors_offset", 4))
    fallback = int(args.get("max_neighbors", 50))

    def _log(msg):
        print(f"[jax_cell_relax] {msg}", flush=True)

    symbols = sorted({s for a in atoms_list for s in a.get_chemical_symbols()})
    backend, symbol_map = _load(args, calcpath, symbols)
    efs = make_energy_forces_stress(backend)
    _log(
        f"{len(atoms_list)} structure(s), backend={args.get('backend')}, "
        f"stages={stages}, fmax={fmax}, steps={steps}, ladder={list(ladder)}"
    )

    out = []
    n_conv = n_failed = 0
    for i, original in enumerate(atoms_list):
        p = float(original.info.get(key, 0.0))
        at = original.copy()
        try:
            types = jnp.asarray(
                [symbol_map[s] for s in at.get_chemical_symbols()]
            )
            calc = _calculator(backend, efs, types, ladder, offset, fallback)
            converged = relax_one(at, calc, stages, p, fmax, steps)
            energy = at.get_potential_energy()
            f_max = float(np.linalg.norm(at.get_forces(), axis=1).max())
            pressure = float(-np.trace(at.get_stress(voigt=False)) / 3 / GPa)
        except Exception as exc:  # one bad structure must not sink the shard
            _log(f"structure {i} failed to relax: {exc}")
            failed = original.copy()
            failed.info[CONVERGED_KEY] = False
            failed.info[RELAX_FAILED_KEY] = True
            failed.info[RELAX_ERROR_KEY] = str(exc).strip()[-300:]
            failed.info[RELAX_ENERGY_KEY] = float("nan")
            out.append(failed)
            n_failed += 1
            continue
        at.calc = None
        n, v = len(at), at.get_volume()
        at.info[RELAX_ENERGY_KEY] = float(energy)
        at.info[RELAX_ENTHALPY_KEY] = float((energy + p * GPa * v) / n)
        at.info[RELAX_VOLUME_KEY] = float(v / n)
        at.info[RELAX_FMAX_KEY] = f_max
        at.info[RELAX_PRESSURE_KEY] = pressure
        at.info[CONVERGED_KEY] = bool(converged)
        n_conv += int(converged)
        out.append(at)
    tail = f", {n_failed} failed" if n_failed else ""
    _log(f"finished: {n_conv}/{len(atoms_list)} converged{tail}")
    return out


# --- host-side spec --------------------------------------------------------
class JaxCellRelaxProcessor(Processor):
    """Relax structures through a jaxrens backend in stages (positions, cell).

    A pure host-side spec: stages the model (a mace-jax bundle dir is tarred,
    a NeuralIL ``.pkl`` goes as-is) and contributes the remote args; the
    compute is :func:`run`. Drop-in for ``TorchSimRelaxProcessor`` (same
    stages, pressure key and output keys) for models torch-sim cannot run.

    Args:
        model: NeuralIL ``.pkl`` or mace-jax bundle dir; ignored for ``lj``.
        backend: jaxrens backend -- ``"neuralil"``, ``"mace"`` or ``"lj"``.
        fmax / steps: Force tolerance (eV/A; the cell stage also converges
            the cell forces to it) and max FIRE steps, per stage.
        stages: Ordered relaxation stages, from ``("positions", "cell")``.
        pressure_key: ``info`` key holding each structure's pressure (GPa).
        supercell_trafo: The backend's image expansion -- match the NS run.
        max_neighbors_list / max_neighbors_offset: Neighbour-bucket ladder
            and headroom (the buffer grows a rung on overflow).
        max_neighbors: Fixed buffer when no ladder is given.
    """

    name = "jax_cell_relax"

    def __init__(
        self,
        model=None,
        *,
        backend="neuralil",
        fmax=0.02,
        steps=1000,
        stages=STAGES,
        pressure_key="grid_P",
        supercell_trafo=None,
        max_neighbors=50,
        max_neighbors_list=(),
        max_neighbors_offset=4,
    ):
        if backend != "lj" and model is None:
            raise ValueError(
                f"JaxCellRelaxProcessor needs a model for backend {backend!r}."
            )
        self.model = None if model is None else Path(model)
        self.backend = backend
        self.fmax = float(fmax)
        self.steps = int(steps)
        self.stages = list(stages)
        self.pressure_key = pressure_key
        self.supercell_trafo = (
            [int(s) for s in supercell_trafo] if supercell_trafo else []
        )
        self.max_neighbors = int(max_neighbors)
        self.max_neighbors_list = [int(b) for b in max_neighbors_list or ()]
        self.max_neighbors_offset = int(max_neighbors_offset)
        self._is_archive = self.model is not None and self.model.is_dir()
        self._staged: Optional[Path] = None

    def _staged_file(self) -> Optional[Path]:
        if self.model is None or not self._is_archive:
            return self.model
        if self._staged is None:
            import tarfile
            import tempfile

            tmp = Path(tempfile.mkdtemp(prefix="jax_cell_relax_"))
            self._staged = tmp / (self.model.name + ".tar")
            with tarfile.open(self._staged, "w") as tar:
                tar.add(self.model, arcname=self.model.name)
        return self._staged

    def shared_files(self) -> list:
        staged = self._staged_file()
        return [] if staged is None else [str(staged)]

    def build_args(self, savepath: str) -> dict:
        staged = self._staged_file()
        return JaxCellRelaxArgs(
            model_remote=(
                "" if staged is None else os.path.join(savepath, staged.name)
            ),
            backend=self.backend,
            is_archive=self._is_archive,
            fmax=self.fmax,
            steps=self.steps,
            stages=self.stages,
            pressure_key=self.pressure_key,
            supercell_trafo=self.supercell_trafo,
            max_neighbors=self.max_neighbors,
            max_neighbors_list=self.max_neighbors_list,
            max_neighbors_offset=self.max_neighbors_offset,
        ).to_dict()


register("jax_cell_relax", run, schema=JaxCellRelaxArgs)
