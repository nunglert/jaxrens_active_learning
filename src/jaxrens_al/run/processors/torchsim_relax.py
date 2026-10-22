"""The ``torchsim_relax`` processor -- staged relaxation with torch-sim.

A downstream processor for ``remote_handler``'s trajectory-processing framework
(mirroring :mod:`jaxrens_al.run.processors.relax`, but on torch-sim + a torch
MLIP instead of the jaxrens backend). Each structure is relaxed in *stages*
rather than all at once, by default:

1. ``"positions"`` -- atomic positions at fixed cell (FIRE);
2. ``"cell"`` -- cell and positions together under the structure's pressure
   (FIRE + torch-sim's Frechet cell filter).

The pressure is read per structure from ``atoms.info[pressure_key]`` (GPa, 0 if
absent); structures sharing a pressure are relaxed as one batched torch-sim
state, since the cell filter takes one scalar pressure per batch.

Writes the relaxed geometry back and annotates ``info`` with
``relax_energy`` (eV), ``relax_enthalpy`` (eV/atom, ``(E + P V) / N``),
``relax_volume`` (A^3/atom), ``relax_fmax`` (eV/A), ``relax_pressure`` (GPa,
from the final stress) and ``is_converged``. Pair it with the ``spacegroup``
processor to label the relaxed basins.

``torch`` / ``torch_sim`` / ``mace`` are imported lazily, so importing this
module host-side is cheap; the cluster env needs ``torch-sim-atomistic`` (and
``mace-torch`` for the MACE backend) alongside ``jaxrens_al``.
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

from jaxrens_al.keys import RELAX_ENERGY_KEY

STAGES = ("positions", "cell")
BACKENDS = ("mace", "lj")

RELAX_ENTHALPY_KEY = "relax_enthalpy"
RELAX_VOLUME_KEY = "relax_volume"
RELAX_FMAX_KEY = "relax_fmax"
RELAX_PRESSURE_KEY = "relax_pressure"
CONVERGED_KEY = "is_converged"


# --- arg schema ------------------------------------------------------------
@dataclass
class TorchSimRelaxArgs(ScriptInput):
    """Args for the ``torchsim_relax`` processor.

    ``model_remote`` is the staged torch model file (a MACE ``.model``); the
    ``"lj"`` backend (a Lennard-Jones stand-in for tests) needs none.
    ``stages`` is the ordered list of relaxation stages (see module docs).
    """

    model_remote: str = ""
    backend: str = "mace"
    device: str = "cuda"
    dtype: str = "float64"
    fmax: float = 0.02
    steps: int = 1000
    stages: list = field(default_factory=lambda: list(STAGES))
    pressure_key: str = "grid_P"

    def validate(self, root: Optional[Path] = None) -> None:
        if self.backend not in BACKENDS:
            raise ScriptInputError(
                f"torchsim_relax: 'backend' must be one of {BACKENDS}, got {self.backend!r}"
            )
        if self.backend != "lj" and not self.model_remote:
            raise ScriptInputError(
                "torchsim_relax: 'model_remote' is required"
            )
        if not self.stages or any(s not in STAGES for s in self.stages):
            raise ScriptInputError(
                f"torchsim_relax: 'stages' must be a non-empty list of {STAGES}, got {self.stages!r}"
            )
        if self.dtype not in ("float32", "float64"):
            raise ScriptInputError(
                "torchsim_relax: 'dtype' must be 'float32' or 'float64'"
            )
        if not isinstance(self.fmax, (int, float)) or self.fmax <= 0:
            raise ScriptInputError(
                "torchsim_relax: 'fmax' must be a positive number"
            )
        if not isinstance(self.steps, int) or self.steps <= 0:
            raise ScriptInputError(
                "torchsim_relax: 'steps' must be a positive int"
            )


# --- cluster-side compute --------------------------------------------------
def _load_model(args, device, dtype):
    if args.get("backend", "mace") == "lj":
        from torch_sim.models.lennard_jones import LennardJonesModel

        return LennardJonesModel(
            sigma=2.3,
            epsilon=0.05,
            cutoff=6.0,
            device=device,
            dtype=dtype,
            compute_stress=True,
        )
    from torch_sim.models.mace import MaceModel

    return MaceModel(model=args["model_remote"], device=device, dtype=dtype)


def _relax_stage(
    atoms_list, model, stage, pressure_gpa, fmax, steps, device, dtype
):
    import torch_sim as ts
    from ase.units import GPa

    state = ts.initialize_state(atoms_list, device=device, dtype=dtype)
    if stage == "positions":
        out = ts.optimize(
            state,
            model,
            optimizer=ts.Optimizer.fire,
            convergence_fn=ts.generate_force_convergence_fn(fmax),
            max_steps=steps,
        )
    else:
        out = ts.optimize(
            state,
            model,
            optimizer=ts.Optimizer.fire,
            convergence_fn=ts.generate_force_convergence_fn(
                fmax, include_cell_forces=True
            ),
            max_steps=steps,
            init_kwargs={
                "cell_filter": ts.CellFilter.frechet,
                "scalar_pressure": pressure_gpa * GPa,
            },
        )
    return out.to_atoms()


def _annotate(relaxed, originals, model, pressure_gpa, fmax, device, dtype):
    import torch
    import torch_sim as ts
    from ase.units import GPa

    state = ts.initialize_state(relaxed, device=device, dtype=dtype)
    res = model(state)
    energy = res["energy"].detach().cpu().numpy()
    forces = res["forces"].detach()
    stress = res["stress"].detach()
    sys_idx = state.system_idx
    out = []
    for k, (new, old) in enumerate(zip(relaxed, originals)):
        f_k = forces[sys_idx == k].norm(dim=1).max().item()
        p_k = float(-torch.diagonal(stress[k]).mean().item() / GPa)
        n = len(new)
        v = new.get_volume()
        at = old.copy()
        at.set_cell(new.get_cell())
        at.set_positions(new.get_positions())
        at.info[RELAX_ENERGY_KEY] = float(energy[k])
        at.info[RELAX_ENTHALPY_KEY] = float(
            (energy[k] + pressure_gpa * GPa * v) / n
        )
        at.info[RELAX_VOLUME_KEY] = float(v / n)
        at.info[RELAX_FMAX_KEY] = f_k
        at.info[RELAX_PRESSURE_KEY] = p_k
        at.info[CONVERGED_KEY] = bool(f_k < fmax)
        out.append(at)
    return out


def run(atoms_list, args, calcpath=None):
    """Relax every structure stage by stage, batched per pressure."""
    import numpy as np
    import torch

    dtype = (
        torch.float64
        if args.get("dtype", "float64") == "float64"
        else torch.float32
    )
    device = torch.device(args.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    fmax = float(args.get("fmax", 0.02))
    steps = int(args.get("steps", 1000))
    stages = list(args.get("stages", STAGES))
    key = args.get("pressure_key", "grid_P")
    model = _load_model(args, device, dtype)

    pressures = np.array([float(a.info.get(key, 0.0)) for a in atoms_list])
    result = list(atoms_list)
    for p in np.unique(pressures):
        idx = np.flatnonzero(pressures == p)
        batch = [atoms_list[i] for i in idx]
        current = [a.copy() for a in batch]
        for stage in stages:
            current = _relax_stage(
                current, model, stage, float(p), fmax, steps, device, dtype
            )
        for i, at in zip(
            idx,
            _annotate(current, batch, model, float(p), fmax, device, dtype),
        ):
            result[i] = at
    return result


# --- host-side spec --------------------------------------------------------
class TorchSimRelaxProcessor(Processor):
    """Relax structures with torch-sim in stages (positions, then cell).

    A pure host-side spec: stages the torch model file and contributes the
    remote args; the compute is :func:`run`. ``registration_module`` defaults
    to this module, so the stock driver imports it and ``register`` runs.

    Args:
        model: Torch MLIP file (a MACE ``.model``); ignored for ``backend="lj"``.
        backend: ``"mace"`` or ``"lj"`` (Lennard-Jones, for tests).
        device / dtype: Torch device and precision on the compute node.
        fmax: Force tolerance (eV/A) of every stage; the cell stage also
            converges the cell forces to it.
        steps: Maximum FIRE steps per stage.
        stages: Ordered relaxation stages, from ``("positions", "cell")``.
        pressure_key: ``info`` key holding each structure's pressure (GPa).
    """

    name = "torchsim_relax"

    def __init__(
        self,
        model=None,
        *,
        backend="mace",
        device="cuda",
        dtype="float64",
        fmax=0.02,
        steps=1000,
        stages=STAGES,
        pressure_key="grid_P",
    ):
        if backend != "lj" and model is None:
            raise ValueError(
                "TorchSimRelaxProcessor needs a model file for backend 'mace'."
            )
        self.model = None if model is None else Path(model)
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.fmax = float(fmax)
        self.steps = int(steps)
        self.stages = list(stages)
        self.pressure_key = pressure_key

    def shared_files(self) -> list:
        return [] if self.model is None else [str(self.model)]

    def build_args(self, savepath: str) -> dict:
        remote = (
            ""
            if self.model is None
            else os.path.join(str(savepath), self.model.name)
        )
        return TorchSimRelaxArgs(
            model_remote=remote,
            backend=self.backend,
            device=self.device,
            dtype=self.dtype,
            fmax=self.fmax,
            steps=self.steps,
            stages=self.stages,
            pressure_key=self.pressure_key,
        ).to_dict()


register("torchsim_relax", run, schema=TorchSimRelaxArgs)
