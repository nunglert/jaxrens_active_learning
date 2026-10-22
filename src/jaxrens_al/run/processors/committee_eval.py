"""The ``committee_eval`` processor -- committee disagreement of N MLIPs.

A downstream processor for ``remote_handler``'s trajectory-processing framework
(mirroring :mod:`jaxrens_al.run.processors.model_eval`). On the cluster it loads
*several* trained models (a committee) through the jaxrens backend, evaluates
each on every frame, and annotates the frame with the committee's **std** of the
predicted energy (``info``) and per-atom forces (``arrays``) -- the ensemble
*disagreement*. A committee of MACE-jax bundles needs ``cuequivariance`` (a CUDA
build), so this runs on a GPU node rather than in a CPU host process.

The predictions land under the shared uncertainty keys
(:data:`jaxrens_al.keys.ENERGY_UNCERTAINTY_KEY` /
:data:`~jaxrens_al.keys.FORCES_UNCERTAINTY_KEY`), which are exactly the defaults
``remote_handler``'s ``UncertaintySelector`` reads -- so
:class:`jaxrens_al.run.committee_selector.CommitteeUncertaintySelector` runs this
job and then hands the annotated pool to the stock selector for ranking, with no
extra wiring. The committee **mean** prediction is written alongside the std --
energy in ``info`` (:data:`~jaxrens_al.keys.COMMITTEE_ENERGY_KEY`) and per-atom
forces in ``arrays`` (:data:`~jaxrens_al.keys.COMMITTEE_FORCES_KEY`). The mean
forces are not informational only: they are the denominator a *relative* force
uncertainty ``|sigma_F| / |F|`` needs, which the absolute spread the selector
ranks on cannot provide.

Each model bundle is staged as a single ``.tar`` (a mace-jax bundle is a
*directory*; ``shared_files`` copies plain files), untarred cluster-side before
the load. ``jaxrens`` is imported lazily (via ``score_atoms``), so importing
this module host-side is cheap; ``jaxrens_al`` must be installed in the cluster
analysis env.
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

from jaxrens_al.keys import (
    COMMITTEE_ENERGY_KEY,
    COMMITTEE_FORCES_KEY,
    ENERGY_UNCERTAINTY_KEY,
    FORCES_UNCERTAINTY_KEY,
)


# --- arg schema ------------------------------------------------------------
@dataclass
class CommitteeEvalArgs(ScriptInput):
    """Args for the ``committee_eval`` processor.

    ``model_remotes`` are the staged committee members on the cluster -- each a
    ``.tar`` of a mace-jax bundle dir when ``is_archive`` (untarred before
    loading) or a plain file (a neuralil ``.pkl``) otherwise. ``backend``
    selects the jaxrens backend. The per-config energy std lands in
    ``energy_key`` and the per-atom force std in ``forces_key``, each next to
    the committee mean (``mean_energy_key`` / ``mean_forces_key``).
    ``single_file_ensemble``: the committee is ONE checkpoint holding every
    member (a NeuralIL ``PlainEnsemble`` ``.pkl``), scored per member via the
    backend's ``members`` method instead of one ``score_atoms`` pass per file.
    ``supercell_trafo`` is the backend's periodic-image expansion (empty = the
    backend default; set it for NeuralIL, whose default is (1, 1, 1)).
    ``batch_size`` / ``max_neighbors_list`` / ``max_neighbors_offset`` are
    forwarded to ``score_atoms`` (vmapped batching + the neighbour-bucket
    ladder); unset keeps the one-config-per-call path at a fixed
    ``max_neighbors``.
    """

    model_remotes: list[str] = field(default_factory=list)
    backend: str = "mace"
    is_archive: bool = True
    energy_key: str = ENERGY_UNCERTAINTY_KEY
    forces_key: str = FORCES_UNCERTAINTY_KEY
    mean_energy_key: str = COMMITTEE_ENERGY_KEY
    mean_forces_key: str = COMMITTEE_FORCES_KEY
    max_neighbors: int = 50
    batch_size: Optional[int] = None
    max_neighbors_list: list = field(default_factory=list)
    max_neighbors_offset: int = 4
    single_file_ensemble: bool = False
    supercell_trafo: list = field(default_factory=list)

    def validate(self, root: Optional[Path] = None) -> None:
        if self.single_file_ensemble:
            if len(self.model_remotes) != 1:
                raise ScriptInputError(
                    "committee_eval: 'single_file_ensemble' needs exactly one "
                    "model (the ensemble checkpoint)"
                )
        elif len(self.model_remotes) < 2:
            raise ScriptInputError(
                "committee_eval: 'model_remotes' needs at least two models "
                "(a committee)"
            )
        sc = list(self.supercell_trafo)
        if sc and (
            len(sc) != 3 or not all(isinstance(x, int) and x > 0 for x in sc)
        ):
            raise ScriptInputError(
                "committee_eval: 'supercell_trafo' must be three positive "
                "ints (or empty for the backend default)"
            )
        if not all(isinstance(m, str) and m for m in self.model_remotes):
            raise ScriptInputError(
                "committee_eval: every 'model_remotes' entry must be a "
                "non-empty string"
            )
        if not self.backend:
            raise ScriptInputError("committee_eval: 'backend' is required")
        if not self.energy_key or not self.forces_key:
            raise ScriptInputError(
                "committee_eval: 'energy_key' and 'forces_key' are required"
            )
        if not isinstance(self.max_neighbors, int) or self.max_neighbors <= 0:
            raise ScriptInputError(
                "committee_eval: 'max_neighbors' must be a positive int"
            )
        if self.batch_size is not None and (
            not isinstance(self.batch_size, int) or self.batch_size <= 0
        ):
            raise ScriptInputError(
                "committee_eval: 'batch_size' must be a positive int or null"
            )
        ladder = list(self.max_neighbors_list)
        if ladder and (
            not all(isinstance(b, int) and b > 0 for b in ladder)
            or ladder != sorted(ladder)
        ):
            raise ScriptInputError(
                "committee_eval: 'max_neighbors_list' must be positive ints "
                "in ascending order"
            )


# --- cluster-side compute --------------------------------------------------
def run(atoms_list, args, calcpath=None):
    """Annotate each structure with the committee's energy/force disagreement.

    Evaluates every committee member on the whole pool, then writes, per frame:
    ``info[energy_key]`` (std of the members' total energies, eV),
    ``info[mean_energy_key]`` (their mean), ``arrays[forces_key]`` (per-atom
    std of the members' forces, ``(n_atoms, 3)``) -- the keys
    ``UncertaintySelector`` reads -- and ``arrays[mean_forces_key]`` (their
    per-atom mean, same shape), which lets a downstream ranker form a
    *relative* force uncertainty instead of the absolute spread. The std is
    the population std over members; ranking by it is invariant to the ddof
    convention.
    """
    import numpy as np

    from jaxrens_al.dashboard.panels.model_eval import (
        score_atoms,
        score_members,
    )

    backend_type = args["backend"]
    model_paths = _materialise_models(args, calcpath)
    energy_key = args["energy_key"]
    forces_key = args["forces_key"]
    mean_energy_key = args.get("mean_energy_key", COMMITTEE_ENERGY_KEY)
    mean_forces_key = args.get("mean_forces_key", COMMITTEE_FORCES_KEY)
    max_neighbors = int(args.get("max_neighbors", 50))
    batch_size = args.get("batch_size")
    ladder = list(args.get("max_neighbors_list") or [])
    offset = int(args.get("max_neighbors_offset", 4))
    supercell_trafo = list(args.get("supercell_trafo") or []) or None

    if args.get("single_file_ensemble", False):
        # One checkpoint holding every member: (M,) energies and (M, N, 3)
        # forces per config straight from the backend's members().
        member_energies, member_forces = score_members(
            model_paths[0],
            backend_type,
            atoms_list,
            max_neighbors=max_neighbors,
            max_neighbors_list=ladder or None,
            max_neighbors_offset=offset,
            supercell_trafo=supercell_trafo,
        )
    else:
        # One (energies, forces) pair per committee member. score_atoms loads
        # each model once and loops the pool internally -- N loads, not
        # N*len(pool).
        per_model = [
            score_atoms(
                path,
                backend_type,
                atoms_list,
                want_forces=True,
                max_neighbors=max_neighbors,
                batch_size=batch_size,
                max_neighbors_list=ladder or None,
                max_neighbors_offset=offset,
                supercell_trafo=supercell_trafo,
            )
            for path in model_paths
        ]
        member_energies = [
            np.asarray([e[i] for e, _ in per_model])
            for i in range(len(atoms_list))
        ]
        member_forces = [
            np.stack([f[i] for _, f in per_model])
            for i in range(len(atoms_list))
        ]

    for i, atoms in enumerate(atoms_list):
        # (n_members,) energies and (n_members, n_atoms, 3) forces -> the
        # committee std (per-atom for forces) and mean.
        e, f = np.asarray(member_energies[i]), np.asarray(member_forces[i])
        atoms.info[energy_key] = float(e.std())
        atoms.info[mean_energy_key] = float(e.mean())
        atoms.new_array(forces_key, f.std(axis=0))
        atoms.new_array(mean_forces_key, f.mean(axis=0))
    return atoms_list


def _materialise_models(args, calcpath):
    """Return local model paths, untarring each staged bundle archive first."""
    remotes = args["model_remotes"]
    if not args.get("is_archive", True):
        return list(remotes)
    base = Path(calcpath or ".")
    paths = []
    for i, remote in enumerate(remotes):
        dest = base / f"committee_bundle_{i}"
        with tarfile.open(remote, "r") as tar:
            tar.extractall(dest)
        # The archive holds a single top-level dir (the bundle); descend into it.
        entries = [p for p in dest.iterdir() if p.is_dir()]
        paths.append(str(entries[0] if len(entries) == 1 else dest))
    return paths


# --- host-side spec --------------------------------------------------------
class CommitteeEvalProcessor(Processor):
    """Score structures with a committee of trained models on the cluster.

    A pure host-side spec: it stages every member (a mace-jax bundle dir is
    tarred; a plain file is staged as-is) and contributes the remote args; the
    compute is :func:`run`. ``registration_module`` defaults to this module, so
    the stock driver imports it and ``register`` runs. All members must be the
    same kind (all bundle dirs, or all plain files).

    Args:
        models: The committee members -- at least two mace-jax bundle
            *directories* or neuralil ``.pkl`` files; or, with
            ``single_file_ensemble``, exactly one ensemble checkpoint.
        backend: jaxrens backend to load them with (``"mace"`` / ``"neuralil"``).
        energy_key / forces_key: ``info`` / ``arrays`` keys the energy std and
            per-atom force std are written under (default: the shared
            uncertainty keys ``UncertaintySelector`` reads).
        mean_energy_key: ``info`` key for the committee mean energy.
        mean_forces_key: ``arrays`` key for the committee mean per-atom forces
            -- the denominator of a relative force uncertainty.
        max_neighbors: Static neighbour-buffer size for each backend call
            (the starting bucket when ``max_neighbors_list`` is given).
        batch_size: Configs per vmapped backend call. Unset evaluates one at a
            time, which leaves a GPU idle on small cells.
        max_neighbors_list: Neighbour-bucket ladder. Set it and the buffer is
            sized per batch from the geometry instead of pinned at
            ``max_neighbors``; the ladder bounds how many XLA compilations the
            job pays for. Only used together with ``batch_size``.
        max_neighbors_offset: Headroom above the observed neighbour count when
            picking a bucket.
        single_file_ensemble: ``models`` is one checkpoint holding the whole
            committee (a NeuralIL ensemble ``.pkl``); its members are scored
            via the backend's ``members`` method.
        supercell_trafo: Backend periodic-image expansion ``(a, b, c)``;
            ``None`` = backend default (NeuralIL's is (1, 1, 1) -- set it).
    """

    name = "committee_eval"

    def __init__(
        self,
        models,
        *,
        backend: str = "mace",
        energy_key: str = ENERGY_UNCERTAINTY_KEY,
        forces_key: str = FORCES_UNCERTAINTY_KEY,
        mean_energy_key: str = COMMITTEE_ENERGY_KEY,
        mean_forces_key: str = COMMITTEE_FORCES_KEY,
        max_neighbors: int = 50,
        batch_size: int | None = None,
        max_neighbors_list=None,
        max_neighbors_offset: int = 4,
        single_file_ensemble: bool = False,
        supercell_trafo=None,
    ):
        self.models = [Path(m) for m in models]
        self.single_file_ensemble = bool(single_file_ensemble)
        if self.single_file_ensemble and len(self.models) != 1:
            raise ValueError(
                "CommitteeEvalProcessor(single_file_ensemble=True) needs "
                "exactly one model (the ensemble checkpoint)."
            )
        if not self.single_file_ensemble and len(self.models) < 2:
            raise ValueError(
                "CommitteeEvalProcessor needs at least two models."
            )
        self.supercell_trafo = (
            [int(x) for x in supercell_trafo] if supercell_trafo else []
        )
        self.backend = backend
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.mean_energy_key = mean_energy_key
        self.mean_forces_key = mean_forces_key
        self.max_neighbors = int(max_neighbors)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.max_neighbors_list = (
            sorted(int(b) for b in max_neighbors_list)
            if max_neighbors_list
            else []
        )
        self.max_neighbors_offset = int(max_neighbors_offset)
        self._is_archive = self.models[0].is_dir()
        self._staged: Optional[list[Path]] = None

    def _archives(self) -> list[Path]:
        """Tar each bundle directory once; cache the archive paths."""
        if self._staged is None:
            tmp = Path(tempfile.mkdtemp(prefix="committee_eval_"))
            staged = []
            for model in self.models:
                archive = tmp / (model.name + ".tar")
                with tarfile.open(archive, "w") as tar:
                    tar.add(model, arcname=model.name)
                staged.append(archive)
            self._staged = staged
        return self._staged

    def shared_files(self) -> list:
        staged = self._archives() if self._is_archive else self.models
        return [str(p) for p in staged]

    def build_args(self, savepath: str) -> dict:
        staged = self._archives() if self._is_archive else self.models
        remotes = [str(Path(savepath) / p.name) for p in staged]
        return CommitteeEvalArgs(
            model_remotes=remotes,
            backend=self.backend,
            is_archive=self._is_archive,
            energy_key=self.energy_key,
            forces_key=self.forces_key,
            mean_energy_key=self.mean_energy_key,
            mean_forces_key=self.mean_forces_key,
            max_neighbors=self.max_neighbors,
            batch_size=self.batch_size,
            max_neighbors_list=self.max_neighbors_list,
            max_neighbors_offset=self.max_neighbors_offset,
            single_file_ensemble=self.single_file_ensemble,
            supercell_trafo=self.supercell_trafo,
        ).to_dict()


register("committee_eval", run, schema=CommitteeEvalArgs)
