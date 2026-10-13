"""Phase identification on a coarse (T, P) grid of nested-sampling samples.

For a finished NS run (one replica per pressure) this picks, for every grid
temperature, the dead point that dominates the NPT ensemble -- the stored
trajectory frame with the largest thermal weight

    log w_i(T) = log X-shell_i - H_i / (k_B T),

(``H`` the dead-point enthalpy the sampler records) -- relaxes each pick with
torch-sim in stages (positions at fixed cell, then cell under the replica's
pressure; :mod:`jaxrens_al.run.processors.torchsim_relax`, or
:mod:`~jaxrens_al.run.processors.jax_cell_relax` for a model torch-sim cannot
drive, e.g. NeuralIL -- see :func:`relaxer_for`) and labels the relaxed basin
with its spglib space group (:mod:`~jaxrens_al.run.processors.spacegroup`).
The result is one labelled structure per (T, P) cell, i.e. which
crystal (or disordered basin) each region of the NS phase diagram belongs to.

Unlike the ``phase_diagram`` dashboard panel, which relaxes *every* dead point
and reweights per-phase populations, this looks at a handful of representative
structures, so it is cheap enough to run on a CPU and easy to inspect.

Usage (see ``scripts/phase_grid.py`` for the CLI)::

    from jaxrens_al.run.phase_grid import run_phase_grid
    run_phase_grid("results/output", model="mace.model", temperatures=[150, 400, 800],
                   config="results/ns_config.yaml", workdir="phase_grid", cluster="local")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from jaxrens_al.keys import SPACEGROUP_KEY, SPACEGROUP_SYMBOL_KEY

logger = logging.getLogger(__name__)

KB_EV = 8.617333262e-5
GRID_T_KEY = "grid_T"
GRID_P_KEY = "grid_P"
GRID_SHARE_KEY = "grid_weight_share"
REPLICA_KEY = "replica"


def _traj_files(output_dir, prefix):
    return sorted(Path(output_dir).glob(f"{prefix}.run[0-9][0-9].traj.extxyz"))


def _replica_pressure(monitor, index, pressures):
    if pressures is not None:
        return float(pressures[index])
    p = getattr(monitor, "pressure_gpa", None)
    if p is None:
        p = (getattr(monitor, "metadata", None) or {}).get("pressure_gpa")
    if p is None:
        raise ValueError(
            f"no pressure for replica {index}: pass pressures= or config= (the NS config "
            "with the ensemble.pressure ladder in GPa)"
        )
    return float(p)


def most_likely_frames(
    frames, dead_enthalpies, n_live, temperatures, *, n_cull=1, k_B=KB_EV
):
    """Index of the highest-weight frame per temperature, plus its weight share.

    ``frames`` are the stored trajectory frames of one replica (each with
    ``info["iter"]``, its dead-point index); only those can be picked. Returns
    ``(indices, shares)`` with ``shares[k]`` the fraction of the stored frames'
    total thermal weight at ``temperatures[k]`` carried by the pick -- a quick
    check of how dominant the selected structure is.
    """
    from jaxrens.postprocess.thermodynamics import calc_log_weights

    iters = np.array([int(a.info["iter"]) for a in frames])
    keep = iters < len(dead_enthalpies)
    log_shell = calc_log_weights(len(dead_enthalpies), n_live, n_cull)
    h = np.asarray(dead_enthalpies)[iters[keep]]
    base = log_shell[iters[keep]]
    frame_idx = np.flatnonzero(keep)
    indices, shares = [], []
    for T in np.atleast_1d(temperatures):
        score = base - (h - h.min()) / (k_B * float(T))
        j = int(np.argmax(score))
        shares.append(float(1.0 / np.exp(score - score[j]).sum()))
        indices.append(int(frame_idx[j]))
    return np.array(indices), np.array(shares)


def select_grid_samples(
    output_dir,
    temperatures,
    *,
    prefix=None,
    config=None,
    pressures=None,
    replicas=None,
    k_B=KB_EV,
):
    """One representative structure per (T, P) cell of an NS pressure sweep.

    Args:
        output_dir: The jaxrens output dir (``<prefix>.runNN.energies`` +
            ``<prefix>.runNN.traj.extxyz`` per replica, plus the checkpoint).
        temperatures: Grid temperatures (K).
        prefix: Run prefix; inferred by jaxrens from the config if omitted.
        config: NS config yaml; supplies the per-replica pressures (GPa).
        pressures: Explicit per-replica pressures (GPa), overriding the config.
        replicas: Subset of replica indices to use (default: all).
        k_B: Boltzmann constant in the energy units of the dead points.

    Returns:
        List of ``ase.Atoms`` (positions + cell only) with ``info`` keys
        ``grid_T``, ``grid_P`` (GPa), ``replica``, ``iter`` and
        ``grid_weight_share``.
    """
    import ase.io
    from ase import Atoms
    from jaxrens.postprocess import MonitorCollection

    monitors = list(
        MonitorCollection.from_multi_run_directory(
            output_dir, prefix=prefix, config=config
        )
    )
    prefix = prefix or _infer_prefix(output_dir)
    trajs = _traj_files(output_dir, prefix)
    if len(trajs) != len(monitors):
        raise ValueError(
            f"{len(trajs)} trajectories but {len(monitors)} replicas in {output_dir}"
        )
    chosen = range(len(monitors)) if replicas is None else replicas

    samples = []
    for r in chosen:
        mon = monitors[r]
        p_gpa = _replica_pressure(mon, r, pressures)
        frames = ase.io.read(str(trajs[r]), index=":")
        idx, shares = most_likely_frames(
            frames,
            mon.dead_energies,
            mon.n_live,
            temperatures,
            n_cull=mon.n_cull,
            k_B=k_B,
        )
        for T, i, share in zip(np.atleast_1d(temperatures), idx, shares):
            src = frames[i]
            at = Atoms(
                src.numbers,
                positions=src.positions,
                cell=src.cell,
                pbc=src.pbc,
            )
            at.info.update(
                {
                    GRID_T_KEY: float(T),
                    GRID_P_KEY: p_gpa,
                    REPLICA_KEY: int(r),
                    "iter": int(src.info["iter"]),
                    GRID_SHARE_KEY: share,
                }
            )
            samples.append(at)
        logger.info(
            "replica %d (P=%.2f GPa): picked iters %s",
            r,
            p_gpa,
            [int(frames[i].info["iter"]) for i in idx],
        )
    return samples


def unique_picks(samples):
    """The distinct ``(replica, iter)`` picks, in first-seen order.

    Neighbouring grid temperatures often select the same dead point, so a fine
    temperature grid would relax identical structures several times; relax
    these instead and :func:`expand_to_grid` the results back.
    """
    seen = {}
    for at in samples:
        seen.setdefault((int(at.info[REPLICA_KEY]), int(at.info["iter"])), at)
    return list(seen.values())


def expand_to_grid(relaxed, samples):
    """One relaxed frame per grid sample, from the relaxed unique picks.

    Each ``samples`` row gets a copy of its pick's relaxed frame, with the
    per-row ``grid_T`` / ``grid_weight_share`` restored.
    """
    by_key = {
        (int(at.info[REPLICA_KEY]), int(at.info["iter"])): at for at in relaxed
    }
    out = []
    for s in samples:
        key = (int(s.info[REPLICA_KEY]), int(s.info["iter"]))
        if key not in by_key:
            raise RuntimeError(
                f"phase grid: no relaxed frame for replica {key[0]}, iter "
                f"{key[1]} -- the relaxation job dropped it"
            )
        at = by_key[key].copy()
        for k in (GRID_T_KEY, GRID_SHARE_KEY):
            at.info[k] = s.info[k]
        out.append(at)
    return out


def ns_outputs_mtime(output_dir, prefix) -> float:
    """Newest mtime of an NS run's per-replica files (0 if none)."""
    files = list(Path(output_dir).glob(f"{prefix}.run*"))
    return max((f.stat().st_mtime for f in files), default=0.0)


def _infer_prefix(output_dir):
    names = sorted(Path(output_dir).glob("*.run00.energies"))
    if not names:
        raise FileNotFoundError(f"no <prefix>.run00.energies in {output_dir}")
    return names[0].name[: -len(".run00.energies")]


def spacegroup_table(labelled, symprecs=(0.01, 0.1, 0.3)):
    """Per-sample space groups at several spglib tolerances, as table rows."""
    import spglib

    rows = []
    for at in labelled:
        cell = (
            at.get_cell()[:],
            at.get_scaled_positions(),
            at.get_atomic_numbers(),
        )
        sgs = {}
        for prec in symprecs:
            sg = spglib.get_spacegroup(cell, symprec=prec)
            sgs[prec] = sg or "unknown"
        rows.append(
            {
                "T": at.info[GRID_T_KEY],
                "P": at.info[GRID_P_KEY],
                "replica": at.info[REPLICA_KEY],
                "iter": at.info["iter"],
                "share": at.info[GRID_SHARE_KEY],
                "V": at.info.get("relax_volume", np.nan),
                "H": at.info.get("relax_enthalpy", np.nan),
                "fmax": at.info.get("relax_fmax", np.nan),
                "converged": at.info.get("is_converged", False),
                **{f"sg_{prec:g}": s for prec, s in sgs.items()},
            }
        )
    return rows


def relaxer_for(model, backend) -> str:
    """``"torchsim"`` for a torch MACE ``.model`` (and the ``lj`` stand-in
    without one), else ``"jax"`` -- a NeuralIL ``.pkl`` or a mace-jax bundle
    goes through its jaxrens backend."""
    if model is None:
        return "torchsim"
    if backend == "mace" and Path(model).suffix == ".model":
        return "torchsim"
    return "jax"


def phase_grid_job(
    picks,
    n_samples,
    *,
    model,
    cluster_config,
    remote_fs,
    remote_savepath,
    local_savepath=None,
    backend="mace",
    device="cuda",
    fmax=0.02,
    steps=1000,
    stages=("positions", "cell"),
    symprec=0.1,
    supercell_trafo=None,
    max_neighbors_list=None,
    max_neighbors_offset=4,
    tag="phasegrid",
    jobname="AL_PHASE_GRID",
    loop_time=30.0,
):
    """The ``[<relax>, spacegroup]`` job over a picks file (not started).

    ``<relax>`` is ``torchsim_relax`` for a torch ``.model`` and
    ``jax_cell_relax`` otherwise (:func:`relaxer_for`); only the latter uses
    ``supercell_trafo`` / ``max_neighbors_list`` / ``max_neighbors_offset``
    (the backend settings the NS run sampled with), and only the former
    ``device``. One shard holds every pick (a grid is small). The labelled
    output lands at ``front_tag(picks, tag)``. Call ``.run()`` to submit and
    wait, or split it into ``allocate_data()`` / ``submit(only_start=True)``
    to queue several.
    """
    from remote_handler.processing import RemoteTrajectoryProcessor

    from jaxrens_al.run.processors.spacegroup import SpaceGroupProcessor

    if relaxer_for(model, backend) == "torchsim":
        from jaxrens_al.run.processors.torchsim_relax import (
            TorchSimRelaxProcessor,
        )

        relax = TorchSimRelaxProcessor(
            model,
            backend=backend,
            device=device,
            fmax=fmax,
            steps=steps,
            stages=stages,
        )
    else:
        from jaxrens_al.run.processors.jax_cell_relax import (
            JaxCellRelaxProcessor,
        )

        relax = JaxCellRelaxProcessor(
            model,
            backend=backend,
            fmax=fmax,
            steps=steps,
            stages=stages,
            supercell_trafo=supercell_trafo,
            max_neighbors_list=max_neighbors_list or (),
            max_neighbors_offset=max_neighbors_offset,
        )

    return RemoteTrajectoryProcessor(
        traj_files=[str(picks)],
        cluster_config=cluster_config,
        remote_fs=remote_fs,
        n_processors=1,
        shard_size=max(int(n_samples), 1),
        savepath=str(remote_savepath),
        num_parallel=1,
        processors=[relax, SpaceGroupProcessor(symprec=symprec)],
        tag=tag,
        local_savepath=local_savepath,
        jobname=jobname,
        loop_time=loop_time,
        every=1,
    )


def run_phase_grid(
    output_dir,
    model,
    temperatures,
    *,
    workdir,
    prefix=None,
    config=None,
    pressures=None,
    replicas=None,
    cluster="local",
    backend="mace",
    device="cuda",
    fmax=0.02,
    steps=1000,
    stages=("positions", "cell"),
    symprec=0.1,
    supercell_trafo=None,
    max_neighbors_list=None,
    max_neighbors_offset=4,
    table_symprecs=(0.01, 0.1, 0.3),
    reuse=True,
):
    """Select, relax and space-group-label a (T, P) grid of NS samples.

    Writes ``<workdir>/grid_samples.extxyz`` (the picks; the distinct ones,
    relaxed once each, in ``grid_unique.extxyz``),
    ``<workdir>/phase_grid.extxyz`` (relaxed + labelled, one per pick) and
    ``<workdir>/phase_grid.csv`` (one row per sample, space groups at
    ``table_symprecs``). ``cluster`` is ``"local"`` (this machine, no
    scheduler) or a ``ClusterConfig`` JSON name/path, as for the dashboard.
    With ``reuse`` an existing labelled file newer than the NS run is kept.
    A NeuralIL ``.pkl`` (``backend="neuralil"``) or mace-jax bundle is
    relaxed through its jaxrens backend (:func:`relaxer_for`); pass the NS
    run's ``supercell_trafo`` and neighbour ladder for it.

    Returns the table rows (see :func:`spacegroup_table`).
    """
    import csv

    import ase.io

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    picks = workdir / "grid_samples.extxyz"
    unique_file = workdir / "grid_unique.extxyz"
    labelled_file = workdir / "phase_grid.extxyz"
    prefix = prefix or _infer_prefix(output_dir)

    if not (
        reuse
        and labelled_file.exists()
        and labelled_file.stat().st_mtime
        >= ns_outputs_mtime(output_dir, prefix)
    ):
        from remote_handler.core.atoms_shard import front_tag

        from jaxrens_al.run.processing_resolver import execution_context

        samples = select_grid_samples(
            output_dir,
            temperatures,
            prefix=prefix,
            config=config,
            pressures=pressures,
            replicas=replicas,
        )
        unique = unique_picks(samples)
        ase.io.write(picks, samples)
        ase.io.write(unique_file, unique)
        logger.info(
            "wrote %d grid samples (%d distinct) to %s",
            len(samples),
            len(unique),
            picks,
        )
        tag = "phasegrid"
        phase_grid_job(
            unique_file,
            len(unique),
            model=model,
            backend=backend,
            device=device,
            fmax=fmax,
            steps=steps,
            stages=stages,
            symprec=symprec,
            supercell_trafo=supercell_trafo,
            max_neighbors_list=max_neighbors_list,
            max_neighbors_offset=max_neighbors_offset,
            tag=tag,
            loop_time=10,
            **execution_context(cluster, workdir / "jobs"),
        ).run()
        relaxed = ase.io.read(front_tag(str(unique_file), tag), index=":")
        ase.io.write(labelled_file, expand_to_grid(relaxed, samples))

    labelled = ase.io.read(labelled_file, index=":")
    rows = spacegroup_table(labelled, table_symprecs)
    with open(workdir / "phase_grid.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "wrote %s (%d samples; '%s' holds the space group at symprec=%g)",
        workdir / "phase_grid.csv",
        len(rows),
        SPACEGROUP_SYMBOL_KEY,
        symprec,
    )
    return rows


__all__ = [
    "GRID_P_KEY",
    "GRID_SHARE_KEY",
    "GRID_T_KEY",
    "REPLICA_KEY",
    "SPACEGROUP_KEY",
    "expand_to_grid",
    "most_likely_frames",
    "ns_outputs_mtime",
    "phase_grid_job",
    "relaxer_for",
    "run_phase_grid",
    "select_grid_samples",
    "spacegroup_table",
    "unique_picks",
]
