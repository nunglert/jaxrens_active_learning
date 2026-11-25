"""run25 -- Si 1-16 GPa: foundation-model bootstrap, then a NeuralIL AL loop.

Replaces run24 (10-100 GPa on the MACE foundation, which was too expensive:
dense cells + huge neighbour buffers). Two phases, each its own
``StagedActiveLearningHandler`` + calchome subdir:

  PHASE 1 -- ``calchome_staged/phase1_mace``: ONE iteration, no training.
    A single jaxrens NS run on the r2SCAN MACE foundation (16 Si, 16 pressures
    1-16 GPa) -> ``StrideSelector(BOOTSTRAP_SIZE)`` slices the full dead-point
    pool evenly along every replica's trajectory -> VASP R2SCAN. The loop
    exits after the evaluator (``cycle_exit_point="evaluation"``), so the
    trainer passed in is never run.

  PHASE 2 -- ``calchome_staged/phase2_neuralil``: the AL loop.
    Enters at the trainer, seeded with ALL of phase 1's labelled configs, and
    retrains a NeuralIL ensemble from scratch each iteration on the whole
    accumulated dataset (run46_si hyperparameters). Each iteration then runs NS
    on that ensemble and a ``NeuralILCommitteeUncertaintySelector`` scores the
    pool with the ensemble's members (one GPU job) and keeps the
    ``SAMPLE_SIZE`` most force-uncertain configs for DFT.

Start with ``STUB = True`` (phase 2's selector then falls back to a stride,
since the committee selector has no stub mode and would submit a real GPU
job). Launch for real with ``bash submit_local.sh``.
"""

import json
import logging
import os
from pathlib import Path

from remote_handler import configure_logging
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.workflows import evaluators, trainers
from remote_handler.workflows.active_learning_base import Schedule
from remote_handler.workflows.selectors import StrideSelector
from remote_handler.workflows.staged_active_learning import (
    StagedActiveLearningHandler,
)

from jaxrens_al import template_dir
from jaxrens_al.run.dashboard_hook import DashboardHook
from jaxrens_al.run.committee_selector import (
    NeuralILCommitteeUncertaintySelector,
)
from jaxrens_al.run.sampler import JaxrensNSSampler
from jaxrens_al.run.staged_al import build_committee_al_handler

configure_logging()

# --- toggles --------------------------------------------------------------
STUB = False
LOOP_TIME = 60
N_GPU_PARALLEL = 4
# musica's walltime is 3 days; NS jobs auto-resume via an afternotok chain.
MAX_RESUBMITS = 3
REATTACH_MAX_WAIT = MAX_RESUBMITS * 3 * 24 * 60 * 60

BOOTSTRAP_SIZE = 1000  # phase 1: configs strided from the foundation NS pool
PHASE2_ITERS = 14
SAMPLE_SIZE = 200  # phase 2: configs the committee selector keeps (run46_si)
STRIDE_POOL = 50 * SAMPLE_SIZE  # cap the pool the committee GPU-evaluates

# --- paths ----------------------------------------------------------------
INPUTS = Path(__file__).resolve().parent / "inputs"
FOUNDATION_MODEL_JAX = INPUTS / "foundation_mace_jax"  # mace-matpes-r2scan-0

# NeuralIL ensemble hyperparameters 
NEURALIL_KWARGS = {
    "r_cut": 4.5,
    "training_fraction": 0.9,
    "supercell_diag": [2, 2, 2],
    "high_energy_penalty_relative": 5.0,
    "n_ensemble": 10,
    "save_reduced": 5,
    "n_batch": 32,
    "optimizer_type": "onecycle",
    "n_epochs": 1000
}
NEURALIL_SUPERCELL = (2, 2, 2)
NEURALIL_MAX_NEIGHBORS = [20, 25, 30, 35, 40, 45, 50, 60, 80]

# Shared VASP INCAR (R2SCAN single points)
INCAR_KWARGS = {
    "xc": "R2SCAN",
    "encut": 300,
    "prec": "Normal",
    "ismear": -1,
    "sigma": 0.025,
    "ibrion": -1,
    "ediff": 1e-7,
    "symprec": 1e-4,
    "kspacing": 0.2,
    "lwave": False,
    "lcharg": False,
    "lreal": False,
}


# ==========================================================================
# Phase-2 sampler schedule
# pressure-grid density and system size, on the 1-16 GPa window.
# ==========================================================================
P_MIN, P_MAX = 1.0, 16.0


def _pressure_grid(n):
    step = (P_MAX - P_MIN) / (n - 1)
    return [round(P_MIN + i * step, 3) for i in range(n)]


PRESSURES_COARSE = _pressure_grid(8)  # 1, 3.143, ..., 16 GPa
PRESSURES_DENSE = _pressure_grid(16)  # 1, 2, ..., 16 GPa
PRESSURES_FINE = _pressure_grid(32)  # 1, 1.484, ..., 16 GPa

# System size (init.start_species = "<Z> <count>").
SPECIES_16 = "14 16"  # 16 Si atoms
SPECIES_32 = "14 32"  # 32 Si atoms


class NeuralILSamplerSchedule(Schedule):
    """run24's per-step (n_live, pressure grid, system size), shifted by one.

    Phase 2 enters at the trainer, so iteration 0 only trains and never
    samples; sampling step ``k`` of run24's schedule runs at iteration
    ``k + 1``.

    =====  ============  ========  =====
    step   n_live        replicas  atoms
    =====  ============  ========  =====
    0-2    100/200/400   8         16
    3-4    200/400       16        16
    5-6    100/200       16        32
    7-     200           32        32
    last   400           32        32   (final K=400 run)
    =====  ============  ========  =====
    """

    def get_kwargs(self, iteration_idx):
        step = max(iteration_idx - 1, 0)
        if step < 3:
            return {
                "run.n_live": (100, 200, 400)[step],
                "ensemble.pressure": PRESSURES_COARSE,
                "init.start_species": SPECIES_16,
            }
        if step < 5:
            return {
                "run.n_live": 200 if step == 3 else 400,
                "ensemble.pressure": PRESSURES_DENSE,
                "init.start_species": SPECIES_16,
            }
        if step < 7:
            return {
                "run.n_live": 100 if step == 5 else 200,
                "ensemble.pressure": PRESSURES_DENSE,
                "init.start_species": SPECIES_32,
            }
        return {
            "run.n_live": (
                400 if iteration_idx == self.num_iterations - 1 else 200
            ),
            "ensemble.pressure": PRESSURES_FINE,
            "init.start_species": SPECIES_32,
        }


def preflight(sampler, schedule=None):
    """Trial-apply the config (+ every scheduled override) host-side."""

    def _build(kwargs):
        sampler._build_config(
            model_remote="PLACEHOLDER", start_remote=None, overrides=kwargs
        )

    if schedule is None:
        _build({})
        return
    for i in range(schedule.num_iterations):
        logging.info("iter %d: %s", i, schedule.get_kwargs(i))
    schedule.test(_build)


def collect_evaluated_outputs(calchome: Path) -> list[Path]:
    """Every DFT-labelled file a finished staged phase wrote (all iters)."""
    with open(calchome / "AL_progress.json") as f:
        progress = json.load(f)
    by_iter = progress.get("outputs", {}).get("evaluator", {})
    return [
        calchome / rel
        for _, rels in sorted(by_iter.items(), key=lambda kv: int(kv[0]))
        for rel in rels
    ]


def _evaluator(dft_cluster, dft_fs):
    return evaluators.VASPEvaluator(
        remote_fs=dft_fs,
        cluster_config=dft_cluster,
        num_parallel=25,
        incar_kwargs=INCAR_KWARGS,
        stub=STUB,
        loop_time=LOOP_TIME,
        remote_savepath=dft_cluster.savepath,
    )


def _neuralil_trainer(gpu_cluster, gpu_fs):
    return trainers.NeuralilILTrainer(
        remote_fs=gpu_fs,
        cluster_config=gpu_cluster,
        training_kwargs=NEURALIL_KWARGS,
        stub=STUB,
        loop_time=LOOP_TIME,
        remote_savepath=gpu_cluster.savepath,
    )


# ==========================================================================
# PHASE 1: one foundation-model NS run -> stride -> DFT.
# ==========================================================================
def run_phase1(calchome, gpu_cluster, dft_cluster, gpu_fs, dft_fs):
    sampler = JaxrensNSSampler(
        remote_fs=gpu_fs,
        cluster_config=gpu_cluster,
        config_file=INPUTS / "ns_mace.yaml",
        n_gpu_parallel=N_GPU_PARALLEL,
        backend_type="mace",
        out_file_prefix="mace_si16",
        remote_savepath=gpu_cluster.savepath,
        loop_time=LOOP_TIME,
        stub=STUB,
        max_resubmits=MAX_RESUBMITS,
        reattach_max_wait=REATTACH_MAX_WAIT,
    )
    preflight(sampler)

    handler = StagedActiveLearningHandler.from_actors(
        sampler,
        _evaluator(dft_cluster, dft_fs),
        _neuralil_trainer(gpu_cluster, gpu_fs),
        calchome=calchome,
        selector=StrideSelector(n_configs=BOOTSTRAP_SIZE),
        initial_trainer_out=[FOUNDATION_MODEL_JAX],
        num_iterations=1,
        cycle_exit_point="evaluation",
    )
    handler.run()


# ==========================================================================
# PHASE 2: NeuralIL ensemble AL, committee-uncertainty selection.
# ==========================================================================
def run_phase2(calchome, seed, gpu_cluster, dft_cluster, gpu_fs, dft_fs):
    sampler = JaxrensNSSampler(
        remote_fs=gpu_fs,
        cluster_config=gpu_cluster,
        config_file=INPUTS / "ns_neuralil.yaml",
        n_gpu_parallel=N_GPU_PARALLEL,
        backend_type="neuralil",
        out_file_prefix="neuralil_si16",
        remote_savepath=gpu_cluster.savepath,
        loop_time=LOOP_TIME,
        stub=STUB,
        max_resubmits=MAX_RESUBMITS,
        reattach_max_wait=REATTACH_MAX_WAIT,
    )
    schedule = NeuralILSamplerSchedule(PHASE2_ITERS)
    preflight(sampler, schedule)

    if STUB:
        # CommitteeUncertaintySelector has no stub mode (it always submits a
        # real GPU job), so a dry run strides instead.
        selector = StrideSelector(n_configs=SAMPLE_SIZE)
    else:
        selector = NeuralILCommitteeUncertaintySelector(
            committee_input="committee",  # the trainer's ensemble .pkl
            n_configs=SAMPLE_SIZE,
            pre_stride=STRIDE_POOL,
            cluster=gpu_cluster,
            num_parallel=N_GPU_PARALLEL,
            force_metric="relative",  # run16's; absolute biases to gas cells
            supercell_trafo=NEURALIL_SUPERCELL,
            max_neighbors_list=NEURALIL_MAX_NEIGHBORS,
            max_neighbors_offset=3,
        )

    common = dict(
        sampler=sampler,
        evaluator=_evaluator(dft_cluster, dft_fs),
        trainer=_neuralil_trainer(gpu_cluster, gpu_fs),
        calchome=calchome,
        sampler_schedule=schedule,
        num_iterations=PHASE2_ITERS,
        combine_evaluator_outputs=True,  # always train on the full dataset
        trainer_output_extension="pkl",
    )
    if STUB:
        handler = StagedActiveLearningHandler.from_actors(
            common.pop("sampler"),
            common.pop("evaluator"),
            common.pop("trainer"),
            selector=selector,
            cycle_entry_point="training",
            initial_evaluator_out=seed,
            **common,
        )
    else:
        handler = build_committee_al_handler(
            selector=selector,
            entry_stage="trainer",
            initial_evaluator_out=seed,
            on_stage_complete=[DashboardHook()],
            **common,
        )
    handler.run()


def main(calchome: str = "calchome_staged"):
    calchome = Path(calchome)
    os.environ["REMOTE_HANDLER_SLURM_TEMPLATES"] = str(template_dir())

    dft_cluster = ClusterConfig.from_name("lugus")
    gpu_cluster = ClusterConfig.from_name("musica")
    lugus_fs = dft_cluster.remote_fs()
    musica_fs = gpu_cluster.remote_fs()

    phase1 = calchome / "phase1_mace"
    run_phase1(phase1, gpu_cluster, dft_cluster, musica_fs, lugus_fs)

    seed = collect_evaluated_outputs(phase1)
    logging.info("Seeding phase 2 with %d labelled file(s).", len(seed))
    phase2 = calchome / "phase2_neuralil"
    run_phase2(phase2, seed, gpu_cluster, dft_cluster, musica_fs, lugus_fs)

    print("Run complete. Dashboards:")
    print(f"  jaxrens-al dashboard {phase1}")
    print(f"  jaxrens-al dashboard {phase2}")


if __name__ == "__main__":
    main()
