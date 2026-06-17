"""Two-phase jaxrens active-learning loop.

Composition root in action: ``remote_handler`` drives the
Sampler -> Evaluator -> Trainer loop and the *sampler* is this project's own
``JaxrensNSSampler`` (jaxrens nested sampling). The loop only *produces* the
``calchome`` run dirs; visualize them afterwards with the standalone CLI::

    jaxrens-al-dashboard <calchome>/phase1_mace
    jaxrens-al-dashboard <calchome>/phase2_neuralil

The run is split into two phases, each its own ``ActiveLearningHandler`` and
``calchome`` subdir:

  PHASE 1 -- foundation-model bootstrap (~3 iters)
    Enter the loop with a MACE foundation model (as a mace-jax bundle) driving
    the RENS sampler. Candidates are chosen by QUESTS structural *diversity*
    (no uncertainty needed yet); DFT-labelled with VASP; then the foundation
    model is *finetuned* and converted torch->jax in one job
    (MaceFinetuningTrainer(convert_to_jax=True)). Each iteration's sampler uses
    the latest finetuned mace-jax model.

  PHASE 2 -- NeuralIL-ensemble refinement
    Seeded with ALL of phase 1's DFT-labelled configs, train a cheaper NeuralIL
    *ensemble* from scratch on the full accumulated dataset each iter, then
    sample with the ensemble and resample by committee *uncertainty*
    (UncertaintySelector on jaxrens' ns_energy_std / ns_force_std, which the
    phase-2 ns config has jaxrens annotate in place).

This is a CONCRETE template -- the values below (SSH aliases, partitions,
savepaths, conda envs, script/model paths) are real-shaped examples; **edit
them for your cluster/model before running**. Needs the ``[run]`` extra
(``pip install ".[run]"``) and a configured cluster
(``REMOTE_HANDLER_CONFIG_DIR`` + SSH config). Run with ``STUB = True`` first
to exercise the full two-phase
wiring/bookkeeping with no cluster.
"""

import json
import logging
import os
import sys
from pathlib import Path

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import RemoteFileSystem
from remote_handler.workflows import active_learning_base, evaluators, trainers
from remote_handler.workflows.selectors import (
    QUESTSSelector,
    UncertaintySelector,
)

from jaxrens_al import template_dir
from jaxrens_al.sampler import JaxrensNSSampler

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# --- toggles --------------------------------------------------------------
STUB = True  # True: dry-run the loop locally (no cluster/SSH/jaxrens)
LOOP_TIME = 10  # seconds between job-completion polls
N_GPU_PARALLEL = 1
PHASE1_ITERS = 3  # foundation-model bootstrap iterations
PHASE2_ITERS = 10  # NeuralIL-ensemble refinement iterations
SAMPLE_SIZE = 100  # candidates the sampler hands to selection per iteration

# --- paths ----------------------------------------------------------------
INPUTS = Path(__file__).resolve().parent / "inputs"
# remote-side payloads shipped + run on the cluster (from remote_handler):
RH_SCRIPTS = Path("/home/you/code/remote_handler/src/remote_handler/scripts")

# Phase-1 MACE artifacts (real-shaped; edit for your model) ----------------
# The foundation model as a mace-jax bundle DIRECTORY (config.json +
# params.msgpack -- what jaxrens' create_mace loads). Iteration 0's sampler
# loads this; later iterations use the finetuned bundles the trainer emits.
# The AL pipeline now ships directories (not just single files), so a bundle
# dir flows through both the handler and the sampler unchanged.
FOUNDATION_MODEL_JAX = INPUTS / "foundation_mace_jax"
# The torch foundation model the finetuner starts from.
FOUNDATION_MODEL = INPUTS / "foundation.model"
MACE_CONFIG = INPUTS / "mace.yaml"  # MACE (finetuning) training config
# MACE run_train.py on the training cluster.
MACE_TRAIN_RUNNER = Path("/home/you/code/mace/mace/cli/run_train.py")


def collect_evaluated_outputs(calchome: Path) -> list[Path]:
    """All DFT-labelled config files a finished phase wrote, from its progress.

    The handler records each iteration's ``evaluator_output`` (paths relative
    to its calchome) in ``AL_progress.json``. Phase 2 is seeded with these so
    it trains on everything phase 1 collected.
    """
    progress_file = calchome / "AL_progress.json"
    with open(progress_file, "r") as f:
        progress = json.load(f)

    outputs: list[Path] = []
    for entry in progress:
        out = entry.get("evaluator_output")
        if out is None:
            continue
        for rel in out if isinstance(out, list) else [out]:
            outputs.append(calchome / rel)
    return outputs


# ==========================================================================
# Sampler schedules: ramp NS settings across iterations (per phase).
#
# kwargs are dotted jaxrens keys, applied by JaxrensNSSampler as
# ``jaxrens run --set run.n_live=...`` overrides on the base ns.yaml. Legacy
# cumulative-threshold style (experiments/al_loop.py): later ``if`` blocks
# override earlier ones. Every stage MUST return the SAME keys --
# Schedule.check() (run in __init__) enforces this.
# ==========================================================================
class Phase1SamplerSchedule(active_learning_base.Schedule):
    def get_kwargs(self, iteration_idx):
        # Foundation bootstrap: coarse walkers, ramping up over 3 iterations.
        kwargs = {"run.n_live": 100}
        if iteration_idx > 0:
            kwargs = {"run.n_live": 200}
        if iteration_idx > 1:
            kwargs = {"run.n_live": 400}
        return kwargs


class Phase2SamplerSchedule(active_learning_base.Schedule):
    def get_kwargs(self, iteration_idx):
        # Coarse for the early refinement iters, then full resolution.
        kwargs = {"run.n_live": 200}
        if iteration_idx > 1:
            kwargs = {"run.n_live": 400}
        if iteration_idx > 3:
            kwargs = {"run.n_live": 800}
        return kwargs


def preflight(schedule, sampler):
    """Print the ramp and trial-apply every iteration's kwargs to the base
    config via the sampler's own assembly -- a bad override fails HERE, not
    mid-run on the cluster. Pure dict work; no cluster needed.
    """
    schedule.visualize()
    schedule.test(
        lambda kwargs: sampler._build_config(
            model_remote="PLACEHOLDER", start_remote=None, overrides=kwargs
        )
    )


# ==========================================================================
# PHASE 1 -- MACE foundation bootstrap.
# ==========================================================================
def run_phase1(calchome: Path, gpu_cluster, dft_cluster, gpu_fs, dft_fs):
    # SAMPLER: jaxrens NS on the MACE (mace-jax) model; QUESTS *diversity*
    # selection (no uncertainty in this phase).
    sampler = JaxrensNSSampler(
        remote_fs=gpu_fs,
        cluster_config=gpu_cluster,
        config_file=INPUTS / "ns_mace.yaml",
        sample_size=SAMPLE_SIZE,
        selector=QUESTSSelector(n_configs=SAMPLE_SIZE, method="msc"),
        n_gpu_parallel=N_GPU_PARALLEL,
        backend_type="mace",
        out_file_prefix="mace_sn16",
        remote_savepath=gpu_cluster.savepath,
        loop_time=LOOP_TIME,
        stub=STUB,
    )

    # EVALUATOR: VASP single-point DFT.
    evaluator = evaluators.VASPEvaluator(
        singlepoint_script=RH_SCRIPTS / "dft_singlepoint.py",
        remote_fs=dft_fs,
        cluster_config=dft_cluster,
        num_parallel=25,
        incar_kwargs=INCAR_KWARGS,
        stub=STUB,
        loop_time=LOOP_TIME,
        remote_savepath=dft_cluster.savepath,
    )

    # TRAINER: finetune the MACE foundation model, convert torch -> mace-jax in
    # the same job so the next iteration's sampler can load the .pkl bundle.
    trainer = trainers.MaceFinetuningTrainer(
        remote_fs=gpu_fs,
        cluster_config=gpu_cluster,
        training_script=RH_SCRIPTS / "mace" / "mace_finetuning.py",
        config_file=MACE_CONFIG,
        foundation_model=FOUNDATION_MODEL,
        remote_mace_train_runner=MACE_TRAIN_RUNNER,
        num_models=1,
        convert_to_jax=True,
        stub=STUB,
        loop_time=LOOP_TIME,
        remote_savepath=gpu_cluster.savepath,
    )

    schedule = Phase1SamplerSchedule(PHASE1_ITERS)
    preflight(schedule, sampler)

    handler = active_learning_base.ActiveLearningHandler(
        sampler=sampler,
        evaluator=evaluator,
        trainer=trainer,
        sampler_schedule=schedule,
        calchome=str(calchome),
        # Enter at the sampler with the foundation model (as mace-jax) so the
        # very first RENS run is foundation-driven; each later iteration's
        # sampler uses the freshly finetuned model.
        cycle_entry_point="sampler",
        initial_trainer_out=[FOUNDATION_MODEL_JAX],
        combine_evaluator_outputs=True,  # finetune on the growing dataset
        num_iterations=PHASE1_ITERS,
    )
    handler.run()


# ==========================================================================
# PHASE 2 -- NeuralIL-ensemble refinement with uncertainty resampling.
# ==========================================================================
def run_phase2(
    calchome, seed_evaluated, gpu_cluster, dft_cluster, gpu_fs, dft_fs
):
    # SAMPLER: jaxrens NS on the NeuralIL ensemble; resample by committee
    # uncertainty. The phase-2 ns config has jaxrens annotate the trajectory
    # in place, so each frame carries ns_energy_std + per-atom ns_force_std.
    sampler = JaxrensNSSampler(
        remote_fs=gpu_fs,
        cluster_config=gpu_cluster,
        config_file=INPUTS / "ns_neuralil.yaml",
        sample_size=SAMPLE_SIZE,
        selector=UncertaintySelector(
            n_configs=SAMPLE_SIZE,
            energy_key="ns_energy_std",  # info: committee energy std
            forces_key="ns_force_std",  # arrays: per-atom force std
            energy_weight=0.0,  # forces drive selection (legacy)
            forces_weight=1.0,
        ),
        n_gpu_parallel=N_GPU_PARALLEL,
        backend_type="neuralil",
        out_file_prefix="neuralil_sn16",
        remote_savepath=gpu_cluster.savepath,
        loop_time=LOOP_TIME,
        stub=STUB,
    )

    evaluator = evaluators.VASPEvaluator(
        singlepoint_script=RH_SCRIPTS / "dft_singlepoint.py",
        remote_fs=dft_fs,
        cluster_config=dft_cluster,
        num_parallel=25,
        incar_kwargs=INCAR_KWARGS,
        stub=STUB,
        loop_time=LOOP_TIME,
        remote_savepath=dft_cluster.savepath,
    )

    # TRAINER: NeuralIL ensemble, retrained from scratch each iteration on the
    # full accumulated dataset (combine_evaluator_outputs=True below).
    trainer = trainers.NeuralilILTrainer(
        training_script=RH_SCRIPTS / "neuralil" / "neuralil_training.py",
        remote_fs=gpu_fs,
        cluster_config=gpu_cluster,
        training_kwargs={
            "r_cut": 4.0,
            "training_fraction": 0.75,
            "supercell_diag": [3, 3, 3],
            "high_energy_penalty": -5.0,
            "n_ensemble": 30,
            "save_reduced": 5,
            "n_batch": 4,
        },
        stub=STUB,
        loop_time=LOOP_TIME,
        remote_savepath=gpu_cluster.savepath,
    )

    schedule = Phase2SamplerSchedule(PHASE2_ITERS)
    preflight(schedule, sampler)

    handler = active_learning_base.ActiveLearningHandler(
        sampler=sampler,
        evaluator=evaluator,
        trainer=trainer,
        sampler_schedule=schedule,
        calchome=str(calchome),
        # Enter at training: iteration 0 trains the NeuralIL ensemble from
        # scratch on ALL of phase 1's labelled configs, then samples with it.
        cycle_entry_point="training",
        initial_evaluator_out=seed_evaluated,
        combine_evaluator_outputs=True,  # always train on the full dataset
        num_iterations=PHASE2_ITERS,
    )
    handler.run()


# Shared VASP INCAR settings for both phases' single-points.
INCAR_KWARGS = {
    "xc": "R2SCAN",
    "encut": 300,
    "prec": "Normal",
    "ismear": -1,
    "sigma": 0.025,
    "ibrion": -1,
    "ediff": 1e-5,
    "kspacing": 0.3,
    "lwave": False,
    "lcharg": False,
    "lreal": False,
}


def main(calchome: str = "calchome"):
    calchome = Path(calchome)

    # Clusters (lugus/musica/...) are read from your remote_handler config --
    # the default ~/.config/remote_handler, or REMOTE_HANDLER_CONFIG_DIR if you
    # export one. (Do NOT force it to a local dir here, or ClusterConfig.
    # from_name won't find your registered clusters.)
    #
    # The only thing remote_handler can't find on its own is jaxrens_al's
    # SLURM *body* template (body_jaxrens.sh.j2) -- the cluster *headers*
    # (musica.sh.j2, dirona.sh.j2) are remote_handler built-ins. So point the
    # template search at the packaged dir that ships the body (no copy needed).
    # If you use a custom, non-built-in cluster header, instead point this at a
    # dir holding BOTH that header and body_jaxrens.sh.j2.
    os.environ["REMOTE_HANDLER_SLURM_TEMPLATES"] = str(template_dir())

    # --- file systems + clusters -----------------------------------------
    lugus_fs = RemoteFileSystem(remote_alias="lugus")
    musica_fs = RemoteFileSystem(remote_alias="musica")
    dft_cluster = ClusterConfig.from_name("lugus")  # CPU: VASP single-points
    gpu_cluster = ClusterConfig.from_name("musica")  # GPU: NS + training

    # --- PHASE 1: MACE foundation bootstrap ------------------------------
    phase1 = calchome / "phase1_mace"
    run_phase1(phase1, gpu_cluster, dft_cluster, musica_fs, lugus_fs)

    # --- PHASE 2: NeuralIL-ensemble refinement ---------------------------
    # Seed with every DFT-labelled config phase 1 produced.
    seed = collect_evaluated_outputs(phase1)
    logging.info("Seeding phase 2 with %d labelled config file(s).", len(seed))
    phase2 = calchome / "phase2_neuralil"
    run_phase2(phase2, seed, gpu_cluster, dft_cluster, musica_fs, lugus_fs)

    print(f"AL run complete. Build dashboards with:")
    print(f"  jaxrens-al-dashboard {phase1}")
    print(f"  jaxrens-al-dashboard {phase2}")


if __name__ == "__main__":
    main()
