"""Example: wire and run a jaxrens active-learning loop.

Composition root in action: ``remote_handler`` drives the
Sampler -> Evaluator -> Trainer loop and the *sampler* is this project's own
``JaxrensNSSampler`` (jaxrens nested sampling). The loop only *produces* the
``calchome`` run directory; visualizing it is a separate step -- run the
standalone dashboard CLI afterwards (decoupled, like the legacy
``al_dashboard.py``)::

    jaxrens-al-dashboard calchome            # -> calchome/dashboard.pdf

This is a CONCRETE template -- the values below (SSH aliases, partitions,
savepaths, conda envs, script/input paths) are real-shaped examples from a
prior run; **edit them for your cluster/model before running**. It needs the
``[run]`` extra (``pip install ".[run]"``) and a configured cluster
(``REMOTE_HANDLER_CONFIG_DIR`` + SSH config).

Run in stub mode first (``STUB = True``) to exercise the wiring/bookkeeping
without touching a cluster.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem, RemoteFileSystem
from remote_handler.workflows import (
    StagedActiveLearningHandler,
    active_learning_base,
    evaluators,
    trainers,
)
from remote_handler.workflows.selectors import UniformSelector

from jaxrens_al import template_dir
from jaxrens_al.sampler import JaxrensNSSampler

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# --- toggles --------------------------------------------------------------
STUB = True  # True: dry-run the loop locally (no cluster/SSH/jaxrens)
NUM_ITERATIONS = 20
LOOP_TIME = 10  # seconds between job-completion polls
N_GPU_PARALLEL = 1

# --- paths ----------------------------------------------------------------
HOME = Path(os.getcwd())
INPUTS = Path(__file__).resolve().parent / "inputs"
# remote-side payloads shipped + run on the cluster (from remote_handler):
RH_SCRIPTS = Path("/home/you/code/remote_handler/src/remote_handler/scripts")


def make_template_dir() -> Path:
    """A SLURM-template dir holding BOTH the cluster header and the jaxrens
    body template. remote_handler searches one custom dir
    (REMOTE_HANDLER_SLURM_TEMPLATES) plus its built-ins, so we co-locate the
    project's ``body_jaxrens.sh.j2`` next to the cluster headers.
    """
    tdir = HOME / "config" / "templates"
    tdir.mkdir(parents=True, exist_ok=True)
    # Ship jaxrens_al's body template into the search dir.
    shutil.copy(template_dir() / "body_jaxrens.sh.j2", tdir)
    # NOTE: your cluster header (e.g. datalab.sh.j2) must also live here.
    return tdir


def main(calchome: str = "calchome"):
    calchome = Path(calchome)

    os.environ.setdefault("REMOTE_HANDLER_CONFIG_DIR", str(HOME / "config"))
    os.environ["REMOTE_HANDLER_SLURM_TEMPLATES"] = str(make_template_dir())

    # --- file systems -----------------------------------------------------
    local_fs = LocalFileSystem()
    remote_fs = RemoteFileSystem(remote_alias="datalab")

    # GPU cluster that runs jaxrens NS + NeuralIL training.
    gpu_cluster = ClusterConfig.from_dict(
        {
            "remote_machine": "datalab",
            "partition": "GPU-a100,GPU-a40",
            "savepath": "/home/you/remote_jobs/al/run/",
            "conda_env": "jaxrens",
            "conda_source": "/home/you/miniconda3",
            "template": "datalab.sh.j2",
        }
    )
    # CPU cluster that runs VASP single-points.
    dft_cluster = ClusterConfig.from_dict(
        {
            "remote_machine": "dirona",
            "partition": "normal",
            "savepath": "/home/you/remote_jobs/al/dft/",
            "conda_env": "vasp",
            "conda_source": "/home/you/miniconda3",
            "template": "dirona.sh.j2",
        }
    )

    # --- SAMPLER: jaxrens nested sampling (this project) ------------------
    # Pure producer: emits ALL dead points. Reducing them to candidates is the
    # selector stage below.
    sampler = JaxrensNSSampler(
        remote_fs=remote_fs,
        cluster_config=gpu_cluster,
        config_file=INPUTS / "ns.yaml",
        n_gpu_parallel=N_GPU_PARALLEL,
        backend_type="neuralil",  # matches the trainer's model
        out_file_prefix="ns",
        remote_savepath=gpu_cluster.savepath,
        loop_time=LOOP_TIME,
        stub=STUB,
    )

    # --- EVALUATOR: VASP single-point DFT (remote_handler) ---------------
    evaluator = evaluators.VASPEvaluator(
        singlepoint_script=RH_SCRIPTS / "dft_singlepoint.py",
        remote_fs=local_fs,
        cluster_config=dft_cluster,
        num_parallel=25,
        incar_kwargs={
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
        },
        stub=STUB,
        loop_time=LOOP_TIME,
        remote_savepath=dft_cluster.savepath,
    )

    # --- TRAINER: NeuralIL force field (remote_handler) ------------------
    trainer = trainers.NeuralilILTrainer(
        training_script=RH_SCRIPTS / "neuralil" / "neuralil_training.py",
        remote_fs=remote_fs,
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

    # --- SAMPLER SCHEDULE: ramp NS settings across iterations ------------
    # A remote_handler ``Schedule`` returns per-iteration kwargs that the
    # handler feeds to ``Sampler.update``. For JaxrensNSSampler the kwargs are
    # dotted jaxrens keys, applied as ``jaxrens run --set run.n_live=...``
    # overrides on the base ns.yaml.
    #
    # Legacy cumulative-threshold style (see experiments/al_loop.py): later
    # ``if`` blocks override earlier ones, so each stage is "the settings from
    # this iteration onward". Every stage MUST return the SAME keys --
    # ``Schedule.check()`` (run in __init__) rejects a schedule that doesn't.
    class SamplerSchedule(active_learning_base.Schedule):
        def get_kwargs(self, iteration_idx):
            # Warm-up: many live points + short walks -> cheap, broad coverage
            # while the model is still crude.
            kwargs = {"run.n_live": 400, "run.n_mcmc_steps": 20}
            if iteration_idx > 3:
                # Model improving: decorrelate harder (longer MCMC walks).
                kwargs = {"run.n_live": 300, "run.n_mcmc_steps": 30}
            if iteration_idx > 7:
                # Refine: fewer live points, longest walks for accuracy.
                kwargs = {"run.n_live": 200, "run.n_mcmc_steps": 40}
            return kwargs

    sampler_schedule = SamplerSchedule(NUM_ITERATIONS)

    # Pre-flight the schedule before committing to a long run (legacy pattern):
    #   * visualize() prints the ramp table so you can eyeball it;
    #   * test() trial-applies EVERY iteration's kwargs to the base config via
    #     the sampler's own config assembly, so a bad override (e.g. a dotted
    #     key that doesn't fit the YAML) fails HERE, not mid-run on the cluster.
    # This is pure dict work -- no cluster, no jaxrens needed.
    sampler_schedule.visualize()
    sampler_schedule.test(
        lambda kwargs: sampler._build_config(
            model_remote="PLACEHOLDER", start_remote=None, overrides=kwargs
        )
    )

    # --- orchestrate ------------------------------------------------------
    # Staged handler: sampler -> selector -> evaluator -> trainer. The selector
    # stage (UniformSelector) does the uniform thinning the sampler used to do
    # internally -- here it picks 100 evenly-spaced configs from the dead-point
    # pool. Swap in QUESTSSelector / UncertaintySelector to change strategy.
    handler = StagedActiveLearningHandler.from_actors(
        sampler=sampler,
        evaluator=evaluator,
        trainer=trainer,
        calchome=str(calchome),
        selector=UniformSelector(n_configs=100),
        sampler_schedule=sampler_schedule,
        # Seed the loop with an initial labelled dataset and enter at
        # evaluation so iteration 0 produces the first model the sampler needs.
        initial_sampler_out=INPUTS / "initial_train.extxyz",
        cycle_entry_point="evaluation",
        num_iterations=NUM_ITERATIONS,
    )
    handler.run()

    # Visualize separately with the standalone dashboard CLI:
    #     jaxrens-al-dashboard <calchome>
    print(f"AL run complete: {calchome}")
    print(f"Build the dashboard with:  jaxrens-al-dashboard {calchome}")


if __name__ == "__main__":
    main()
