"""Example: wire and run a jaxrens active-learning loop, then dashboard it.

This is a TEMPLATE. It needs a configured cluster (REMOTE_HANDLER_CONFIG_DIR
+ SSH config) and the [run] extra installed (``pip install ".[run]"``).
Adapt the actor construction to your cluster/model before running.

It illustrates the composition root: remote_handler drives the
Sampler -> Evaluator -> Trainer loop, and afterwards jaxrens_al builds the
alviz dashboard from the run directory it produced.
"""

from pathlib import Path

# Orchestrator + concrete evaluator/trainer come from remote_handler (the
# [run] extra). The *sampler* is the project's own Sampler (jaxrens_al).
from remote_handler.workflows import (  # noqa: F401  (template imports)
    ActiveLearningHandler,
    NeuralilILTrainer,
    VASPEvaluator,
)

from jaxrens_al import build_dashboard
from jaxrens_al.sampler import JaxrensNSSampler


def main(calchome: str = "calchome"):
    calchome = Path(calchome)

    # --- construct the actors (cluster/model specific) -------------------
    # These constructors take cluster configs, scripts and inputs specific
    # to your setup; see remote_handler's examples/ for concrete values.
    sampler = ...  # JaxrensNSSampler(remote_fs=..., cluster_config=..., ...)
    evaluator = ...  # VASPEvaluator(...)
    trainer = ...  # NeuralilILTrainer(...)

    handler = ActiveLearningHandler(
        sampler=sampler,
        evaluator=evaluator,
        trainer=trainer,
        calchome=str(calchome),
        # ... n_iterations, entry/exit points, combine_* flags ...
    )
    handler.run()

    # --- visualize the run we just produced ------------------------------
    board = build_dashboard(calchome)
    board.savefig(calchome / "dashboard.pdf")
    print(f"Wrote {calchome / 'dashboard.pdf'}")


if __name__ == "__main__":
    main()
