"""Run-side trajectory processors for ``remote_handler``'s processing layer.

Downstream steps that plug into
:class:`remote_handler.processing.RemoteTrajectoryProcessor`. Each is a single
module (arg schema + cluster-side ``run`` + host-side
:class:`~remote_handler.processing.base.Processor` spec + a ``register`` call),
mirroring ``remote_handler``'s built-in ``processors/`` -- so the orchestrator
itself stays unaware of the heavier upstreams (here, ``jaxrens``). The stock
shard driver imports the processor's module on the cluster (via the
processor's ``registration_module``) before resolving steps, so nothing else is
needed; ``jaxrens_al`` must be installed in the cluster analysis env.

Importing this package pulls in ``remote_handler`` (the ``[run]`` extra) --
keep it off the dashboard import path (the reader panels never import it).
"""

from jaxrens_al.processors.model_eval import ModelEvalArgs, ModelEvalProcessor
from jaxrens_al.processors.relax import RelaxArgs, RelaxProcessor
from jaxrens_al.processors.spacegroup import (
    SpaceGroupArgs,
    SpaceGroupProcessor,
)
from jaxrens_al.processors.steinhardt import (
    SteinhardtArgs,
    SteinhardtProcessor,
)

__all__ = [
    "SteinhardtArgs",
    "SteinhardtProcessor",
    "ModelEvalArgs",
    "ModelEvalProcessor",
    "RelaxArgs",
    "RelaxProcessor",
    "SpaceGroupArgs",
    "SpaceGroupProcessor",
]
