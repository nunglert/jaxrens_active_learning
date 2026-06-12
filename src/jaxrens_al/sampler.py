"""The jaxrens nested-sampling Sampler for this project.

In the layered architecture, concrete samplers belong to the *project*, not
to the orchestrator: ``remote_handler`` owns the ``Sampler`` *interface* and
the generic remote-execution machinery, while the application picks and wires
the actual sampling method.

``JaxrensNSSampler`` implements ``remote_handler``'s ``Sampler`` ABC for
JAXNEST nested sampling. It is written as a thin **adapter** that delegates
to an inner sampling method object:

    * **today** the method is ``remote_handler.NSTrajectorySampler`` (the NS
      logic + remote execution still live there);
    * **future** once the pure ``sampler`` package is extracted, only the one
      line that constructs ``self._method`` changes -- the adapter keeps
      exposing the same ``Sampler`` interface to the orchestration loop.

Requires the ``[run]`` extra (``pip install ".[run]"`` -> remote_handler).
"""

from remote_handler.workflows import NSTrajectorySampler
from remote_handler.workflows.active_learning_base import Sampler


class JaxrensNSSampler(Sampler):
    """Project-owned jaxrens nested-sampling Sampler.

    Implements ``remote_handler``'s ``Sampler`` interface by delegating to an
    inner NS method. Construction arguments are forwarded verbatim to the
    underlying method (see ``NSTrajectorySampler``): ``remote_fs``,
    ``cluster_config``, ``sampling_script``, ``initial_input_file``,
    ``n_gpu_parallel``, ``sample_size``, ``resample_size``, ``stub``, ...

    Attribute access (e.g. ``.stub``, ``.sample_size``) transparently
    forwards to the inner method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # The single line that changes once the pure `sampler` package
        # exists: swap NSTrajectorySampler for the extracted method.
        self._method = NSTrajectorySampler(*args, **kwargs)

    def update(self, calcdir, sampler_input, **kwargs) -> None:
        return self._method.update(calcdir, sampler_input, **kwargs)

    def get_sampled_configs(self):
        return self._method.get_sampled_configs()

    @property
    def method(self):
        """The underlying sampling-method object (for advanced access)."""
        return self._method

    def __getattr__(self, name):
        # Transparent forwarding of read access to the inner method. Guarded
        # so it does not recurse before ``_method`` is set in __init__.
        method = self.__dict__.get("_method")
        if method is None:
            raise AttributeError(name)
        return getattr(method, name)
