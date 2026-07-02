"""Tests for the run-side Steinhardt trajectory processor.

Needs the [run] extra (remote_handler's ``processing`` framework); the module
self-skips otherwise. The host-side spec + arg schema + registration are
exercised with NO cluster or jaxrens -- only remote_handler + ase on the host.
The one test that runs the actual ``jaxrens`` computation self-skips when
jaxrens is absent.
"""

import pytest

pytest.importorskip("remote_handler.processing")

from remote_handler.exceptions import ScriptInputError  # noqa: E402
from remote_handler.processing.base import Processor  # noqa: E402
from remote_handler.processing.registry import (  # noqa: E402
    REGISTRY,
    validate_args,
)

from jaxrens_al.processors import (  # noqa: E402
    SteinhardtArgs,
    SteinhardtProcessor,
)

# Run-side: needs the producer package (remote_handler).
pytestmark = pytest.mark.integration


def test_steinhardt_processor_is_a_processor_spec():
    proc = SteinhardtProcessor(r_cut=3.5, l_values=(4, 6))
    assert isinstance(proc, Processor)
    assert proc.name == "steinhardt"
    # Nothing to stage: jaxrens supplies the science from the cluster env.
    assert proc.shared_files() == []


def test_registration_module_defaults_to_this_module():
    """The framework auto-resolves it -- driver imports this to register."""
    assert (
        SteinhardtProcessor.registration_module
        == "jaxrens_al.processors.steinhardt"
    )


def test_build_args_round_trips_through_the_schema():
    proc = SteinhardtProcessor(r_cut=4.2, l_values=(4, 6), r_cut_min=1.5)
    args = proc.build_args("/remote/savepath")
    assert args == {"l_values": [4, 6], "r_cut": 4.2, "r_cut_min": 1.5}
    # The emitted payload passes the same schema the driver validates against.
    SteinhardtArgs.from_dict(args).validate()


def test_processor_validates_l_values():
    with pytest.raises(ValueError, match="at least one l value"):
        SteinhardtProcessor(r_cut=3.5, l_values=())


def test_driver_registered_steinhardt_with_schema():
    """Importing the module registered the step's run + arg schema."""
    assert "steinhardt" in REGISTRY
    # validate_args resolves the schema registered under the name.
    validate_args("steinhardt", {"l_values": [6], "r_cut": 3.5})


def test_validate_args_rejects_bad_payload():
    with pytest.raises(ScriptInputError):
        validate_args("steinhardt", {"l_values": [], "r_cut": 3.5})
    with pytest.raises(ScriptInputError):
        validate_args("steinhardt", {"l_values": [6], "r_cut": -1.0})
    with pytest.raises(ScriptInputError):  # r_cut_min must be < r_cut
        validate_args(
            "steinhardt", {"l_values": [6], "r_cut": 3.5, "r_cut_min": 4.0}
        )


def test_run_annotates_atoms():
    """The cluster-side ``run`` attaches q_l/w_l via jaxrens (self-skips)."""
    pytest.importorskip("jaxrens")
    import numpy as np
    from ase import Atoms

    from jaxrens_al.processors import steinhardt

    rng = np.random.default_rng(0)
    atoms = Atoms(
        symbols=["Cu"] * 12 + ["O"] * 4,
        positions=rng.uniform(0, 6, (16, 3)),
        cell=[6, 6, 6],
        pbc=True,
    )
    out = steinhardt.run([atoms], {"l_values": [4, 6], "r_cut": 3.5})
    at = out[0]
    for key in ("q4", "w4", "q6", "w6"):
        assert key in at.arrays and len(at.arrays[key]) == len(at)
    assert "q6_mean" in at.info and "w4_mean" in at.info
