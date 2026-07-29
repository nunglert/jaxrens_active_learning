"""Config ``info``/``arrays`` keys shared across the dashboard side (+ a reader).

One home for the property keys the panels agree on, so the same string is not
redefined in every module. ``remote_handler``'s evaluators label ground-truth
energies/forces with ``REF_energy`` / ``REF_forces``; jaxrens trajectories carry
the sampler's surrogate energy under ``ns_energy`` (preserved through the
evaluator, so evaluated configs may carry both -> a "how far off was the
sampler" parity).
"""

from __future__ import annotations

REF_ENERGY_KEY = "REF_energy"
REF_FORCES_KEY = "REF_forces"
SURROGATE_ENERGY_KEY = "ns_energy"

# Committee (ensemble) disagreement the ``committee_eval`` processor writes and
# ``remote_handler``'s ``UncertaintySelector`` reads back: per-config energy std
# (``info``) and per-atom force std (``arrays``, shape ``(n_atoms, 3)``). The
# defaults match ``UncertaintySelector``'s own key names, so the stock selector
# consumes them with no configuration.
ENERGY_UNCERTAINTY_KEY = "energy_uncertainty"
FORCES_UNCERTAINTY_KEY = "forces_uncertainty"

# Committee mean prediction (the ensemble's own answer), written alongside the
# std as an informational annotation -- not consumed by the stock selector. The
# mean *forces* (``arrays``, ``(n_atoms, 3)``) are what a *relative* force
# uncertainty needs: the raw ``FORCES_UNCERTAINTY_KEY`` is an absolute spread,
# so ranking on it favours configurations whose forces are large in the first
# place (hot, disordered, low-density ones); ``|sigma_F| / |F|`` does not.
COMMITTEE_ENERGY_KEY = "committee_energy"
COMMITTEE_FORCES_KEY = "committee_forces"

# Space-group label the ``spacegroup`` processor writes and the phase panel
# reads (relaxed-basin phase assignment).
SPACEGROUP_KEY = "spacegroup"
SPACEGROUP_SYMBOL_KEY = "spacegroup_symbol"

# Potential energy of the relaxed structure the ``relax`` processor writes
# (the optimizer's objective evaluated at the local minimum, eV).
RELAX_ENERGY_KEY = "relax_energy"

# Set by the ``relax`` processor on a structure whose relaxation *crashed* (e.g.
# a neighbour-bucket overflow the ladder can't cover, or a backend error), so a
# single bad structure degrades to a skip instead of failing the whole shard.
# Carries ``relax_failed=True`` + ``relax_error=<short reason>`` in ``info``; the
# structure keeps its sampled geometry and is marked ``is_converged=False``.
RELAX_FAILED_KEY = "relax_failed"
RELAX_ERROR_KEY = "relax_error"

# Per-atom sampled (pre-relaxation) positions the ``relax`` processor stashes as
# an ``arrays`` entry before overwriting ``positions`` with the relaxed basin --
# so one frame keeps both the sampled dead point and where it relaxed to. The
# active ``positions`` remain the relaxed geometry (what ``spacegroup`` labels).
INITIAL_POSITIONS_KEY = "initial_positions"

# Per-atom relaxed positions the ``relax`` processor writes (an ``arrays``
# entry, so the frame keeps its original sampled positions -- relaxation is an
# annotation, not an overwrite). The ``spacegroup`` step reads these when
# present so it labels the relaxed basin, not the sampled geometry.
RELAXED_POSITIONS_KEY = "relaxed_positions"


def first_has_info_key(files, key) -> bool:
    """True if the first config of ``files`` carries ``key`` in its ``info``.

    A cheap header-peek so a panel/section is only wired when the key it reads
    is actually present (reading a missing key would raise at compute time). Any
    read error -> False. ``ase`` is imported lazily to keep this import-light.
    """
    if not files:
        return False
    try:
        from ase.io import read

        return key in read(str(files[0]), index=0).info
    except Exception:
        return False
