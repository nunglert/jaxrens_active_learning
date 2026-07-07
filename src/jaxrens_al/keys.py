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
