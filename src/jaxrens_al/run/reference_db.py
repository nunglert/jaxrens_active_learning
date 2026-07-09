"""Build a starting/reference structure database for an AL run.

This condenses the ad-hoc notebook workflow (``experiments/phases.ipynb``)
into one importable place. The recipe is:

1. **Fetch** every experimentally observed elemental phase of one element
   from the Materials Project (e.g. all pure-Si polymorphs), tagged with its
   space group and ``mp-id``.
2. **Rattle** each phase into a cloud of perturbed configs: build a supercell
   large enough to host a target interaction cutoff, then scale its volume and
   jitter the atomic positions over a grid of (volume-scaling, rattle-strength)
   combinations.
3. **Write** the resulting configurations to an ``extxyz`` you can feed the AL
   loop as iteration-zero candidates.

The Materials Project query needs ``mp_api`` + ``pymatgen`` (the ``[refdb]``
extra); they are imported lazily so the rattling half works on any list of
``ase.Atoms`` with only numpy + ase (the base install). The supercell search
is a small numpy-only port of jaxnest's ``structure_utils`` so this module
does not drag in jax/numba.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from ase import Atoms

# ---------------------------------------------------------------------------
# Supercell search (numpy-only port of jaxnest.structure_utils.find_super).
# ---------------------------------------------------------------------------


def _hnf_diagonals(det_s: int) -> np.ndarray:
    """Diagonal supercell transforms ``[a, b, c]`` with ``a*b*c == det_s``.

    Returns an ``(N, 3)`` array of every ordered diagonal whose product equals
    ``det_s`` (e.g. ``det_s=12`` yields ``[3, 2, 2]``, ``[4, 1, 3]``, ...).
    """
    diags = []
    for i in range(1, det_s + 1):
        if det_s % i:
            continue
        quotient = det_s // i
        for j in range(1, quotient + 1):
            if quotient % j == 0:
                diags.append((i, j, quotient // j))
    return np.array(diags, dtype=np.int64)


def _max_cutoff(cell: np.ndarray) -> float:
    """Largest sphere radius fitting in the cell (min face spacing / 2)."""
    cell = np.asarray(cell, dtype=float)
    volume = abs(np.linalg.det(cell))
    face_areas = np.array(
        [
            np.linalg.norm(np.cross(a, b))
            for a, b in zip(cell, np.roll(cell, 1, axis=0))
        ]
    )
    return float(np.min(volume / face_areas) / 2.0)


def find_supercell(
    cell: np.ndarray, target_cutoff: float, max_det: int = 1000
) -> np.ndarray:
    """Smallest diagonal supercell of ``cell`` that hosts ``target_cutoff``.

    Iterates over diagonal transforms by increasing determinant (number of
    replicas) and returns the first ``[a, b, c]`` whose supercell admits a
    sphere larger than ``target_cutoff``.

    Args:
        cell: Cell vectors as rows, shape ``(3, 3)``.
        target_cutoff: Interaction cutoff (Å) that must fit inside the cell.
        max_det: Cap on the replication count before giving up.

    Returns:
        The diagonal of the transform, shape ``(3,)``.
    """
    cell = np.asarray(cell, dtype=float)
    for det_s in range(1, max_det + 1):
        diagonals = _hnf_diagonals(det_s)
        cutoffs = [_max_cutoff(cell * d[:, None]) for d in diagonals]
        best = int(np.argmax(cutoffs))
        if cutoffs[best] > target_cutoff:
            return diagonals[best]
    raise ValueError(
        f"No supercell hosts cutoff {target_cutoff} within max_det={max_det}."
    )


# ---------------------------------------------------------------------------
# Rattling: perturb each phase over (volume-scaling, rattle-strength) grid.
# ---------------------------------------------------------------------------


def rattle_atoms(
    structures: Sequence[Atoms],
    *,
    rattle_strengths: Sequence[float] = (0.0, 0.2),
    vol_scalings: Optional[Sequence[float]] = None,
    num_realizations: int = 1,
    target_cutoff: float = 3.0,
    seed: Optional[int] = None,
) -> list[Atoms]:
    """Expand each structure into rattled, volume-scaled supercells.

    For every input structure a supercell hosting ``target_cutoff`` is built
    once, then for each volume scaling and rattle strength ``num_realizations``
    perturbed copies are produced (a zero-strength entry is emitted only once
    per volume, since the realizations would be identical). Each output carries
    ``info["vol_scaling"]`` and ``info["rattle_strength"]``.

    Args:
        structures: Base configurations (e.g. from :func:`fetch_mp_elemental`).
        rattle_strengths: Gaussian position-jitter std-devs (Å).
        vol_scalings: Isotropic scale factors applied to the *cell vectors*
            (so a factor ``s`` scales the volume by ``s**3``), matching the
            legacy lattice-scan convention; defaults to ``linspace(0.9, 1.1,
            11)``.
        num_realizations: Random draws per (structure, volume, strength).
        target_cutoff: Cutoff the supercell must host (Å).
        seed: Seed for the RNG (``None`` -> nondeterministic).

    Returns:
        The flat list of perturbed ``ase.Atoms``.
    """
    if vol_scalings is None:
        vol_scalings = np.linspace(0.9, 1.1, 11)
    rng = np.random.default_rng(seed)

    out: list[Atoms] = []
    for base in structures:
        trafo = find_supercell(base.cell[...], target_cutoff=target_cutoff)
        supercell = base * tuple(int(t) for t in trafo)

        for vol_scaling in vol_scalings:
            zero_emitted = False
            for strength in rattle_strengths:
                for _ in range(num_realizations):
                    if strength == 0.0 and zero_emitted:
                        continue
                    at = supercell.copy()
                    at.set_cell(at.cell[...] * vol_scaling, scale_atoms=True)
                    at.set_positions(
                        at.positions
                        + rng.normal(0.0, strength, size=at.positions.shape)
                    )
                    at.info["vol_scaling"] = float(vol_scaling)
                    at.info["rattle_strength"] = float(strength)
                    out.append(at)
                    zero_emitted = True
    return out


# ---------------------------------------------------------------------------
# EOS reference: per-phase volume scan at zero rattle.
# ---------------------------------------------------------------------------


def _phase_label(atoms: Atoms, index: int) -> str:
    """A filesystem-safe label for one phase (``mp_id`` > spacegroup > index).

    Space-group symbols contain ``/`` (e.g. ``P6_3/mmc``), so anything but
    ``[0-9A-Za-z._-]`` is collapsed to ``_`` -- the label becomes a file stem
    (``eos_<label>.extxyz``) the EOS panel reads as one phase.
    """
    raw = atoms.info.get("mp_id") or atoms.info.get("spacegroup")
    raw = str(raw) if raw else f"phase{index}"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", raw)


def eos_phase_scans(
    structures: Sequence[Atoms],
    *,
    vol_scalings: Optional[Sequence[float]] = None,
    target_cutoff: float = 3.0,
) -> dict[str, list[Atoms]]:
    """Per-phase, zero-rattle volume scans for an equation-of-state reference.

    For each input phase a supercell hosting ``target_cutoff`` is built and
    scanned over ``vol_scalings`` with **no** position jitter (``rattle_atoms``
    with a single ``0.0`` strength), so each returned list is a clean E(V)
    curve for one phase -- exactly what :class:`~jaxrens_al.dashboard.panels.eos.
    EquationOfStatePanel` fits (it reads the volume off each frame's cell and
    the DFT energy off ``REF_energy``, so no other provenance is needed).

    Args:
        structures: Base phases, e.g. from :func:`fetch_mp_elemental` (each
            tagged with ``info["mp_id"]`` / ``info["spacegroup"]``).
        vol_scalings: Cell-vector scale factors (volume ∝ ``s**3``); defaults
            to ``rattle_atoms``' own default grid.
        target_cutoff: Cutoff (Å) the per-phase supercell must host.

    Returns:
        ``{phase_label: [volume-scan configs]}``, one entry per phase (a
        duplicate label gets an index suffix so file stems stay unique).
    """
    scans: dict[str, list[Atoms]] = {}
    for i, base in enumerate(structures):
        label = _phase_label(base, i)
        if label in scans:  # keep every phase even if labels collide
            label = f"{label}_{i}"
        scans[label] = rattle_atoms(
            [base],
            rattle_strengths=(0.0,),
            vol_scalings=vol_scalings,
            num_realizations=1,
            target_cutoff=target_cutoff,
        )
    return scans


# ---------------------------------------------------------------------------
# Materials Project fetch (needs the [refdb] extra: mp_api + pymatgen).
# ---------------------------------------------------------------------------


def fetch_mp_elemental(
    element: str,
    *,
    api_key: Optional[str] = None,
    theoretical: bool = False,
) -> list[Atoms]:
    """Fetch elemental phases of ``element`` from the Materials Project.

    Queries the single-element chemical system (so compounds are excluded) and
    returns one ``ase.Atoms`` per entry, each tagged with
    ``info["spacegroup"]`` and ``info["mp_id"]``.

    Args:
        element: Chemical symbol, e.g. ``"Si"``.
        api_key: Materials Project API key; falls back to the ``MP_API_KEY``
            environment variable.
        theoretical: If ``False`` (default), keep only experimentally observed
            phases.

    Returns:
        The fetched structures as ``ase.Atoms``.
    """
    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError(
            "No Materials Project API key: pass api_key= or set MP_API_KEY."
        )
    try:
        from mp_api.client import MPRester
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "fetch_mp_elemental needs mp_api + pymatgen; install the "
            "'refdb' extra: pip install 'jaxrens_active_learning[refdb]'."
        ) from exc

    with MPRester(key) as mpr:
        results = mpr.materials.summary.search(
            formula=element,
            elements=[element],
            chemsys=element,
            theoretical=theoretical,
        )

    structures: list[Atoms] = []
    for r in results:
        at = AseAtomsAdaptor.get_atoms(r.structure)
        at.info["spacegroup"] = str(r.symmetry.symbol)
        at.info["mp_id"] = str(r.material_id)
        structures.append(at)
    return structures


# ---------------------------------------------------------------------------
# One-call orchestration.
# ---------------------------------------------------------------------------


def build_reference_database(
    element: str,
    *,
    api_key: Optional[str] = None,
    theoretical: bool = False,
    target_cutoff: float = 3.0,
    rattle_strengths: Sequence[float] = (0.0, 0.2),
    vol_scalings: Optional[Sequence[float]] = None,
    num_realizations: int = 1,
    seed: Optional[int] = None,
    output: Optional[str | Path] = None,
) -> list[Atoms]:
    """Fetch elemental phases, rattle them, and (optionally) write an extxyz.

    Thin wrapper tying :func:`fetch_mp_elemental` to :func:`rattle_atoms`; the
    knobs mirror those functions. If ``output`` is given the configurations are
    written there with ``ase.io.write``.

    Returns:
        The starting-database configurations.
    """
    structures = fetch_mp_elemental(
        element, api_key=api_key, theoretical=theoretical
    )
    rattled = rattle_atoms(
        structures,
        rattle_strengths=rattle_strengths,
        vol_scalings=vol_scalings,
        num_realizations=num_realizations,
        target_cutoff=target_cutoff,
        seed=seed,
    )
    if output is not None:
        import ase.io

        ase.io.write(str(output), rattled)
    return rattled


# ---------------------------------------------------------------------------
# CLI: jaxrens-al-refdb <element> <output.extxyz>
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Console-script entry point. Returns a process exit code."""
    import argparse
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        prog="jaxrens-al-refdb",
        description="Build a starting structure database for an AL run: "
        "fetch an element's experimentally observed phases from the "
        "Materials Project, rattle + volume-scale them, write an extxyz.",
    )
    parser.add_argument("element", help="Chemical symbol, e.g. Si.")
    parser.add_argument("output", help="Output extxyz path.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Materials Project API key (default: $MP_API_KEY).",
    )
    parser.add_argument(
        "--theoretical",
        action="store_true",
        help="Include theoretical (not just experimental) phases.",
    )
    parser.add_argument(
        "--target-cutoff",
        type=float,
        default=3.0,
        help="Interaction cutoff (Å) the supercell must host (default: 3.0).",
    )
    parser.add_argument(
        "--rattle-strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.2],
        metavar="STD",
        help="Gaussian position-jitter std-devs in Å (default: 0.0 0.2).",
    )
    parser.add_argument(
        "--num-realizations",
        type=int,
        default=1,
        help="Random draws per (structure, volume, strength) (default: 1).",
    )
    parser.add_argument(
        "--n-vol",
        type=int,
        default=11,
        help="Number of volume scalings in [vol-min, vol-max] (default: 11).",
    )
    parser.add_argument(
        "--vol-min",
        type=float,
        default=0.9,
        help="Smallest volume scale factor (default: 0.9).",
    )
    parser.add_argument(
        "--vol-max",
        type=float,
        default=1.1,
        help="Largest volume scale factor (default: 1.1).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducibility."
    )
    args = parser.parse_args(argv)

    vol_scalings = np.linspace(args.vol_min, args.vol_max, args.n_vol)
    try:
        configs = build_reference_database(
            args.element,
            api_key=args.api_key,
            theoretical=args.theoretical,
            target_cutoff=args.target_cutoff,
            rattle_strengths=args.rattle_strengths,
            vol_scalings=vol_scalings,
            num_realizations=args.num_realizations,
            seed=args.seed,
            output=args.output,
        )
    except (ValueError, ImportError) as exc:
        logger.error(str(exc))
        return 1
    logger.info(f"Wrote {len(configs)} configs to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
