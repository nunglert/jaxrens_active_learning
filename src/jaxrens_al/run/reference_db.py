"""Build a starting/reference structure database for an AL run.

This condenses the ad-hoc notebook workflow (``experiments/phases.ipynb``)
into one importable place. The recipe is:

1. **Fetch** every experimentally observed phase of one *system* from the
   Materials Project -- an element (``"Si"``: all pure-Si polymorphs), a
   formula (``"SiO2"``: every quartz-stoichiometry polymorph) or a chemical
   system (``"Si-O"``: every Si-O compound) -- tagged with its space group,
   ``mp-id``, formula and hull distance.
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

import logging
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from ase import Atoms

# The numpy-only supercell search now lives at the package root (base-install
# safe, shared with the dashboard's model-eval re-score); re-exported here so
# ``jaxrens_al.run.reference_db.find_supercell`` keeps working.
from jaxrens_al.supercell import (  # noqa: F401
    _hnf_diagonals,
    _max_cutoff,
    find_supercell,
    min_image_supercell,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rattling: perturb each phase over (volume-scaling, rattle-strength) grid.
# ---------------------------------------------------------------------------


def rattle_atoms(
    structures: Sequence[Atoms],
    *,
    rattle_strengths: Sequence[float] = (0.0, 0.2),
    vol_scalings: Optional[Sequence[float]] = None,
    num_realizations: int = 1,
    target_cutoff: Optional[float] = 3.0,
    seed: Optional[int] = None,
) -> list[Atoms]:
    """Expand each structure into rattled, volume-scaled supercells.

    For every input structure a supercell hosting ``target_cutoff`` is built
    once, then for each volume scaling and rattle strength ``num_realizations``
    perturbed copies are produced (a zero-strength entry is emitted only once
    per volume, since the realizations would be identical). Each output carries
    ``info["vol_scaling"]`` and ``info["rattle_strength"]``.

    Args:
        structures: Base configurations (e.g. from
            :func:`fetch_mp_structures`).
        rattle_strengths: Gaussian position-jitter std-devs (Å).
        vol_scalings: Isotropic scale factors applied to the *cell vectors*
            (so a factor ``s`` scales the volume by ``s**3``), matching the
            legacy lattice-scan convention; defaults to ``linspace(0.9, 1.1,
            11)``.
        num_realizations: Random draws per (structure, volume, strength).
        target_cutoff: Cutoff the supercell must host (Å), or ``None`` to keep
            each structure at its input (unit-cell) size -- e.g. a DFT reference
            that needs no minimum-image cell, leaving the cutoff-hosting
            supercell to be built later at force-field evaluation time.
        seed: Seed for the RNG (``None`` -> nondeterministic).

    Returns:
        The flat list of perturbed ``ase.Atoms``.
    """
    if vol_scalings is None:
        vol_scalings = np.linspace(0.9, 1.1, 11)
    rng = np.random.default_rng(seed)

    out: list[Atoms] = []
    for base in structures:
        if target_cutoff is None:
            supercell = base.copy()
        else:
            supercell = min_image_supercell(base, target_cutoff)

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


def _phase_label(
    atoms: Atoms, index: int, *, with_formula: bool = False
) -> str:
    """A filesystem-safe label for one phase (``mp_id`` > spacegroup > index).

    Space-group symbols contain ``/`` (e.g. ``P6_3/mmc``), so anything but
    ``[0-9A-Za-z._-]`` is collapsed to ``_`` -- the label becomes a file stem
    (``eos_<label>.extxyz``) the EOS panel reads as one phase.

    ``with_formula`` prefixes the composition (``SiO2_mp-6922``), which is
    what you want once a scan set spans several stoichiometries -- bare
    ``mp-`` ids say nothing about which compound a curve belongs to. Off by
    default so existing per-phase file names (and their on-disk reuse) hold.
    """
    raw = atoms.info.get("mp_id") or atoms.info.get("spacegroup")
    raw = str(raw) if raw else f"phase{index}"
    formula = atoms.info.get("formula")
    if with_formula and formula:
        raw = f"{formula}_{raw}"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", raw)


def eos_phase_scans(
    structures: Sequence[Atoms],
    *,
    vol_scalings: Optional[Sequence[float]] = None,
    target_cutoff: Optional[float] = 3.0,
    label_formula: bool = False,
) -> dict[str, list[Atoms]]:
    """Per-phase, zero-rattle volume scans for an equation-of-state reference.

    For each input phase a supercell hosting ``target_cutoff`` is built and
    scanned over ``vol_scalings`` with **no** position jitter (``rattle_atoms``
    with a single ``0.0`` strength), so each returned list is a clean E(V)
    curve for one phase -- exactly what :class:`~jaxrens_al.dashboard.panels.eos.
    EquationOfStatePanel` fits (it reads the volume off each frame's cell and
    the DFT energy off ``REF_energy``, so no other provenance is needed).

    Args:
        structures: Base phases, e.g. from :func:`fetch_mp_structures` (each
            tagged with ``info["mp_id"]`` / ``info["spacegroup"]``).
        vol_scalings: Cell-vector scale factors (volume ∝ ``s**3``); defaults
            to ``rattle_atoms``' own default grid.
        target_cutoff: Cutoff (Å) the per-phase supercell must host, or ``None``
            to keep each phase at its unit-cell size. Since the EOS panel works
            entirely per atom, a ``None`` (unit-cell) DFT scan is directly
            comparable to a model scored on a cutoff-hosting supercell built at
            eval time -- so a DFT reference can skip the (costlier) supercell.
        label_formula: Prefix each label with the phase's composition
            (``SiO2_mp-6922``). Worth turning on for a multi-compound system,
            where bare ``mp-`` ids make the EOS legend unreadable.

    Returns:
        ``{phase_label: [volume-scan configs]}``, one entry per phase (a
        duplicate label gets an index suffix so file stems stay unique).
    """
    scans: dict[str, list[Atoms]] = {}
    for i, base in enumerate(structures):
        label = _phase_label(base, i, with_formula=label_formula)
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

#: Query filters :func:`fetch_mp_structures` understands beyond the system
#: itself -- the set a caller (CLI, resolver) may forward verbatim, see
#: :func:`mp_query_kwargs`.
MP_QUERY_KEYS = (
    "theoretical",
    "include_subsystems",
    "elements",
    "exclude_elements",
    "num_elements",
    "max_e_above_hull",
    "only_stable",
    "max_sites",
    "max_per_composition",
    "max_structures",
)

#: Summary fields the fetch asks for (a light query -- ``all_fields`` would
#: pull megabytes of unused properties per material).
_MP_FIELDS = (
    "material_id",
    "formula_pretty",
    "chemsys",
    "symmetry",
    "energy_above_hull",
    "is_stable",
    "nsites",
    "structure",
)


def _chemical_symbols() -> frozenset:
    """The set of valid element symbols (ase's table, no extra dependency)."""
    from ase.data import chemical_symbols

    return frozenset(chemical_symbols[1:])


def _parse_system(system: str | Sequence[str]) -> dict:
    """Interpret a system spec as one Materials-Project query constraint.

    The one place the "what did the user mean" rule lives:

    * a sequence of symbols (``["Si", "O"]``) -> the ``chemsys`` they span;
    * a string holding ``-`` (``"Si-O"``, ``"Li-Fe-O"``, ``"*-*-O"``) ->
      ``chemsys``, i.e. *every compound of exactly those elements*;
    * a bare element symbol (``"Si"``) -> ``chemsys`` too, so the elemental
      case keeps its historical meaning (all pure-Si polymorphs);
    * anything else (``"SiO2"``, ``"Li2FeSiO4"``, ``"**O3"``) -> ``formula``,
      i.e. *every polymorph of that stoichiometry*.

    Returns:
        ``{"chemsys": ...}`` or ``{"formula": ...}``, ready to splat into the
        summary search.
    """
    symbols = _chemical_symbols()

    if not isinstance(system, str):
        parts = [str(s).strip() for s in system if str(s).strip()]
        if not parts:
            raise ValueError("Empty system specification.")
        unknown = sorted({p for p in parts if p not in symbols})
        if unknown:
            raise ValueError(
                f"Not chemical symbols: {', '.join(unknown)}. Pass a system "
                "as symbols (['Si', 'O']), a chemsys ('Si-O') or a formula "
                "('SiO2')."
            )
        return {"chemsys": "-".join(sorted(set(parts)))}

    spec = system.strip()
    if not spec:
        raise ValueError("Empty system specification.")

    if "-" in spec:
        parts = [p.strip() for p in spec.split("-")]
        unknown = sorted({p for p in parts if p != "*" and p not in symbols})
        if unknown:
            raise ValueError(
                f"Bad chemsys {spec!r}: {', '.join(unknown)} is not a "
                "chemical symbol (a chemsys is '-'-joined symbols, e.g. "
                "'Li-Fe-O', wildcards allowed: '*-*-O')."
            )
        # MP stores chemsys alphabetically; sort so equal systems written in
        # any order hit the same query (wildcards keep the given order).
        if "*" not in parts:
            parts = sorted(set(parts))
        return {"chemsys": "-".join(parts)}

    if spec in symbols:
        return {"chemsys": spec}
    return {"formula": spec}


def _expand_subsystems(chemsys: str) -> list[str]:
    """All sub-systems of a chemsys, including the elements and itself.

    ``"Li-Fe-O"`` -> the 7 non-empty combinations (``Fe``, ``Li``, ``O``,
    ``Fe-Li``, ...). Needed because MP's ``chemsys`` matches a material's
    system *exactly*: querying ``"Si-O"`` returns the binaries only -- not
    pure Si, not pure O -- which is rarely what a reference database wants.
    """
    from itertools import combinations

    parts = sorted({p.strip() for p in chemsys.split("-")})
    if "*" in parts:  # a wildcard already spans systems; nothing to expand
        return [chemsys]
    return [
        "-".join(combo)
        for n in range(1, len(parts) + 1)
        for combo in combinations(parts, n)
    ]


def _as_range(value) -> tuple:
    """Normalise an ``int``/``(lo, hi)`` filter into MP's ``(lo, hi)`` form.

    A bare number reads as an upper bound (``max_sites=20`` -> ``(1, 20)``).
    """
    if isinstance(value, (tuple, list)):
        lo, hi = value
        return (int(lo), int(hi))
    return (1, int(value))


def _sort_key(atoms: Atoms) -> tuple:
    """Most-stable-first, ties broken by ``mp_id`` (a stable, reproducible
    order -- MP's own result order is not guaranteed between calls)."""
    e_hull = atoms.info.get("e_above_hull")
    return (
        float("inf") if e_hull is None else float(e_hull),
        str(atoms.info.get("mp_id", "")),
    )


def mp_query_kwargs(params) -> dict:
    """Pick the :func:`fetch_mp_structures` query filters out of a mapping.

    Lets a caller carrying a loose parameter bag (a dashboard
    ``ReferenceNeed``'s params, a parsed config) forward only the keys the
    fetch understands, ignoring its own (``target_cutoff``, ``level``, ...).
    ``None`` values are dropped so the fetch's own defaults win.
    """
    return {
        k: params[k]
        for k in MP_QUERY_KEYS
        if params.get(k) is not None and params.get(k) is not False
    }


def fetch_mp_structures(
    system: Optional[str | Sequence[str]] = None,
    *,
    api_key: Optional[str] = None,
    theoretical: bool = False,
    chemsys: Optional[str | Sequence[str]] = None,
    formula: Optional[str | Sequence[str]] = None,
    include_subsystems: bool = False,
    elements: Optional[Sequence[str]] = None,
    exclude_elements: Optional[Sequence[str]] = None,
    num_elements: Optional[int | tuple] = None,
    max_e_above_hull: Optional[float] = None,
    only_stable: bool = False,
    max_sites: Optional[int] = None,
    max_per_composition: Optional[int] = None,
    max_structures: Optional[int] = None,
    **extra_query,
) -> list[Atoms]:
    """Fetch a system's phases from the Materials Project as ``ase.Atoms``.

    The general fetch behind :func:`fetch_mp_elemental`: ``system`` is an
    element (``"Si"``), a formula (``"SiO2"``) or a chemical system
    (``"Si-O"``, ``["Li", "Fe", "O"]``) -- see :func:`_parse_system` for the
    dispatch rule -- or pass ``chemsys=``/``formula=`` explicitly to bypass it.

    Every returned structure is tagged with ``info["mp_id"]``,
    ``info["spacegroup"]``, ``info["formula"]``, ``info["e_above_hull"]`` and
    ``info["is_stable"]``, and the list is ordered **most stable first** (ties
    by ``mp_id``) so the output is reproducible and a ``max_structures`` cut
    keeps the physically relevant phases.

    The filters matter far more for compounds than for elements: a binary
    system holds hundreds of polymorphs where an element holds a handful, and
    each one is multiplied by the rattle grid downstream. ``max_e_above_hull``
    (hull distance in eV/atom) and ``max_sites`` are the usual two knobs.

    Args:
        system: Element symbol, formula, chemsys string, or a sequence of
            symbols. Mutually exclusive with ``chemsys``/``formula``.
        api_key: Materials Project API key; falls back to ``$MP_API_KEY``.
        theoretical: If ``False`` (default), keep only experimentally
            observed phases.
        chemsys: Explicit ``chemsys`` constraint (``"Si-O"``, ``"*-*-O"``, or
            a list of those) -- matches materials of *exactly* those elements.
        formula: Explicit ``formula`` constraint (``"SiO2"``, ``"ABC3"``) --
            matches every polymorph of that stoichiometry.
        include_subsystems: For a chemsys query, also fetch every sub-system
            (``"Li-Fe-O"`` -> also ``Li-O``, ``Fe``, ...). What you usually
            want for a reference database, since the end members and binaries
            bound the composition space the sampler explores.
        elements: Additionally require these elements to be present.
        exclude_elements: Drop materials containing any of these.
        num_elements: Cap (or ``(lo, hi)`` range) on the number of distinct
            elements -- useful with a wildcard chemsys.
        max_e_above_hull: Keep only phases within this hull distance
            (eV/atom); ``0.0`` means on the hull.
        only_stable: Keep only MP's ``is_stable`` phases (stricter than
            ``max_e_above_hull=0`` -- it also excludes metastable duplicates).
        max_sites: Skip cells with more than this many atoms -- a DFT-cost
            guard, since each phase is expanded to a supercell later.
        max_per_composition: Keep at most this many polymorphs per
            composition (the most stable ones). The compound-system knob:
            it trims the polymorph tail of one stoichiometry without letting
            it crowd out the others.
        max_structures: Keep at most this many phases overall (the most
            stable ones). Applied *after* ``max_per_composition``; on its own
            it is composition-blind, so a system whose elements sit on the
            hull can lose its compounds to them -- combine the two.
        **extra_query: Passed straight to ``summary.search`` (any filter
            ``mp_api`` supports, e.g. ``band_gap=(0, 0.1)``).

    Returns:
        The fetched structures as ``ase.Atoms``.
    """
    given = {
        "system": system,
        "chemsys": chemsys,
        "formula": formula,
    }
    named = [k for k, v in given.items() if v is not None]
    if "system" in named and len(named) > 1:
        raise ValueError(
            "Pass system= or chemsys=/formula=, not both "
            f"(got {', '.join(sorted(named))})."
        )
    if not named:
        raise ValueError(
            "No system given: pass an element ('Si'), a formula ('SiO2') or "
            "a chemsys ('Si-O')."
        )

    if system is not None:
        constraint = _parse_system(system)
    else:
        constraint = {}
        if chemsys is not None:
            constraint["chemsys"] = chemsys
        if formula is not None:
            constraint["formula"] = formula

    if include_subsystems:
        if "chemsys" not in constraint:
            logger.warning(
                "include_subsystems is ignored for a formula query (%s); it "
                "only applies to a chemical system.",
                constraint.get("formula"),
            )
        else:
            spec = constraint["chemsys"]
            spec = [spec] if isinstance(spec, str) else list(spec)
            expanded = {s for one in spec for s in _expand_subsystems(one)}
            constraint["chemsys"] = sorted(expanded)

    query: dict = dict(constraint)
    query["theoretical"] = theoretical
    if elements is not None:
        query["elements"] = list(elements)
    if exclude_elements is not None:
        query["exclude_elements"] = list(exclude_elements)
    if num_elements is not None:
        query["num_elements"] = _as_range(num_elements)
    if max_e_above_hull is not None:
        query["energy_above_hull"] = (0.0, float(max_e_above_hull))
    if only_stable:
        query["is_stable"] = True
    if max_sites is not None:
        query["num_sites"] = _as_range(max_sites)
    query.update(extra_query)

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
            "fetch_mp_structures needs mp_api + pymatgen; install the "
            "'refdb' extra: pip install 'jaxrens_active_learning[refdb]'."
        ) from exc

    logger.info("Materials Project query: %s", query)
    with MPRester(key) as mpr:
        results = mpr.materials.summary.search(
            fields=list(_MP_FIELDS), **query
        )

    structures: list[Atoms] = []
    for r in results:
        at = AseAtomsAdaptor.get_atoms(r.structure)
        at.info["mp_id"] = str(r.material_id)
        symmetry = getattr(r, "symmetry", None)
        if symmetry is not None:
            at.info["spacegroup"] = str(symmetry.symbol)
        for key_name, attr in (
            ("formula", "formula_pretty"),
            ("e_above_hull", "energy_above_hull"),
            ("is_stable", "is_stable"),
        ):
            value = getattr(r, attr, None)
            if value is not None:  # never write None into info (extxyz)
                at.info[key_name] = value
        structures.append(at)

    structures.sort(key=_sort_key)
    if max_per_composition is not None:
        kept: list[Atoms] = []
        seen: dict = {}
        for at in structures:  # sorted -> the kept ones are the stablest
            comp = at.info.get("formula", at.get_chemical_formula())
            if seen.get(comp, 0) >= int(max_per_composition):
                continue
            seen[comp] = seen.get(comp, 0) + 1
            kept.append(at)
        if len(kept) < len(structures):
            logger.info(
                "Keeping <=%d polymorph(s) per composition: %d of %d "
                "phase(s) over %d composition(s).",
                int(max_per_composition),
                len(kept),
                len(structures),
                len(seen),
            )
        structures = kept
    if max_structures is not None and len(structures) > int(max_structures):
        dropped = len(structures) - int(max_structures)
        structures = structures[: int(max_structures)]
        logger.info(
            "Keeping the %d most stable phase(s); dropped %d.",
            len(structures),
            dropped,
        )
    logger.info(
        "Fetched %d structure(s) from Materials Project.", len(structures)
    )
    return structures


def fetch_mp_elemental(
    element: str,
    *,
    api_key: Optional[str] = None,
    theoretical: bool = False,
    **query,
) -> list[Atoms]:
    """Fetch elemental phases of ``element`` from the Materials Project.

    The single-element special case of :func:`fetch_mp_structures` (kept as
    the historical entry point): queries the one-element chemical system, so
    compounds are excluded.

    Args:
        element: Chemical symbol, e.g. ``"Si"``.
        api_key: Materials Project API key; falls back to the ``MP_API_KEY``
            environment variable.
        theoretical: If ``False`` (default), keep only experimentally observed
            phases.
        **query: Further :func:`fetch_mp_structures` filters
            (``max_e_above_hull``, ``max_sites``, ...).

    Returns:
        The fetched structures as ``ase.Atoms``.
    """
    return fetch_mp_structures(
        chemsys=element, api_key=api_key, theoretical=theoretical, **query
    )


# ---------------------------------------------------------------------------
# One-call orchestration.
# ---------------------------------------------------------------------------


def resolve_system(
    system: Optional[str | Sequence[str]] = None,
    element: Optional[str] = None,
) -> str | Sequence[str]:
    """Accept either the general ``system=`` or the legacy ``element=``.

    The compatibility seam every general entry point shares: ``element`` was
    the original (elemental-only) argument name and stays valid -- it just no
    longer has to name an element, since anything :func:`_parse_system`
    understands works.
    """
    if system is not None and element is not None:
        raise ValueError("Pass system= or element=, not both.")
    resolved = system if system is not None else element
    if resolved is None:
        raise ValueError(
            "No system given: pass an element ('Si'), a formula ('SiO2') or "
            "a chemsys ('Si-O')."
        )
    return resolved


def build_reference_database(
    system: Optional[str | Sequence[str]] = None,
    *,
    element: Optional[str] = None,
    api_key: Optional[str] = None,
    theoretical: bool = False,
    target_cutoff: float = 3.0,
    rattle_strengths: Sequence[float] = (0.0, 0.2),
    vol_scalings: Optional[Sequence[float]] = None,
    num_realizations: int = 1,
    seed: Optional[int] = None,
    output: Optional[str | Path] = None,
    **query,
) -> list[Atoms]:
    """Fetch a system's phases, rattle them, and (optionally) write an extxyz.

    Thin wrapper tying :func:`fetch_mp_structures` to :func:`rattle_atoms`;
    the knobs mirror those functions. If ``output`` is given the
    configurations are written there with ``ase.io.write``.

    Args:
        system: Element (``"Si"``), formula (``"SiO2"``) or chemsys
            (``"Si-O"``) -- see :func:`fetch_mp_structures`.
        element: Legacy alias for ``system`` (kept working; may itself hold a
            formula or chemsys).
        api_key, theoretical: Passed to the fetch.
        target_cutoff, rattle_strengths, vol_scalings, num_realizations,
            seed: Passed to :func:`rattle_atoms`.
        output: Optional extxyz path to write.
        **query: Further fetch filters -- ``include_subsystems``,
            ``max_e_above_hull``, ``max_sites``, ``max_structures``,
            ``only_stable``, ``exclude_elements``, ... Reach for these on a
            compound: a binary system has *far* more phases than an element,
            and each is multiplied by the rattle grid.

    Returns:
        The starting-database configurations.
    """
    structures = fetch_mp_structures(
        resolve_system(system, element),
        api_key=api_key,
        theoretical=theoretical,
        **query,
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
# CLI: jaxrens-al refdb <element> <output.extxyz>
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Console-script entry point. Returns a process exit code."""
    import argparse
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        prog="jaxrens-al refdb",
        description="Build a starting structure database for an AL run: "
        "fetch a system's experimentally observed phases from the Materials "
        "Project, rattle + volume-scale them, write an extxyz.",
        epilog="SYSTEM is an element (Si -> all pure-Si polymorphs), a "
        "formula (SiO2 -> every polymorph of that stoichiometry) or a "
        "chemical system (Si-O -> every Si-O compound; add "
        "--include-subsystems for the elements too).",
    )
    parser.add_argument(
        "system",
        metavar="SYSTEM",
        help="Element (Si), formula (SiO2) or chemsys (Si-O, Li-Fe-O).",
    )
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
    mp_group = parser.add_argument_group(
        "Materials Project filters",
        "Narrow the phase set -- a compound system holds far more phases "
        "than an element, and each is multiplied by the rattle grid.",
    )
    mp_group.add_argument(
        "--include-subsystems",
        action="store_true",
        help="For a chemsys, also fetch every sub-system (Li-Fe-O -> also "
        "Li-O, Fe, ...). Usually wanted: a bare chemsys matches materials of "
        "EXACTLY those elements, so 'Si-O' excludes pure Si and pure O.",
    )
    mp_group.add_argument(
        "--max-e-above-hull",
        type=float,
        default=None,
        metavar="EV",
        help="Keep only phases within this hull distance (eV/atom).",
    )
    mp_group.add_argument(
        "--only-stable",
        action="store_true",
        help="Keep only phases Materials Project marks stable.",
    )
    mp_group.add_argument(
        "--max-sites",
        type=int,
        default=None,
        metavar="N",
        help="Skip cells with more than N atoms (DFT-cost guard).",
    )
    mp_group.add_argument(
        "--max-per-composition",
        type=int,
        default=None,
        metavar="N",
        help="Keep at most N polymorphs per composition (the most stable) -- "
        "trims one stoichiometry's tail without crowding out the others.",
    )
    mp_group.add_argument(
        "--max-structures",
        type=int,
        default=None,
        metavar="N",
        help="Keep at most N phases overall (the most stable ones). "
        "Composition-blind: pair it with --max-per-composition.",
    )
    mp_group.add_argument(
        "--num-elements",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of distinct elements per phase.",
    )
    mp_group.add_argument(
        "--exclude-elements",
        nargs="+",
        default=None,
        metavar="SYM",
        help="Drop phases containing any of these elements.",
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
            args.system,
            api_key=args.api_key,
            target_cutoff=args.target_cutoff,
            rattle_strengths=args.rattle_strengths,
            vol_scalings=vol_scalings,
            num_realizations=args.num_realizations,
            seed=args.seed,
            output=args.output,
            **mp_query_kwargs(vars(args)),
        )
    except (ValueError, ImportError) as exc:
        logger.error(str(exc))
        return 1
    logger.info(f"Wrote {len(configs)} configs to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
