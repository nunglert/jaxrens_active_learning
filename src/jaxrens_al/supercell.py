"""Minimum-image supercell search (numpy + ase only, base-install safe).

A small numpy-only port of jaxnest's ``structure_utils.find_super``: the
smallest diagonal supercell of a cell that hosts a target interaction cutoff
(so a minimum-image neighbour list is valid). This lives at the package root --
next to :mod:`jaxrens_al.keys` -- because **both** sides need it without
crossing the dashboard/run seam:

* the run side builds cutoff-hosting supercells for reference databases
  (:mod:`jaxrens_al.run.reference_db` re-exports these);
* the dashboard side expands a unit-cell reference frame to a supercell *at
  model-eval time* (:func:`jaxrens_al.dashboard.panels.model_eval.score_atoms`),
  so the DFT reference can stay on the cheap single unit cells while the
  force-field re-score still sees a cell large enough for its neighbour list.

No jax/numba/remote_handler -- importing this stays cheap on either path.
"""

from __future__ import annotations

import numpy as np


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


def min_image_supercell(atoms, target_cutoff: float):
    """Replicate ``atoms`` into the smallest supercell hosting ``target_cutoff``.

    Returns a new ``ase.Atoms`` (the input is not mutated). Already large enough
    -> the transform is ``[1, 1, 1]`` and a plain copy is returned, so this is
    safe to call unconditionally. The replication factor is
    ``len(result) // len(atoms)``.
    """
    trafo = find_supercell(atoms.cell[...], target_cutoff=target_cutoff)
    return atoms * tuple(int(t) for t in trafo)
