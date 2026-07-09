"""Tests for the starting-database builder (jaxrens_al.run.reference_db).

Only the numpy + ase half is exercised here -- the supercell search and the
rattling -- since the Materials Project fetch needs network + the [refdb]
extra. The MP path is covered just for its no-key/no-dep error handling.
"""

import numpy as np
import pytest
from ase import Atoms

pytest.importorskip("ase")

from jaxrens_al.run.reference_db import (  # noqa: E402
    _hnf_diagonals,
    _max_cutoff,
    build_reference_database,
    eos_phase_scans,
    fetch_mp_elemental,
    find_supercell,
    rattle_atoms,
)

pytestmark = pytest.mark.local


def _bulk_si(a=5.43):
    """A small cubic 8-atom diamond-Si cell (only the geometry matters)."""
    return Atoms(
        "Si8",
        cell=[a, a, a],
        pbc=True,
        scaled_positions=[
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
    )


# --------------------------------------------------------------------------
# Supercell search
# --------------------------------------------------------------------------
def test_hnf_diagonals_have_target_determinant():
    diags = _hnf_diagonals(12)
    assert diags.shape[1] == 3
    assert all(int(np.prod(d)) == 12 for d in diags)
    # The 1x1x cube of det 12 includes e.g. [3, 2, 2] in some order.
    assert any(sorted(d) == [2, 2, 3] for d in diags.tolist())


def test_find_supercell_hosts_cutoff():
    cell = np.eye(3) * 5.43  # max sphere radius without scaling = 2.715
    trafo = find_supercell(cell, target_cutoff=6.0)
    super_cell = cell * np.asarray(trafo)[:, None]
    assert _max_cutoff(super_cell) > 6.0
    # No replication needed when the cutoff already fits.
    assert list(find_supercell(cell, target_cutoff=2.0)) == [1, 1, 1]


# --------------------------------------------------------------------------
# Rattling
# --------------------------------------------------------------------------
def test_rattle_counts_and_tags():
    base = _bulk_si()
    vol_scalings = np.linspace(0.95, 1.05, 4)
    out = rattle_atoms(
        [base],
        rattle_strengths=(0.0, 0.1, 0.2),
        vol_scalings=vol_scalings,
        num_realizations=2,
        target_cutoff=3.0,
        seed=0,
    )
    # Per volume: zero-strength once + two non-zero strengths x 2 draws = 5.
    assert len(out) == len(vol_scalings) * 5
    for at in out:
        assert "vol_scaling" in at.info and "rattle_strength" in at.info
    # Zero-strength copies keep the (scaled) ideal positions; strong ones move.
    zero = [a for a in out if a.info["rattle_strength"] == 0.0]
    assert len(zero) == len(vol_scalings)


def test_rattle_scales_volume_and_is_seeded():
    base = _bulk_si()
    a = rattle_atoms(
        [base],
        vol_scalings=[1.2],
        rattle_strengths=(0.0,),
        target_cutoff=2.0,
        seed=7,
    )[0]
    # Factor scales the cell vectors -> volume scales by the cube (legacy).
    assert a.get_volume() == pytest.approx(base.get_volume() * 1.2**3)
    # Same seed -> identical jitter.
    one = rattle_atoms(
        [base],
        vol_scalings=[1.0],
        rattle_strengths=(0.3,),
        target_cutoff=2.0,
        seed=42,
    )[0]
    two = rattle_atoms(
        [base],
        vol_scalings=[1.0],
        rattle_strengths=(0.3,),
        target_cutoff=2.0,
        seed=42,
    )[0]
    assert np.allclose(one.positions, two.positions)


def test_build_reference_database_writes_extxyz(tmp_path, monkeypatch):
    import jaxrens_al.run.reference_db as refdb

    base = _bulk_si()
    monkeypatch.setattr(refdb, "fetch_mp_elemental", lambda *a, **k: [base])
    out = tmp_path / "ref.extxyz"
    configs = build_reference_database(
        "Si",
        vol_scalings=[1.0],
        rattle_strengths=(0.0,),
        target_cutoff=2.0,
        output=out,
    )
    assert out.exists() and out.stat().st_size > 0
    assert len(configs) == 1


# --------------------------------------------------------------------------
# eos_phase_scans: per-phase, zero-rattle volume scans
# --------------------------------------------------------------------------
def test_eos_phase_scans_zero_rattle_per_phase():
    a, b = _bulk_si(), _bulk_si(6.0)
    a.info["mp_id"], b.info["mp_id"] = "mp-149", "mp-165"
    scans = eos_phase_scans(
        [a, b], vol_scalings=np.linspace(0.95, 1.05, 5), target_cutoff=2.0
    )
    assert set(scans) == {"mp-149", "mp-165"}
    for scan in scans.values():
        # One config per volume (zero rattle emits a single copy per volume).
        assert len(scan) == 5
        assert all(at.info["rattle_strength"] == 0.0 for at in scan)
        vols = sorted(at.get_volume() for at in scan)
        assert vols[0] < vols[-1]  # a real volume spread to fit an EOS to


def test_eos_phase_scans_sanitizes_labels_for_filenames():
    a = _bulk_si()
    a.info["spacegroup"] = "P6_3/mmc"  # space-group symbols carry a slash
    (label,) = eos_phase_scans([a], vol_scalings=[0.98, 1.0, 1.02])
    assert "/" not in label and label == "P6_3_mmc"


# --------------------------------------------------------------------------
# MP fetch error handling (no network needed)
# --------------------------------------------------------------------------
def test_fetch_requires_api_key(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        fetch_mp_elemental("Si")
