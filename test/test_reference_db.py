"""Tests for the starting-database builder (jaxrens_al.run.reference_db).

Only the numpy + ase half is exercised here -- the supercell search and the
rattling -- since the Materials Project fetch needs network + the [refdb]
extra. The MP path is covered just for its no-key/no-dep error handling.
"""

import sys
import types

import numpy as np
import pytest
from ase import Atoms

pytest.importorskip("ase")

from jaxrens_al.run.reference_db import (  # noqa: E402
    _expand_subsystems,
    _hnf_diagonals,
    _max_cutoff,
    _parse_system,
    build_reference_database,
    eos_phase_scans,
    fetch_mp_elemental,
    fetch_mp_structures,
    find_supercell,
    mp_query_kwargs,
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
    monkeypatch.setattr(refdb, "fetch_mp_structures", lambda *a, **k: [base])
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


def test_build_reference_database_accepts_legacy_element_kwarg(
    tmp_path, monkeypatch
):
    """``element=`` predates general systems; it must keep working (and may
    itself now hold a formula/chemsys, forwarded verbatim to the fetch)."""
    import jaxrens_al.run.reference_db as refdb

    seen = {}

    def _fetch(system, **kwargs):
        seen["system"] = system
        seen.update(kwargs)
        return [_bulk_si()]

    monkeypatch.setattr(refdb, "fetch_mp_structures", _fetch)
    build_reference_database(
        element="SiO2",
        vol_scalings=[1.0],
        rattle_strengths=(0.0,),
        target_cutoff=2.0,
        max_e_above_hull=0.05,  # a query filter forwarded to the fetch
    )
    assert seen["system"] == "SiO2"
    assert seen["max_e_above_hull"] == 0.05

    with pytest.raises(ValueError, match="not both"):
        build_reference_database("Si", element="Si")
    with pytest.raises(ValueError, match="No system given"):
        build_reference_database()


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


def test_eos_phase_scans_can_label_by_composition():
    # A multi-compound scan set: bare mp- ids say nothing about WHICH
    # compound a curve belongs to, so the label can carry the formula.
    a, b = _bulk_si(), _bulk_si(6.0)
    a.info.update(mp_id="mp-149", formula="Si")
    b.info.update(mp_id="mp-6930", formula="SiO2")
    plain = eos_phase_scans([a, b], vol_scalings=[1.0], target_cutoff=None)
    assert set(plain) == {"mp-149", "mp-6930"}  # default: unchanged
    labelled = eos_phase_scans(
        [a, b], vol_scalings=[1.0], target_cutoff=None, label_formula=True
    )
    assert set(labelled) == {"Si_mp-149", "SiO2_mp-6930"}


def test_eos_phase_scans_none_cutoff_keeps_unit_cell():
    # target_cutoff=None -> no supercell: the DFT reference stays unit-cell,
    # the cutoff-hosting supercell is built later at model-eval time.
    base = _bulk_si(5.43)  # 8-atom cell, max sphere radius 2.715 < any cutoff
    (scan,) = eos_phase_scans(
        [base], vol_scalings=[1.0], target_cutoff=None
    ).values()
    assert len(scan[0]) == len(base)  # not replicated
    # ... whereas a real cutoff would have inflated it.
    (super_scan,) = eos_phase_scans(
        [base], vol_scalings=[1.0], target_cutoff=6.0
    ).values()
    assert len(super_scan[0]) > len(base)


def test_min_image_supercell_hosts_cutoff_and_is_noop_when_large():
    from jaxrens_al.supercell import _max_cutoff, min_image_supercell

    base = _bulk_si(5.43)
    sup = min_image_supercell(base, target_cutoff=6.0)
    assert len(sup) % len(base) == 0 and len(sup) > len(base)
    assert _max_cutoff(sup.cell[...]) > 6.0
    # Already large enough -> a plain copy, input untouched.
    same = min_image_supercell(base, target_cutoff=2.0)
    assert len(same) == len(base)


# --------------------------------------------------------------------------
# MP fetch error handling (no network needed)
# --------------------------------------------------------------------------
def test_fetch_requires_api_key(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        fetch_mp_elemental("Si")
    with pytest.raises(ValueError, match="API key"):
        fetch_mp_structures("SiO2")


def test_fetch_requires_a_system(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "dummy")
    with pytest.raises(ValueError, match="No system given"):
        fetch_mp_structures()
    with pytest.raises(ValueError, match="not both"):
        fetch_mp_structures("SiO2", chemsys="Si-O")


# --------------------------------------------------------------------------
# System spec -> Materials Project query constraint
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "system, expected",
    [
        ("Si", {"chemsys": "Si"}),  # element -> the one-element system
        ("Sn ", {"chemsys": "Sn"}),
        ("Si-O", {"chemsys": "O-Si"}),  # chemsys, alphabetised like MP's
        ("O-Si", {"chemsys": "O-Si"}),
        ("Li-Fe-O", {"chemsys": "Fe-Li-O"}),
        ("*-*-O", {"chemsys": "*-*-O"}),  # wildcards keep their order
        (["O", "Si"], {"chemsys": "O-Si"}),
        (("Si", "O"), {"chemsys": "O-Si"}),
        ("SiO2", {"formula": "SiO2"}),  # anything else is a formula
        ("Li2FeSiO4", {"formula": "Li2FeSiO4"}),
        ("ABC3", {"formula": "ABC3"}),
    ],
)
def test_parse_system_dispatch(system, expected):
    assert _parse_system(system) == expected


@pytest.mark.parametrize("bad", ["", "  ", [], "Si-Xx", ["Si", "Xx"]])
def test_parse_system_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        _parse_system(bad)


def test_expand_subsystems_covers_every_combination():
    # MP's chemsys matches EXACTLY those elements, so a Si-O query returns no
    # pure Si; the expansion is what puts the end members back in.
    assert set(_expand_subsystems("Si-O")) == {"O", "Si", "O-Si"}
    assert set(_expand_subsystems("Li-Fe-O")) == {
        "Fe",
        "Li",
        "O",
        "Fe-Li",
        "Fe-O",
        "Li-O",
        "Fe-Li-O",
    }
    assert _expand_subsystems("*-O") == ["*-O"]  # nothing to expand


# --------------------------------------------------------------------------
# Query assembly + post-filters (a stubbed MPRester -- no network)
# --------------------------------------------------------------------------
class _FakeSummary:
    def __init__(self, results):
        self._results = results
        self.query = None

    def search(self, **kwargs):
        kwargs.pop("fields", None)
        self.query = kwargs
        return self._results


class _FakeDoc:
    """The subset of an MP summary doc the fetch reads."""

    def __init__(self, mp_id, formula, e_hull, nsites=2, spacegroup="Fm-3m"):
        self.material_id = mp_id
        self.formula_pretty = formula
        self.energy_above_hull = e_hull
        self.is_stable = e_hull == 0.0
        self.nsites = nsites
        self.symmetry = type("S", (), {"symbol": spacegroup})()
        self.structure = None


def _fake_mp(monkeypatch, results):
    """Stub mp_api + pymatgen so the fetch runs offline; returns the summary
    stub, whose ``.query`` records what was asked of the server."""
    summary = _FakeSummary(results)

    class _Rester:
        def __init__(self, key):
            self.key = key

        def __enter__(self):
            self.materials = type("M", (), {"summary": summary})()
            return self

        def __exit__(self, *exc):
            return False

    mp_client = types.ModuleType("mp_api.client")
    mp_client.MPRester = _Rester
    mp_pkg = types.ModuleType("mp_api")
    mp_pkg.client = mp_client
    adaptor = types.ModuleType("pymatgen.io.ase")
    adaptor.AseAtomsAdaptor = type(
        "A", (), {"get_atoms": staticmethod(lambda structure: _bulk_si())}
    )
    for name, mod in [
        ("mp_api", mp_pkg),
        ("mp_api.client", mp_client),
        ("pymatgen", types.ModuleType("pymatgen")),
        ("pymatgen.io", types.ModuleType("pymatgen.io")),
        ("pymatgen.io.ase", adaptor),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("MP_API_KEY", "dummy")
    return summary


def test_fetch_builds_the_expected_query(monkeypatch):
    summary = _fake_mp(monkeypatch, [])
    fetch_mp_structures(
        "Li-Fe-O",
        include_subsystems=True,
        max_e_above_hull=0.05,
        max_sites=20,
        num_elements=2,
        exclude_elements=["H"],
        only_stable=True,
    )
    assert summary.query == {
        # every sub-system, so the end members and binaries come along
        "chemsys": ["Fe", "Fe-Li", "Fe-Li-O", "Fe-O", "Li", "Li-O", "O"],
        "theoretical": False,
        "exclude_elements": ["H"],
        "num_elements": (1, 2),
        "energy_above_hull": (0.0, 0.05),
        "is_stable": True,
        "num_sites": (1, 20),
    }


def test_fetch_elemental_query_is_the_single_element_system(monkeypatch):
    # The historical entry point keeps its meaning: pure-element polymorphs.
    summary = _fake_mp(monkeypatch, [])
    fetch_mp_elemental("Si", theoretical=True)
    assert summary.query == {"chemsys": "Si", "theoretical": True}


def test_fetch_ignores_subsystems_for_a_formula(monkeypatch):
    summary = _fake_mp(monkeypatch, [])
    fetch_mp_structures("SiO2", include_subsystems=True)
    assert summary.query == {"formula": "SiO2", "theoretical": False}


def test_fetch_tags_and_orders_by_stability(monkeypatch):
    _fake_mp(
        monkeypatch,
        [
            _FakeDoc("mp-2", "SiO2", 0.30),
            _FakeDoc("mp-1", "Si", 0.00),
            _FakeDoc("mp-3", "SiO2", 0.10),
        ],
    )
    got = fetch_mp_structures("Si-O", include_subsystems=True)
    assert [at.info["mp_id"] for at in got] == ["mp-1", "mp-3", "mp-2"]
    assert got[0].info["formula"] == "Si"
    assert got[0].info["is_stable"] is True
    assert got[0].info["spacegroup"] == "Fm-3m"


def test_fetch_caps_keep_the_most_stable(monkeypatch):
    docs = [
        _FakeDoc("mp-1", "Si", 0.00),
        _FakeDoc("mp-2", "SiO2", 0.01),
        _FakeDoc("mp-3", "SiO2", 0.02),
        _FakeDoc("mp-4", "SiO2", 0.03),
    ]
    _fake_mp(monkeypatch, docs)
    got = fetch_mp_structures("Si-O", max_structures=2)
    assert [at.info["mp_id"] for at in got] == ["mp-1", "mp-2"]

    _fake_mp(monkeypatch, docs)
    # Per-composition cap: the SiO2 polymorph tail is trimmed, but SiO2 is
    # not crowded out by the (more stable) element -- what max_structures
    # alone would do.
    got = fetch_mp_structures("Si-O", max_per_composition=1)
    assert [at.info["mp_id"] for at in got] == ["mp-1", "mp-2"]
    assert {at.info["formula"] for at in got} == {"Si", "SiO2"}


def test_mp_query_kwargs_picks_only_query_filters():
    picked = mp_query_kwargs(
        {
            "element": "Si",  # the system itself is not a filter
            "target_cutoff": 3.0,  # a rattling knob
            "level": "dft",  # a caller's own key
            "max_e_above_hull": 0.05,
            "max_sites": None,  # unset -> the fetch default wins
            "theoretical": False,  # off -> not forwarded
            "include_subsystems": True,
        }
    )
    assert picked == {"max_e_above_hull": 0.05, "include_subsystems": True}
