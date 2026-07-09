"""Tests for the one-shot reference evaluation orchestration.

``jaxrens_al.run.reference_eval`` drives a ground-truth evaluator over named config
sets purely through the ``update`` / ``get_evaluated_configs`` contract, so a
tiny fake evaluator (no ``remote_handler``, no VASP) exercises the whole flow --
per-set working dirs, labelling, and the concatenation into one reference db.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

pytest.importorskip("ase")

from jaxrens_al.run.reference_eval import (  # noqa: E402
    concatenate_configs,
    evaluate_config_sets,
)

pytestmark = pytest.mark.local


def _atoms(n=2):
    return Atoms(
        "Si" * n, positions=np.zeros((n, 3)), cell=np.eye(3) * 5, pbc=True
    )


class _FakeEvaluator:
    """Mimics remote_handler's GroundTruthEvaluator (update + collect)."""

    def update(self, calcdir, evaluator_input, **kwargs):
        self._input = Path(evaluator_input)
        self._out = (
            Path(calcdir) / "dft" / "results" / "evaluated_configs.extxyz"
        )

    def get_evaluated_configs(self):
        import ase.io

        frames = ase.io.read(str(self._input), index=":")
        for at in frames:
            at.info["REF_energy"] = -1.0 * len(at)
            at.arrays["REF_forces"] = np.zeros((len(at), 3))
        self._out.parent.mkdir(parents=True, exist_ok=True)
        ase.io.write(str(self._out), frames)
        return self._out


def test_evaluate_config_sets_labels_each_set(tmp_path):
    import ase.io

    sets = {"eos_mp-149": [_atoms(2)], "pool": [_atoms(2), _atoms(3)]}
    labelled = evaluate_config_sets(_FakeEvaluator(), sets, tmp_path)

    assert set(labelled) == {"eos_mp-149", "pool"}
    # Per-set working dirs keep the outputs apart.
    assert labelled["eos_mp-149"] != labelled["pool"]
    for path in labelled.values():
        frames = ase.io.read(str(path), index=":")
        assert all("REF_energy" in at.info for at in frames)


def test_evaluate_config_sets_skips_empty(tmp_path):
    labelled = evaluate_config_sets(
        _FakeEvaluator(), {"a": [_atoms()], "empty": []}, tmp_path
    )
    assert set(labelled) == {"a"}


def test_concatenate_configs_merges_labelled_files(tmp_path):
    import ase.io

    sets = {"eos": [_atoms(2), _atoms(2)], "pool": [_atoms(3)]}
    labelled = evaluate_config_sets(_FakeEvaluator(), sets, tmp_path)
    ref_db = tmp_path / "reference_db.extxyz"

    n = concatenate_configs(list(labelled.values()), ref_db)
    assert n == 3 and ref_db.exists()
    assert len(ase.io.read(str(ref_db), index=":")) == 3


def test_build_eos_reference_writes_per_phase_files(tmp_path, monkeypatch):
    import ase.io

    import jaxrens_al.run.reference_db as refdb
    from jaxrens_al.run.reference_eval import build_eos_reference

    def _phase(mp_id):
        at = Atoms(
            "Sn2", positions=np.zeros((2, 3)), cell=np.eye(3) * 6, pbc=True
        )
        at.info["mp_id"] = mp_id
        return at

    monkeypatch.setattr(
        refdb,
        "fetch_mp_elemental",
        lambda *a, **k: [_phase("mp-1"), _phase("mp-2")],
    )

    class _EosEvaluator:
        def update(self, calcdir, evaluator_input, **kw):
            self._in = str(evaluator_input)
            self._out = tmp_path / "out" / str(len(str(calcdir))) / "e.extxyz"

        def get_evaluated_configs(self):
            frames = ase.io.read(self._in, index=":")
            for at in frames:
                at.info["REF_energy"] = -1.0 * len(at)
                at.arrays["REF_forces"] = np.zeros((len(at), 3))
            self._out.parent.mkdir(parents=True, exist_ok=True)
            ase.io.write(str(self._out), frames)
            return self._out

    files = build_eos_reference(
        _EosEvaluator(),
        element="Sn",
        vol_scalings=[0.98, 1.0, 1.02],
        target_cutoff=2.0,
        out_dir=tmp_path / "ref",
        workdir=tmp_path / "work",
    )
    assert {f.name for f in files} == {"eos_mp-1.extxyz", "eos_mp-2.extxyz"}
    for f in files:
        frames = ase.io.read(str(f), index=":")
        assert len(frames) == 3  # one per volume scaling
        assert all("REF_energy" in at.info for at in frames)
