"""Tests for the one-shot reference evaluation orchestration.

``jaxrens_al.reference_eval`` drives a ground-truth evaluator over named config
sets purely through the ``update`` / ``get_evaluated_configs`` contract, so a
tiny fake evaluator (no ``remote_handler``, no VASP) exercises the whole flow --
per-set working dirs, labelling, and the concatenation into one reference db.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

pytest.importorskip("ase")

from jaxrens_al.reference_eval import (  # noqa: E402
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
