"""End-to-end test of the run -> dashboard bridge.

Builds a *synthetic* remote_handler run directory (calchome tree +
AL_progress.json + DFT-labelled extxyz), then drives the composition root's
``build_dashboard`` through to a saved figure. Needs only alviz + ase -- no
cluster, no remote_handler import.
"""

import json

import numpy as np
import pytest

import ase.io 
from ase import Atoms 

from jaxrens_al import build_dashboard 
from jaxrens_al.run_layout import ( 
    evaluated_configs,
    load_progress,
)


def _labelled_configs(rng, n_configs, n_atoms=4):
    configs = []
    for _ in range(n_configs):
        atoms = Atoms(
            "H" * n_atoms,
            positions=rng.uniform(0, 5, size=(n_atoms, 3)),
            cell=np.eye(3) * rng.uniform(8, 12),
            pbc=True,
        )
        # remote_handler ground-truth label convention.
        atoms.info["REF_energy"] = float(rng.uniform(-10, -5))
        atoms.arrays["REF_forces"] = rng.normal(size=(n_atoms, 3))
        configs.append(atoms)
    return configs


@pytest.fixture
def synthetic_run(tmp_path):
    """A minimal remote_handler-style run dir with 2 iterations."""
    rng = np.random.default_rng(0)
    calchome = tmp_path / "calchome"
    eval_dir = calchome / "evaluator_output"
    eval_dir.mkdir(parents=True)

    progress = []
    counts = [3, 5]
    for it, n in enumerate(counts):
        rel = f"evaluator_output/evaluator_it{it}_0.extxyz"
        ase.io.write(calchome / rel, _labelled_configs(rng, n))
        progress.append(
            {
                "steps": [0, 0, 0],
                "sampler_output": [],
                "evaluator_output": [rel],
                "trainer_output": [],
            }
        )

    with open(calchome / "AL_progress.json", "w") as f:
        json.dump(progress, f)

    return calchome, counts


@pytest.mark.local
def test_run_layout_reads_progress(synthetic_run):
    calchome, counts = synthetic_run
    progress = load_progress(calchome)
    assert len(progress) == len(counts)

    evald = evaluated_configs(calchome)
    assert len(evald) == 2
    assert all(p.exists() for p in evald)


@pytest.mark.local
def test_build_dashboard_end_to_end(synthetic_run):
    calchome, counts = synthetic_run
    board = build_dashboard(calchome)

    out = calchome / "dashboard.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0

    # Panel caches were written under calchome/dashboard.
    assert (calchome / "dashboard" / "sampled_energy_volume.json").exists()
    assert (calchome / "dashboard" / "dataset_growth.json").exists()


@pytest.mark.local
def test_dataset_growth_counts(synthetic_run, tmp_path):
    """The project panel counts configs per iteration and cumulatively."""
    from alviz import Dashboard

    from jaxrens_al.panels import DatasetGrowthPanel

    calchome, counts = synthetic_run
    datapath = tmp_path / "dp"
    datapath.mkdir()

    panel = DatasetGrowthPanel(calchome=calchome, datapath=datapath)
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()

    assert panel.data["per_iteration"] == counts  # [3, 5]
    assert panel.data["cumulative"] == list(np.cumsum(counts))  # [3, 8]
