"""Shared test fixtures for the dashboard side.

The synthetic ``calchome`` builder lives here so every layer's tests draw from
*one* run-dir factory instead of each file growing its own. The factory and the
small artifact writers are exposed as fixtures returning callables, so a test
asks for exactly the inputs it needs (a single ``.npz``, a whole run, ...).

Only the dashboard-side deps are imported here (numpy + ase + matplotlib are
base dependencies, always present); nothing pulls in remote_handler / jaxrens.
"""

import json
from pathlib import Path

import ase.io
import matplotlib
import numpy as np
import pytest
from ase import Atoms

# Headless backend so the suite renders without a display.
matplotlib.use("Agg")

# Persistent, git-ignored scratch dir for human-inspectable renders.
_ARTIFACT_DIR = Path(__file__).parent / "data" / "temp"


@pytest.fixture(autouse=True)
def _close_figures():
    """Close figures opened by a test (each Dashboard creates one) to avoid
    leaking them across tests / matplotlib's >20-open warning."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# ---------------------------------------------------------------------------
# Low-level artifact writers (plain functions; exposed as fixtures below).
# ---------------------------------------------------------------------------
def _write_configs(path, n, info=None, *, with_forces=False, seed=0):
    """Write ``n`` extxyz frames; ``info(i)`` gives each frame's info dict."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        at = Atoms(
            "Cu2O",
            positions=rng.uniform(0, 3, (3, 3)),
            cell=[6, 6, 6],
            pbc=True,
        )
        if info is not None:
            at.info.update(info(i))
        if with_forces:
            at.arrays["REF_forces"] = rng.uniform(-1, 1, (3, 3))
        frames.append(at)
    ase.io.write(path, frames)
    return path


def _write_npz(path):
    """A NeuralIL ``save_data`` parity bundle (3-atom configs)."""
    rng = np.random.default_rng(1)
    n = 8
    e_obs = rng.uniform(-30, -20, n)
    np.savez(
        path,
        e_obs=e_obs,
        e_pred=e_obs + rng.normal(0, 0.1, n),
        f_obs=rng.uniform(-1, 1, (n, 3, 3)),
        f_pred=rng.uniform(-1, 1, (n, 3, 3)),
        types=np.array([[0, 0, 1]] * n),  # 3 atoms each
    )
    return path


def _write_error_log(path):
    """A NeuralIL ``error_*.txt`` log: two epochs, two ensemble members."""
    path.write_text(
        "i_epoch=0\n"
        "Model #1:\n"
        "F_RMSE = 0.50 eV/A. \n"
        "E_RMSE = 0.20 eV. \n"
        "Model #2:\n"
        "F_RMSE = 0.60 eV/A. \n"
        "E_RMSE = 0.30 eV. \n"
        "i_epoch=10\n"
        "Model #1:\n"
        "F_RMSE = 0.10 eV/A. \n"
        "E_RMSE = 0.05 eV. \n"
    )
    return path


def _write_resampling_artifacts(results_dir, n_total=6, n_selected=2):
    """Write a quests_pca.json + resampled_provenance.json pair (synthetic)."""
    rng = np.random.default_rng(2)
    results_dir.mkdir(parents=True, exist_ok=True)
    structures = []
    for i in range(n_total):
        structures.append(
            {
                "index": i,
                "pc1": float(rng.normal()),
                "pc2": float(rng.normal()),
                "selected": i < n_selected,
                "n_atoms": 3,
                "formula": "Cu2O",
            }
        )
    (results_dir / "quests_pca.json").write_text(
        json.dumps(
            {
                "mode": "diversity",
                "method": "msc",
                "n_total": n_total,
                "n_selected": n_selected,
                "explained_variance_ratio": [0.6, 0.25],
                "structures": structures,
            }
        )
    )
    selected = [
        {"index": i, "run_id": i % 2, "iter": 10 * i}
        for i in range(n_selected)
    ]
    (results_dir / "resampled_provenance.json").write_text(
        json.dumps(
            {
                "n_candidates": n_total,
                "n_selected": n_selected,
                "selected": selected,
            }
        )
    )


def _make_calchome(
    tmp_path,
    *,
    n_iter=2,
    n_evaluated=3,
    n_sampled=5,
    with_forces=True,
    with_training=False,
    with_resampling=False,
    with_ns=False,
):
    """Build a remote_handler-style run dir under ``tmp_path``.

    ``n_evaluated`` / ``n_sampled`` are an int (same every iteration) or a
    per-iteration list; ``0`` skips that actor's output for an iteration (so a
    panel simply gets no input). Evaluated (DFT) configs carry ``REF_forces``
    by default (the evaluated-energy/volume panel reads them), like real
    evaluator output; sampled configs carry ``ns_energy`` + ``run_id``.
    ``with_training`` adds NeuralIL parity/error artifacts under
    ``<i>/training/results``. ``with_resampling`` adds QUESTS
    ``quests_pca.json`` + ``resampled_provenance.json`` under
    ``<i>/ns_sampler/results``. ``with_ns`` drops a stub ``ns.energies`` there
    so the thermodynamics panel is discovered (its compute still needs jaxrens
    -- patch it with the ``fake_ns_monitor`` fixture to render).
    """

    def _per_iter(x):
        return list(x) if isinstance(x, (list, tuple)) else [x] * n_iter

    ev_counts = _per_iter(n_evaluated)
    sa_counts = _per_iter(n_sampled)

    calchome = tmp_path / "calchome"
    (calchome / "evaluator_output").mkdir(parents=True)
    (calchome / "sampler_output").mkdir(parents=True)

    progress = []
    for it in range(n_iter):
        entry = {
            "steps": [0, 0, 0],
            "sampler_output": [],
            "evaluator_output": [],
        }

        if ev_counts[it]:
            e = f"evaluator_output/e{it}.extxyz"
            _write_configs(
                calchome / e,
                ev_counts[it],
                lambda i, it=it: {"REF_energy": -9.0 - it + 0.1 * i},
                with_forces=with_forces,
            )
            entry["evaluator_output"] = [e]

        if sa_counts[it]:
            s = f"sampler_output/s{it}.extxyz"
            _write_configs(
                calchome / s,
                sa_counts[it],
                lambda i, it=it: {
                    "ns_energy": -10.0 - it + 0.1 * i,
                    "run_id": i % 2,
                },
            )
            entry["sampler_output"] = [s]

        if with_training:
            tdir = calchome / str(it) / "training" / "results"
            tdir.mkdir(parents=True)
            _write_npz(tdir / "val.npz")
            _write_error_log(tdir / "error_0.txt")

        if with_resampling:
            _write_resampling_artifacts(
                calchome / str(it) / "ns_sampler" / "results"
            )

        if with_ns:
            nsres = calchome / str(it) / "ns_sampler" / "results"
            nsres.mkdir(parents=True, exist_ok=True)
            (nsres / "ns.energies").write_text("# stub NS energies\n")

        progress.append(entry)

    (calchome / "AL_progress.json").write_text(json.dumps(progress))
    return calchome


# ---------------------------------------------------------------------------
# Fixtures: hand the factories to tests as callables.
# ---------------------------------------------------------------------------
@pytest.fixture
def write_configs():
    return _write_configs


@pytest.fixture
def write_npz():
    return _write_npz


@pytest.fixture
def write_error_log():
    return _write_error_log


@pytest.fixture
def make_calchome():
    return _make_calchome


@pytest.fixture
def fake_ns_monitor(monkeypatch):
    """Patch the thermodynamics panel's jaxrens seam with a fake monitor.

    The NS thermodynamics panel computes via ``jaxrens.postprocess.Monitor``;
    patching its ``_load_monitor`` seam lets a full-dashboard render include it
    without jaxrens or a real ``.energies``/``.checkpoint``.
    """
    from jaxrens_al.panels.thermodynamics import NSThermodynamicsHeatmap

    class _FakeMonitor:
        def __init__(self, *args):
            self.dead_volumes = np.ones(7)

        def heat_capacity(self, T, k_B=1.0):
            return np.abs(np.asarray(T, dtype=float)) + 1.0

        def expectation(self, obs, T, k_B=1.0):
            return np.asarray(T, dtype=float) * 0 + float(np.mean(obs))

    monkeypatch.setattr(
        NSThermodynamicsHeatmap,
        "_load_monitor",
        lambda self, result_dir: _FakeMonitor(),
    )


@pytest.fixture
def artifact_dir():
    """Git-ignored dir (``test/data/temp``) for human-inspectable renders.

    Unit tests assert on *data*, but a dashboard's layout (grid balance,
    titles, overlaps) is invariant to the synthetic data and can only be
    checked by eye -- so layout tests drop a PDF here to open. The run dir
    stays in ``tmp_path`` (hermetic); only the final PDF lands here, named
    per test so reruns overwrite rather than accumulate.
    """
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return _ARTIFACT_DIR
