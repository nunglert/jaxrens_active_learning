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


def _write_mace_train_log(path):
    """A MACE ``*_train.txt`` JSON-lines log: a baseline + two eval epochs.

    Interleaves ``mode == "opt"`` rows (which the panel must skip) and the
    ``epoch: null`` baseline eval (also skipped) with two real eval epochs.
    """
    path.write_text(
        '{"loss": 9.0, "rmse_e_per_atom": 20.0, "rmse_f": 3.0, '
        '"mode": "eval", "epoch": null, "head": "Default"}\n'
        '{"loss": 5.0, "mode": "opt", "epoch": 0}\n'
        '{"loss": 4.0, "rmse_e_per_atom": 12.0, "rmse_f": 2.0, '
        '"mode": "eval", "epoch": 0, "head": "Default"}\n'
        '{"loss": 2.0, "mode": "opt", "epoch": 1}\n'
        '{"loss": 1.0, "rmse_e_per_atom": 6.0, "rmse_f": 1.0, '
        '"mode": "eval", "epoch": 1, "head": "Default"}\n'
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
    with_mace_training=False,
    eval_with_surrogate=False,
    with_resampling=False,
    with_ns=False,
    with_acc_rates=False,
    with_adaptation=False,
    with_re_stats=False,
    with_model=False,
    staged=False,
):
    """Build a remote_handler-style run dir under ``tmp_path``.

    ``n_evaluated`` / ``n_sampled`` are an int (same every iteration) or a
    per-iteration list; ``0`` skips that actor's output for an iteration (so a
    panel simply gets no input). Evaluated (DFT) configs carry ``REF_forces``
    by default (the evaluated-energy/volume panel reads them), like real
    evaluator output; sampled configs carry ``ns_energy`` + ``run_id``.
    ``eval_with_surrogate`` adds an ``ns_energy`` (sampler surrogate) key to
    the evaluated configs, so the surrogate-vs-DFT parity/error panels appear.
    ``with_mace_training`` drops a MACE ``*_train.txt`` JSON-lines log under
    ``<i>/training/results`` so the ``mace_training_curve`` panel is found.
    ``with_training`` adds NeuralIL parity/error artifacts under
    ``<i>/training/results``. ``with_resampling`` adds QUESTS
    ``quests_pca.json`` + ``resampled_provenance.json`` under
    ``<i>/ns_sampler/results``. ``with_ns`` drops a stub ``ns.energies`` there
    so the thermodynamics panel is discovered (its compute still needs jaxrens
    -- patch it with the ``fake_ns_monitor`` fixture to render).
    ``with_acc_rates`` drops a stub ``ns.acc_rates.h5`` there so the move-
    acceptance panel is discovered (compute needs jaxrens -- patch it with the
    ``fake_acc_log`` fixture to render). ``with_adaptation`` drops a stub
    ``ns.adaptation.h5`` for the step-size / adaptation-acceptance panels
    (render with ``fake_adaptation_log``). ``with_re_stats`` drops a stub
    ``ns.re_stats.h5`` for the RE-swap panel (render with ``fake_re_log``).
    ``with_model`` drops a stub
    ``neuralil_model.pkl`` under ``<i>/training/results`` so the EOS panel
    finds a model (compute needs jaxrens -- patch ``fake_eos_eval`` to render).
    ``staged`` serializes ``AL_progress.json`` in the ``StagedActiveLearning
    Handler`` schema (``{outputs: {stage: {iter: [rel]}}}``) instead of the
    classic per-iteration list; the on-disk artifacts are identical either way.
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

            def _ev_info(i, it=it):
                info = {"REF_energy": -9.0 - it + 0.1 * i}
                if eval_with_surrogate:
                    # Surrogate (sampler) energy preserved through the
                    # evaluator, so a surrogate-vs-DFT parity is available.
                    info["ns_energy"] = -9.0 - it + 0.1 * i + 0.05
                return info

            _write_configs(
                calchome / e,
                ev_counts[it],
                _ev_info,
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

        if with_mace_training:
            tdir = calchome / str(it) / "training" / "results"
            tdir.mkdir(parents=True, exist_ok=True)
            _write_mace_train_log(tdir / "mace_run-0_train.txt")

        if with_resampling:
            _write_resampling_artifacts(
                calchome / str(it) / "ns_sampler" / "results"
            )

        if with_ns:
            nsres = calchome / str(it) / "ns_sampler" / "results"
            nsres.mkdir(parents=True, exist_ok=True)
            (nsres / "ns.energies").write_text("# stub NS energies\n")

        if with_acc_rates:
            nsres = calchome / str(it) / "ns_sampler" / "results"
            nsres.mkdir(parents=True, exist_ok=True)
            (nsres / "ns.acc_rates.h5").write_bytes(b"")  # stub for discovery

        if with_adaptation:
            nsres = calchome / str(it) / "ns_sampler" / "results"
            nsres.mkdir(parents=True, exist_ok=True)
            (nsres / "ns.adaptation.h5").write_bytes(b"")  # stub: discovery

        if with_re_stats:
            nsres = calchome / str(it) / "ns_sampler" / "results"
            nsres.mkdir(parents=True, exist_ok=True)
            (nsres / "ns.re_stats.h5").write_bytes(b"")  # stub for discovery

        if with_model:
            tdir = calchome / str(it) / "training" / "results"
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / "neuralil_model.pkl").write_bytes(b"")  # stub: discovery

        progress.append(entry)

    if staged:
        progress = _classic_to_staged(progress)
    (calchome / "AL_progress.json").write_text(json.dumps(progress))
    return calchome


def _classic_to_staged(progress):
    """Rewrite a classic per-iteration list as the staged ``outputs`` schema.

    The inverse of ``run_layout._normalize_staged`` -- used to exercise the
    dashboard/CLI against a ``StagedActiveLearningHandler``-style file.
    """
    stage_key = {
        "sampler": "sampler_output",
        "selector": "selector_output",
        "evaluator": "evaluator_output",
        "trainer": "trainer_output",
    }
    outputs = {}
    for i, entry in enumerate(progress):
        for stage, key in stage_key.items():
            rels = entry.get(key, []) or []
            if rels:
                outputs.setdefault(stage, {})[str(i)] = rels
    return {
        "stages": list(stage_key),
        "entry_index": 0,
        "exit_index": len(stage_key) - 1,
        "num_iterations": len(progress),
        "outputs": outputs,
    }


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
def write_mace_train_log():
    return _write_mace_train_log


@pytest.fixture
def make_calchome():
    return _make_calchome


@pytest.fixture
def eos_reference():
    """Factory: write an EOS volume scan (one phase) with REF_energy.

    Energies follow a Birch-Murnaghan curve so the panel's fit converges; the
    9 lattice constants give enough points for the 4-parameter fit.
    """

    def _make(path):
        from jaxrens_al.panels.eos import _birch_murnaghan

        frames = []
        for a in np.linspace(5.4, 6.6, 9):
            at = Atoms(
                "Cu2O",
                positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                cell=[a, a, a],
                pbc=True,
            )
            v = at.get_volume() / len(at)  # per atom
            # BM(E0=-4, V0=24, B0=0.5 eV/A^3, B0'=4), total energy.
            at.info["REF_energy"] = float(
                _birch_murnaghan(v, -4.0, 24.0, 0.5, 4.0) * len(at)
            )
            frames.append(at)
        ase.io.write(path, frames)
        return path

    return _make


@pytest.fixture
def fake_ns_monitor(monkeypatch):
    """Patch the thermodynamics panel's jaxrens seam with fake monitors.

    The NS thermodynamics panel computes via
    ``MonitorCollection.from_multi_run_directory``; patching its
    ``_load_monitors`` seam (one Monitor per replica) lets a full-dashboard
    render include it without jaxrens or a real ``.energies``/``.checkpoint``.
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
        "_load_monitors",
        lambda self, result_dir: [_FakeMonitor()],
    )


@pytest.fixture
def fake_ns_traces(monkeypatch):
    """Patch the energy-trace panel's jaxrens seam with fake replicas.

    Two fake Monitors carrying ``iteration_trace`` / ``energy_trace`` so the
    per-replica energy-vs-iteration panel renders without jaxrens.
    """
    from jaxrens_al.panels.traces import NSEnergyTracePanel

    class _FakeTraceMonitor:
        def __init__(self, label):
            self.label = label
            self.iteration_trace = np.arange(20)
            self.energy_trace = -np.arange(20, dtype=float)

    monkeypatch.setattr(
        NSEnergyTracePanel,
        "_load_monitors",
        lambda self, d: [_FakeTraceMonitor("run0"), _FakeTraceMonitor("run1")],
    )


@pytest.fixture
def fake_adaptation_log(monkeypatch):
    """Patch the adaptation panels' jaxrens reader with a fake log.

    Lets the step-size and adaptation-acceptance panels compute/render from a
    stub ``.adaptation.h5`` without h5py. Shapes match the real
    ``AdaptationLog``: ``(n_records, n_runs, n_moves)``.
    """
    from jaxrens_al.panels.adaptation import _AdaptationTracePanel

    class _FakeAdaptLog:
        iterations = np.array([0, 10, 20])
        move_names = ["gmc", "swap"]
        n_runs = 2
        n_moves = 2
        step_sizes = np.full((3, 2, 2), 0.1)
        acceptance_rates = np.full((3, 2, 2), 0.5)

    monkeypatch.setattr(
        _AdaptationTracePanel,
        "_load_log",
        lambda self, path: _FakeAdaptLog(),
    )


@pytest.fixture
def fake_acc_log(monkeypatch):
    """Patch the acceptance panel's jaxrens reader with a fake log.

    Lets the move-acceptance panel compute/render from a stub
    ``.acc_rates.h5`` without jaxrens or h5py.
    """
    from jaxrens_al.panels.acceptance import MoveAcceptancePanel

    class _FakeAccLog:
        iterations = np.array([0, 10, 20])
        move_names = ["gmc", "swap"]

        @property
        def acceptance_rates(self):
            # (n_entries, n_runs, n_moves)
            return np.array(
                [
                    [[0.8, 0.5]],
                    [[0.7, 0.45]],
                    [[0.6, 0.4]],
                ],
                dtype=float,
            )

    monkeypatch.setattr(
        MoveAcceptancePanel, "_load_log", lambda self, path: _FakeAccLog()
    )


@pytest.fixture
def fake_re_log(monkeypatch):
    """Patch the RE-swap panel's jaxrens reader with a fake RE log."""
    from jaxrens_al.panels.acceptance import RESwapAcceptancePanel

    class _FakeRELog:
        iterations = np.array([0, 100, 200])
        n_pairs = 2
        flavor = "pressure"

        @property
        def acceptance_rates(self):
            # (n_entries, n_pairs)
            return np.array(
                [[0.4, 0.3], [0.45, 0.35], [0.5, 0.38]], dtype=float
            )

    monkeypatch.setattr(
        RESwapAcceptancePanel, "_load_log", lambda self, path: _FakeRELog()
    )


@pytest.fixture
def fake_quests_descriptors(monkeypatch):
    """Patch the atomic-envs panel's descriptor seam (no ``quests`` needed).

    Returns one fake per-atom descriptor row per atom, so PCA + plotting run
    without the ``quests`` package installed.
    """
    from jaxrens_al.panels.environments import AtomicEnvsPanel

    rng = np.random.default_rng(3)

    def _fake(self, atoms_list):
        n_atoms = sum(len(at) for at in atoms_list)
        return rng.normal(size=(n_atoms, 8))

    monkeypatch.setattr(AtomicEnvsPanel, "_descriptors", _fake)


@pytest.fixture
def fake_eos_eval(monkeypatch):
    """Patch the EOS panel's model-evaluation seam (no jaxrens / model needed).

    Returns a smooth synthetic energy per structure (quadratic in volume), so
    the panel computes/renders without loading a real model.
    """
    from jaxrens_al.panels.eos import EquationOfStatePanel, _birch_murnaghan

    def _fake(self, model_file, atoms_list):
        # BM curve (slightly off the DFT one) so the model fit also converges.
        return [
            float(
                _birch_murnaghan(
                    at.get_volume() / len(at), -3.9, 23.5, 0.55, 4.0
                )
                * len(at)
            )
            for at in atoms_list
        ]

    monkeypatch.setattr(EquationOfStatePanel, "_evaluate", _fake)


@pytest.fixture
def fake_model_parity_eval(monkeypatch):
    """Patch the model-parity panel's re-scoring seam (no jaxrens / model).

    Returns predicted (energies, forces) per config -- the reference values
    slightly perturbed -- so the panel computes/renders a non-trivial parity
    without loading a real model.
    """
    from jaxrens_al.panels.training import ModelParityPanel

    def _fake(self, model_file, atoms_list):
        energies = [
            float(at.info.get("REF_energy", 0.0)) + 0.1 for at in atoms_list
        ]
        forces = None
        if self._show_forces:
            forces = [
                (
                    np.asarray(at.arrays["REF_forces"], dtype=float) + 0.01
                    if "REF_forces" in at.arrays
                    else np.zeros((len(at), 3))
                )
                for at in atoms_list
            ]
        return energies, forces

    monkeypatch.setattr(ModelParityPanel, "_evaluate", _fake)


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
