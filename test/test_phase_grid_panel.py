"""Tests for the phase-grid dashboard panel and its run-side resolver."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("alviz")

from ase import Atoms  # noqa: E402

from jaxrens_al.dashboard.panels.phase_grid import (  # noqa: E402
    PhaseGridNeed,
    PhaseGridPanel,
    PhaseGridSpec,
    _build_phase_grid,
    hm_tex,
    phase_labels,
)

# (T, P, space group, symbol, V, converged)
_GRID = [
    (300.0, 0.0, 216, "F-43m", 19.7, True),
    (300.0, 20.0, 225, "Fm-3m", 14.1, True),
    (1200.0, 0.0, 186, "P6_3mc", 19.7, True),
    (1200.0, 20.0, 225, "Fm-3m", 14.1, True),
    (2500.0, 0.0, 1, "P1", 20.4, False),
    (2500.0, 20.0, 7, "Pc", 16.0, True),
]


def _write_labelled(path):
    import ase.io

    frames = []
    for k, (T, P, sg, sym, V, ok) in enumerate(_GRID):
        at = Atoms(
            "ZnS", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True
        )
        at.info.update(
            grid_T=T,
            grid_P=P,
            replica=int(P // 10),
            iter=100 * k,
            grid_weight_share=0.01,
            relax_volume=V,
            relax_enthalpy=-9.0,
            is_converged=ok,
            spacegroup=sg,
            spacegroup_symbol=sym,
        )
        frames.append(at)
    ase.io.write(path, frames)
    return path


def _panel(tmp_path, **kw):
    return PhaseGridPanel(
        tmp_path / "output",
        model_file="m.model",
        temperatures=[300, 1200, 2500],
        config="ns_config.yaml",
        prefix="ns",
        datapath=tmp_path,
        **kw,
    )


def test_hm_tex():
    assert hm_tex("F-43m") == r"F$\bar{4}$3m"
    assert hm_tex("P6_3mc") == "P6$_{3}$mc"
    assert hm_tex("Fm-3m") == r"Fm$\bar{3}$m"


def test_phase_labels_groups_rare_and_low_symmetry():
    nums = [216, 216, 216, 225, 225, 186, 1, 2, 7]
    syms = ["F-43m"] * 3 + ["Fm-3m"] * 2 + ["P6_3mc", "P1", "P-1", "Pc"]
    labels, order = phase_labels(nums, syms, max_phases=2)
    assert order[0].startswith(r"F$\bar{4}$3m") and order[1].startswith(
        r"Fm$\bar{3}$m"
    )
    assert order[-1] == "other / low symmetry" and len(order) == 3
    # P1 / P-1 are never a phase of their own; P6_3mc and Pc exceed max_phases
    assert labels.count("other / low symmetry") == 4


def test_need_is_hashable_and_canonical(tmp_path):
    a = _panel(tmp_path).phase_grid_needs()[0]
    b = _panel(tmp_path, stages=["positions", "cell"]).phase_grid_needs()[0]
    assert isinstance(a, PhaseGridNeed)
    assert a == b and hash(a) == hash(b)
    assert a.temperatures == (300.0, 1200.0, 2500.0) and a.stages == (
        "positions",
        "cell",
    )
    assert len({a, b}) == 1


def test_panel_errors_without_delivery(tmp_path):
    with pytest.raises(RuntimeError, match="resolve_phase_grid"):
        _panel(tmp_path).compute_data({})


def test_panel_reads_and_renders(tmp_path):
    import matplotlib.pyplot as plt

    panel = _panel(tmp_path, swap_axes=True, max_phases=3)
    (need,) = panel.phase_grid_needs()
    panel.receive_phase_grid(
        {need: _write_labelled(tmp_path / "labelled.extxyz")}
    )
    data = panel.compute_data({})
    assert len(data["rows"]) == len(_GRID)
    assert (
        data["rows"][1]["symbol"] == "Fm-3m" and data["rows"][1]["sg"] == 225
    )
    assert data["rows"][4]["converged"] is False
    assert data["model"] == "m.model"

    panel.data = data
    fig = plt.figure(figsize=(8, 4))
    panel.subspec = fig.add_gridspec(1, 1)[0, 0]
    panel.allocate_panels(fig)
    panel.plot_data()
    assert panel.ax.get_xlabel() == "$P$"  # swap_axes -> pressure on x
    legend = [t.get_text() for t in panel.ax.get_legend().get_texts()]
    assert "other / low symmetry" in legend and "not converged" in legend
    plt.close(fig)


# --- builder ---------------------------------------------------------------
def _calchome(tmp_path):
    """Minimal AL layout: iteration 1's NS output + iteration 0's torch model."""
    calchome = tmp_path / "calchome"
    out = calchome / "1" / "ns_sampler" / "results" / "output"
    out.mkdir(parents=True)
    for r in range(4):
        (out / f"ns.run{r:02d}.traj.extxyz").write_text("")
    (out.parent / "ns_config.yaml").write_text("ensemble: {}\n")
    models = calchome / "0" / "train" / "results" / "calcdirs" / "2" / "models"
    models.mkdir(parents=True)
    (models / "Model.model").write_text("torch")
    bundle = calchome / "trainer_output" / "trainer_it0_2.bundle"
    bundle.mkdir(parents=True)
    return calchome, out, bundle, models / "Model.model"


def _ctx(calchome, out, tmp_path, **kw):
    from jaxrens_al.dashboard import RunContext

    return RunContext(
        calchome=calchome,
        datapath=tmp_path,
        ns_dirs=[out],
        ns_prefix="ns",
        iteration=1,
        **kw,
    )


def test_builder_maps_sampling_bundle_to_torch_model(tmp_path):
    calchome, out, bundle, torch_model = _calchome(tmp_path)
    ctx = _ctx(calchome, out, tmp_path, sampling_model_file=bundle)
    spec = PhaseGridSpec(
        temperatures=[100.0, 3000.0],
        n_grid_T=3,
        replica_stride=2,
        k_B=8.617333262e-5,
    )
    panel = _build_phase_grid(spec, ctx, ctx.common)
    (need,) = panel.phase_grid_needs()
    assert Path(need.model) == torch_model
    assert Path(need.config) == out.parent / "ns_config.yaml"
    assert need.replicas == (0, 2)
    np.testing.assert_allclose(need.temperatures, [100.0, 1550.0, 3000.0])


def test_builder_explicit_model_and_grid(tmp_path):
    calchome, out, _, _ = _calchome(tmp_path)
    ctx = _ctx(calchome, out, tmp_path)
    spec = PhaseGridSpec(
        model="/x/foundation.model", grid_temperatures=[150, 500]
    )
    need = _build_phase_grid(spec, ctx, ctx.common).phase_grid_needs()[0]
    assert need.model == "/x/foundation.model"
    assert need.temperatures == (150.0, 500.0) and need.replicas is None


@pytest.mark.parametrize(
    "ctx_kw, spec_kw",
    [
        (
            {},
            {"grid_temperatures": [300]},
        ),  # iteration 0: no torch model known
        (
            {"iteration": None},
            {"model": "m.model", "grid_temperatures": [300]},
        ),  # summary
        ({}, {"model": "m.model"}),  # no temperatures anywhere
    ],
)
def test_builder_drops_out(tmp_path, ctx_kw, spec_kw):
    calchome, out, _, _ = _calchome(tmp_path)
    ctx = _ctx(calchome, out, tmp_path)
    for k, v in ctx_kw.items():
        setattr(ctx, k, v)
    assert _build_phase_grid(PhaseGridSpec(**spec_kw), ctx, ctx.common) is None


# --- resolver ----------------------------------------------------------------
def test_resolve_phase_grid_runs_once_and_reuses(tmp_path, monkeypatch):
    pytest.importorskip("remote_handler.processing")
    import ase.io

    import jaxrens_al.run.phase_grid as pg
    from jaxrens_al.run.processing_resolver import resolve_phase_grid

    out = tmp_path / "output"
    out.mkdir()
    (out / "ns.run00.energies").write_text("")
    calls = {"select": 0, "jobs": []}

    def fake_select(result_dir, temperatures, **kw):
        calls["select"] += 1
        out = []
        for k, T in enumerate(temperatures):
            at = Atoms("Cu", cell=[3, 3, 3], pbc=True)
            # the two hottest cells pick the same dead point
            at.info.update(
                grid_T=float(T),
                grid_P=0.0,
                replica=0,
                iter=min(k, 1),
                grid_weight_share=0.1 * (k + 1),
            )
            out.append(at)
        return out

    class FakeJob:
        def __init__(self, picks, n, *, tag, **kw):
            self.picks, self.tag, self.kw = picks, tag, kw
            calls["jobs"].append(self)

        def allocate_data(self):
            pass

        def submit(self, only_start=False):
            pass

        class executor:
            @staticmethod
            def wait_until_finished():
                pass

        def collect_and_harvest(self):
            from remote_handler.core.atoms_shard import front_tag

            frames = ase.io.read(self.picks, index=":")
            calls["relaxed"] = len(frames)
            for at in frames:
                at.info.update(spacegroup=225, spacegroup_symbol="Fm-3m")
            ase.io.write(front_tag(str(self.picks), self.tag), frames)

    monkeypatch.setattr(pg, "select_grid_samples", fake_select)
    monkeypatch.setattr(pg, "phase_grid_job", FakeJob)

    panels = [
        PhaseGridPanel(
            out,
            model_file="m.model",
            temperatures=[300, 600, 900],
            datapath=tmp_path,
        )
        for _ in range(2)  # identical needs -> one job
    ]
    ctx = dict(
        cluster_config=None,
        remote_fs=None,
        remote_savepath="/remote",
        local_savepath=tmp_path / "pg",
    )
    delivered = resolve_phase_grid(panels, **ctx)
    assert calls["select"] == 1 and len(calls["jobs"]) == 1
    job = calls["jobs"][0]
    assert job.kw["remote_savepath"] == f"/remote/{job.tag}"
    (labelled,) = delivered.values()
    assert labelled.exists() and all(p._labelled == labelled for p in panels)
    assert calls["relaxed"] == 2  # iters {0, 1}: the duplicate relaxed once
    grid = ase.io.read(labelled, index=":")
    assert [a.info["grid_T"] for a in grid] == [300.0, 600.0, 900.0]
    assert [a.info["iter"] for a in grid] == [0, 1, 1]
    assert grid[2].info["grid_weight_share"] == pytest.approx(0.3)
    assert all(a.info["spacegroup"] == 225 for a in grid)
    assert len(panels[0].compute_data({})["rows"]) == 3

    resolve_phase_grid(panels, **ctx)  # labelled newer than the NS files
    assert calls["select"] == 1 and len(calls["jobs"]) == 1


def test_builder_initial_model_only_for_unrecorded_seed(tmp_path):
    calchome, out, bundle, torch_model = _calchome(tmp_path)
    spec = PhaseGridSpec(
        initial_model="/x/foundation.model", grid_temperatures=[300]
    )
    # iteration 0: no sampling model recorded -> the seed model
    ctx0 = _ctx(calchome, out, tmp_path)
    need = _build_phase_grid(spec, ctx0, ctx0.common).phase_grid_needs()[0]
    assert need.model == "/x/foundation.model"
    # later iterations keep their own sampling model
    ctx1 = _ctx(calchome, out, tmp_path, sampling_model_file=bundle)
    need = _build_phase_grid(spec, ctx1, ctx1.common).phase_grid_needs()[0]
    assert Path(need.model) == torch_model
