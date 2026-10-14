"""DashboardHook: fire-and-forget dashboard renders after AL stages."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("remote_handler")
from remote_handler.core.cluster_handler import ClusterConfig  # noqa: E402

import jaxrens_al.run.dashboard_hook as dashboard_hook  # noqa: E402
from jaxrens_al.run.dashboard_hook import (  # noqa: E402
    OUTPUT_SUBDIR,
    AutoDashboardSettings,
    DashboardHook,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def fake_cluster(monkeypatch):
    """``cluster: <name>`` resolves to this, no config dir needed."""
    cfg = ClusterConfig(
        remote_machine="lugus",
        partition="normal",
        conda_env="jaxrens_al",
        conda_source="/opt/conda/bin/activate",
        savepath="/tmp/remote_handler",
        template="dirona.sh.j2",
    )
    loaded = []

    def _load(spec):
        loaded.append(spec)
        return cfg

    monkeypatch.setattr(dashboard_hook, "_load_cluster", _load)
    cfg.loaded = loaded
    return cfg


@pytest.fixture
def calchome(tmp_path):
    home = (tmp_path / "calc").resolve()
    home.mkdir()
    return home


def _write_config(calchome, block, name="dashboard.json", **flat):
    data = dict(flat)
    if block is not None:
        data["auto_dashboard"] = block
    path = calchome / name
    path.write_text(json.dumps(data))
    return path


def _event(calchome, stage="sampler", iteration=1, is_last=False):
    return SimpleNamespace(
        stage=stage,
        iteration=iteration,
        calchome=Path(calchome),
        outputs=(),
        is_last=is_last,
    )


def _hook(submitted, **kw):
    return DashboardHook(
        submit=lambda p: submitted.append(p) or "Submitted", **kw
    )


def test_submits_one_render_from_the_config_block(calchome, fake_cluster):
    _write_config(calchome, {"cluster": "lugus"}, prefix="ns")
    submitted = []
    _hook(submitted)(_event(calchome, "sampler", 1))

    (script,) = submitted
    assert script == calchome / OUTPUT_SUBDIR / "dashboard_it1_sampler.sh"
    assert fake_cluster.loaded == ["lugus"]
    text = script.read_text()
    assert f"--job-name={DashboardHook.jobname(calchome)}" in text
    assert "#SBATCH --partition=normal" in text
    n_cores = AutoDashboardSettings().n_cores  # the block sets none
    assert f"#SBATCH --ntasks={n_cores}" in text
    assert "#SBATCH --dependency=singleton" in text
    assert "#SBATCH --output=dashboard_slurm_%j.out" in text
    out_pdf = calchome / OUTPUT_SUBDIR / "dashboard_it1_sampler.pdf"
    # The auto-discovered config is found by the render itself: no --config.
    assert (
        f"python -m jaxrens_al dashboard {calchome} --iterations 1 "
        f"-o {out_pdf}"
    ) in text
    # Not the last step: no whole-run summary, and no --force per page
    # (each stage writes its own PDF name, so there is nothing to overwrite).
    assert "--force" not in text
    assert "--config" not in text
    assert text.count("jaxrens_al dashboard") == 1


def test_block_settings_reach_the_job(calchome):
    _write_config(
        calchome,
        {
            "cluster": "lugus",
            "n-cores": 4,
            "memory": "32G",
            "time_limit": "02:00:00",
            "extra_args": "--sections",
        },
    )
    submitted = []
    _hook(submitted)(_event(calchome))
    text = submitted[0].read_text()
    assert "#SBATCH --ntasks=4" in text
    assert "#SBATCH --mem=32G" in text
    assert "#SBATCH --time=02:00:00" in text
    assert text.rstrip().endswith("--sections")


def test_last_step_also_renders_the_summary(calchome):
    _write_config(calchome, {"cluster": "lugus"})
    submitted = []
    _hook(submitted)(_event(calchome, "trainer", 3, is_last=True))
    text = submitted[0].read_text()
    summary = calchome / OUTPUT_SUBDIR / "dashboard.pdf"
    assert f"dashboard {calchome} -o {summary} --force" in text
    assert text.count("jaxrens_al dashboard") == 2


def test_no_config_or_no_block_submits_nothing(calchome):
    submitted = []
    hook = _hook(submitted)
    hook(_event(calchome))  # no dashboard config at all
    _write_config(calchome, None, prefix="ns")
    hook(_event(calchome))  # a config, but no auto_dashboard block
    assert submitted == []


def test_skips_other_stages_disabled_and_stub(calchome):
    submitted = []
    _write_config(calchome, {"cluster": "lugus", "after": ["trainer"]})
    _hook(submitted)(_event(calchome, "sampler"))
    _hook(submitted)(_event(calchome, "evaluator"))
    _write_config(calchome, {"cluster": "lugus", "enabled": False})
    _hook(submitted)(_event(calchome, "sampler"))
    _write_config(calchome, {"cluster": "lugus"})
    _hook(submitted, enabled=False)(_event(calchome, "sampler"))
    assert submitted == []


def test_config_is_reread_every_event(calchome):
    submitted = []
    hook = _hook(submitted)
    _write_config(calchome, {"cluster": "lugus", "enabled": False})
    hook(_event(calchome, "sampler", 0))
    _write_config(calchome, {"cluster": "lugus"})
    hook(_event(calchome, "trainer", 0))
    assert [p.name for p in submitted] == ["dashboard_it0_trainer.sh"]


def test_explicit_config_path_is_passed_to_the_render(calchome, tmp_path):
    cfg = tmp_path / "dash.yaml"
    cfg.write_text("auto_dashboard:\n  cluster: lugus\n")
    submitted = []
    _hook(submitted, config=cfg)(_event(calchome))
    assert f"--config {cfg.resolve()}" in submitted[0].read_text()


def test_bad_block_raises(calchome):
    hook = _hook([])
    _write_config(calchome, {"cluster": "lugus", "n_core": 4})
    with pytest.raises(ValueError, match="n_core"):
        hook(_event(calchome))
    _write_config(calchome, {"after": ["sampler"]})
    with pytest.raises(ValueError, match="needs 'cluster'"):
        hook(_event(calchome))


def test_dashboard_cli_ignores_the_block(calchome):
    """The render reads the same file; the block must not trip it."""
    import argparse
    import logging

    from jaxrens_al.cli_args import add_config_arg, apply_config

    _write_config(calchome, {"cluster": "lugus"}, prefix="ns")
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--prefix", default=None)
    add_config_arg(parser)
    argv = [str(calchome)]
    args = apply_config(
        parser, argv, parser.parse_args(argv), logging.getLogger()
    )
    assert args.prefix == "ns"


def test_jobname_is_stable_per_calchome(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert DashboardHook.jobname(a) == DashboardHook.jobname(a)
    assert DashboardHook.jobname(a) != DashboardHook.jobname(b)


def test_fires_from_the_staged_handler(calchome, tmp_path):
    """Wired as on_stage_complete: one submit per sampler/trainer step."""
    staged = pytest.importorskip(
        "remote_handler.workflows.staged_active_learning"
    )
    if not hasattr(staged, "StageCompleted"):
        pytest.skip("remote_handler without on_stage_complete hooks")

    class _Actor:
        def __init__(self, name, ext="xyz"):
            self.out = tmp_path / f"{name}_out.{ext}"
            self.out.write_text(name)

        def update(self, calcdir, actor_input, **kwargs):
            pass

        def get_output(self):
            return self.out

    _write_config(calchome, {"cluster": "lugus"})
    seed = tmp_path / "model.pkl"
    seed.write_text("model")
    submitted = []
    staged.StagedActiveLearningHandler(
        [
            staged.Stage("sampler", _Actor("sampler"), initial_input=seed),
            staged.Stage("evaluator", _Actor("evaluator")),
            staged.Stage(
                "trainer", _Actor("trainer", "pkl"), output_extension="pkl"
            ),
        ],
        calchome=calchome,
        num_iterations=2,
        exit_stage="sampler",
        on_stage_complete=[_hook(submitted)],
    ).run()

    assert [p.name for p in submitted] == [
        "dashboard_it0_sampler.sh",
        "dashboard_it0_trainer.sh",
        "dashboard_it1_sampler.sh",
    ]
    # Only the plan's last step (iteration 1's sampler) adds the summary.
    assert "--force" in submitted[-1].read_text()
    assert "--force" not in submitted[0].read_text()
