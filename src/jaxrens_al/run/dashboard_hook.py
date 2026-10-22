"""Submit a dashboard render after AL loop stages, fire-and-forget.

:class:`DashboardHook` is an ``on_stage_complete`` hook for
``remote_handler``'s ``StagedActiveLearningHandler``. After each chosen
stage it ``sbatch``-es one job that runs ``jaxrens-al dashboard`` on the
loop's calchome for that iteration, and returns at once: the loop never
waits for (or collects) the render.

It is configured by the **dashboard config** -- the same
``<calchome>/dashboard.yaml`` the render reads (or an explicit ``config=``
path) -- through an ``auto_dashboard:`` block, which every dashboard CLI
skips::

    prefix: neuralil_si16          # the render's own settings, as usual
    cluster: datalab.json          # (where its annotation/model-eval runs)
    auto_dashboard:
      cluster: lugus               # ClusterConfig the render job runs on
      after: [sampler, trainer]    # stages that trigger a render
      summary_on_last: true        # + whole-run dashboard.pdf at the end
      n_cores: 8
      memory: 32G                  # optional; time_limit too
      extra_args: [--sections]     # further jaxrens-al dashboard flags
      enabled: true

The config is re-read at every event, so editing it (or setting
``enabled: false``) takes effect mid-run. No config, or no
``auto_dashboard:`` block, means no render: wiring the hook into a run
script is harmless until the block is added.

Each submission writes ``<calchome>/auto_dashboard/dashboard_it<i>_
<stage>.pdf``, so a later stage of the same iteration renders anew while
every panel cache from the earlier render is reused; the whole-run summary
overwrites its PDF with ``--force`` (which never recomputes caches). All jobs of one calchome share a job name and
``--dependency=singleton``, so they run one after another and never write
the same caches concurrently. The job is submitted with a local ``sbatch``:
the loop must run where that cluster's scheduler is and share the
calchome's filesystem (lugus).
"""

from __future__ import annotations

import hashlib
import logging
import shlex
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Optional

from jaxrens_al.cli_args import AUTO_DASHBOARD_KEY, find_config, load_config

logger = logging.getLogger(__name__)

OUTPUT_SUBDIR = "auto_dashboard"
# body_single.sh.j2 renders "python <python_script_path> <input_file>".
_DASHBOARD_MODULE = "-m jaxrens_al dashboard"


@dataclass
class AutoDashboardSettings:
    """The ``auto_dashboard:`` block of a dashboard config."""

    cluster: Optional[str] = None
    after: list = field(default_factory=lambda: ["sampler", "trainer"])
    summary_on_last: bool = True
    n_cores: int = 2
    memory: Optional[str] = None
    time_limit: Optional[str] = None
    extra_args: list = field(default_factory=list)
    enabled: bool = True

    @classmethod
    def from_block(cls, block, path) -> "AutoDashboardSettings":
        if not isinstance(block, dict):
            raise ValueError(
                f"{path}: '{AUTO_DASHBOARD_KEY}' must be a mapping, got "
                f"{type(block).__name__}."
            )
        block = {str(k).replace("-", "_"): v for k, v in block.items()}
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(block) - valid)
        if unknown:
            raise ValueError(
                f"{path}: unknown '{AUTO_DASHBOARD_KEY}' key(s) {unknown}; "
                f"valid: {sorted(valid)}."
            )
        settings = cls(**block)
        for name in ("after", "extra_args"):
            value = getattr(settings, name)
            if isinstance(value, str):
                setattr(settings, name, [value])
        return settings


def _sbatch(script: Path) -> str:
    """Submit ``script`` from its own directory; return sbatch's stdout."""
    result = subprocess.run(
        ["sbatch", script.name],
        cwd=script.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _load_cluster(spec):
    """A ``ClusterConfig`` from a registered name or a JSON path."""
    from remote_handler.core.cluster_handler import ClusterConfig

    if Path(spec).is_absolute():
        return ClusterConfig.from_file(str(spec))
    return ClusterConfig.from_name(str(spec))


class DashboardHook:
    """``on_stage_complete`` hook that submits a dashboard render.

    Args:
        config: Dashboard config to read (and to hand the render via
            ``--config``). Default: the ``dashboard.*`` auto-discovered in
            the loop's calchome, which the render then finds by itself.
        enabled: ``False`` makes the hook a no-op whatever the config says
            (pass ``not STUB``).
        submit: Callable taking the written job script's path and
            submitting it; defaults to a local ``sbatch``. A test seam.
    """

    def __init__(
        self,
        config: Optional[Path] = None,
        enabled: bool = True,
        submit: Optional[Callable[[Path], str]] = None,
    ):
        self.config = Path(config).resolve() if config else None
        self.enabled = enabled
        self.submit = submit or _sbatch

    def __repr__(self) -> str:
        return f"DashboardHook(config={self.config})"

    @staticmethod
    def jobname(calchome: Path) -> str:
        """One name per calchome -- the ``singleton`` serialisation key."""
        digest = hashlib.sha1(str(Path(calchome).resolve()).encode())
        return f"dash_{digest.hexdigest()[:8]}"

    def settings(self, calchome: Path):
        """``(config path, settings)``; settings ``None`` if no block."""
        path = self.config or find_config(calchome)
        if path is None:
            return None, None
        block = load_config(path).get(AUTO_DASHBOARD_KEY)
        if block is None:
            return path, None
        return path, AutoDashboardSettings.from_block(block, path)

    def dashboard_args(
        self,
        calchome: Path,
        iteration: int,
        stage: str,
        is_last: bool,
        settings: AutoDashboardSettings,
    ) -> list[str]:
        """``jaxrens-al dashboard`` argument string(s), one per render."""
        out_dir = Path(calchome) / OUTPUT_SUBDIR
        base = [str(calchome)]
        if self.config is not None:
            base += ["--config", str(self.config)]
        renders = [
            base
            + [
                "--iterations",
                str(iteration),
                "-o",
                str(out_dir / f"dashboard_it{iteration}_{stage}.pdf"),
            ]
            + list(settings.extra_args)
        ]
        if is_last and settings.summary_on_last:
            renders.append(
                base
                + ["-o", str(out_dir / "dashboard.pdf"), "--force"]
                + list(settings.extra_args)
            )
        return [shlex.join(r) for r in renders]

    def render_script(
        self,
        calchome: Path,
        renders: list[str],
        settings: AutoDashboardSettings,
        cluster_config,
    ) -> str:
        """The job script: the cluster's SLURM header + one line a render.

        The first render goes through the stock ``body_single.sh.j2``; any
        further one (the summary) is appended as its own line.
        """
        from remote_handler.core.slurm_utils import get_slurm_str

        cfg = cluster_config
        header_add = "\n".join(
            line
            for line in (
                cfg.slurm_header_add,
                "#SBATCH --dependency=singleton",
                "#SBATCH --output=dashboard_slurm_%j.out",
            )
            if line
        )
        script = get_slurm_str(
            input_file=renders[0],
            python_script_path=_DASHBOARD_MODULE,
            jobname=self.jobname(calchome),
            partition=cfg.partition,
            n_cores=settings.n_cores,
            conda_env=cfg.conda_env,
            conda_source=cfg.conda_source,
            header_template=cfg.template,
            job_type_template="body_single.sh.j2",
            slurm_header_add=header_add,
            time_limit=settings.time_limit,
            memory=settings.memory,
        )
        extra = [f"python {_DASHBOARD_MODULE} {r}" for r in renders[1:]]
        return "\n".join([script.rstrip("\n"), *extra]) + "\n"

    def __call__(self, event) -> None:
        if not self.enabled:
            return
        calchome = Path(event.calchome).resolve()
        path, settings = self.settings(calchome)
        if settings is None:
            logger.debug(
                "No '%s' block in %s; no dashboard render.",
                AUTO_DASHBOARD_KEY,
                path or f"{calchome} (no dashboard config)",
            )
            return
        if not settings.enabled or event.stage not in settings.after:
            return
        if not settings.cluster:
            raise ValueError(
                f"{path}: '{AUTO_DASHBOARD_KEY}' needs 'cluster' (the "
                "ClusterConfig the render job runs on, e.g. lugus)."
            )
        out_dir = calchome / OUTPUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        renders = self.dashboard_args(
            calchome, event.iteration, event.stage, event.is_last, settings
        )
        script = out_dir / f"dashboard_it{event.iteration}_{event.stage}.sh"
        script.write_text(
            self.render_script(
                calchome, renders, settings, _load_cluster(settings.cluster)
            )
        )
        reply = self.submit(script)
        logger.info(
            "[it %d] submitted dashboard render after '%s': %s",
            event.iteration,
            event.stage,
            reply,
        )


__all__ = ["AutoDashboardSettings", "DashboardHook"]
