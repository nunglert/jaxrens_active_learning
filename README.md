# jaxrens_active_learning

The **composition root** for jaxrens active-learning projects. It ties
together the layered packages and owns all application-specific glue:

```
jaxrens_active_learning   ← this repo: the jaxrens NS Sampler + wiring + panels + dashboard
  ├──▶ remote_handler     orchestration loop, Sampler/Evaluator/Trainer ABCs,
  │                       concrete evaluators (VASP/ORCA) and trainers (MACE/NeuralIL)
  ├──▶ alviz              dashboard engine + generic, reusable panels
  └──▶ jaxrens            the nested-sampling method (runs on the cluster)
```

No package depends on another except this one, which depends on all of them.
`alviz` and `remote_handler` never know about each other — this repo is the
only place that bridges them. The concrete **nested-sampling Sampler lives
here** (`jaxrens_al.sampler`), implementing `remote_handler`'s `Sampler`
interface for `jaxrens` — it does *not* use `remote_handler`'s own deprecated
`NSTrajectorySampler`.

## What lives here

- `jaxrens_al.sampler` — `JaxrensNSSampler`, the native nested-sampling
  `Sampler`. Per AL iteration it injects the freshly trained model into a
  `jaxrens` YAML config (`backend.checkpoint_path`), runs `jaxrens run` on the
  cluster via a `remote_handler` `SingleScriptTask`, and reads the resulting
  `<prefix>.traj.extxyz` trajectories. Needs the `[run]` extra.
- `jaxrens_al.template_dir()` — directory of the shipped SLURM body template
  `body_jaxrens.sh.j2` (the `jaxrens run` job body). Copy it into your
  `REMOTE_HANDLER_SLURM_TEMPLATES` dir alongside your cluster header.
- `jaxrens_al.run_layout` — reads a remote_handler run's artifacts
  (`AL_progress.json` + the `calchome` tree). **Reads files only; does not
  import remote_handler** — the dashboard is coupled to the orchestrator's
  output *format*, not its code.
- `jaxrens_al.dashboard.build_dashboard(calchome)` — assembles an alviz
  dashboard for a run: generic panels from `alviz.panels` fed the run's
  paths, plus this project's own panels.
- `jaxrens_al.panels` — project-specific panels (e.g. `DatasetGrowthPanel`)
  that understand remote_handler's layout. Generic panels belong in `alviz`.
- `examples/run_al.py` — concrete wiring of the AL loop (jaxrens sampler +
  VASP evaluator + NeuralIL trainer + dashboard); needs the `[run]` extra + a
  configured cluster. Run it with `STUB = True` first to exercise the wiring
  with no cluster. `examples/inputs/ns.yaml` is the base jaxrens NS config.

## Install

The private dependencies are pinned in `pyproject.toml` as git URLs:

| dep | host | extra |
|-----|------|-------|
| `alviz` | GitLab (madsen group) | base |
| `remote_handler` | GitLab (madsen group) | `[run]` |
| `jaxrens` | GitHub (`nunglert/jaxrens`) | `[run]` |

**Local dev** — the git URLs aren't all reachable locally (alviz isn't
pushed yet, and they need auth), so install the sibling checkouts editable
and then the project itself with `--no-deps` so pip doesn't try to fetch the
URLs:
```bash
pip install -e ../alviz
pip install -e ../remote_handler        # only for the [run] extra
pip install -e ../jaxrens               # only for the [run] extra
pip install -e ".[dev]" --no-deps       # this project (deps already present)
```

**CI / GitLab** — let pip resolve the git URLs, injecting per-host auth in
`before_script`: `CI_JOB_TOKEN` (via `git config insteadOf`) for the GitLab
deps, and a masked `GH_TOKEN` for the GitHub one. The dashboard job needs
only `alviz`; the run job adds `remote_handler` + `jaxrens`.

## Visualize a run

```python
from jaxrens_al import build_dashboard

board = build_dashboard("path/to/calchome")
board.savefig("dashboard.pdf")
```

## Test

The dashboard side is fully testable without a cluster (synthetic run dir):
```bash
pytest
```
