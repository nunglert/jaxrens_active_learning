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
here** (`jaxrens_al.run.sampler`), implementing `remote_handler`'s `Sampler`
interface for `jaxrens` — it does *not* use `remote_handler`'s own deprecated
`NSTrajectorySampler`.

## What lives here

- `jaxrens_al.run.sampler` — `JaxrensNSSampler`, the native nested-sampling
  `Sampler`. Per AL iteration it injects the freshly trained model into a
  `jaxrens` YAML config (`backend.checkpoint_path`), runs `jaxrens run` on the
  cluster via a `remote_handler` `SingleScriptTask`, and reads the resulting
  `<prefix>.traj.extxyz` trajectories. Needs the `[run]` extra.
- `jaxrens_al.template_dir()` — directory of the shipped SLURM body template
  `body_jaxrens.sh.j2` (the `jaxrens run` job body). Copy it into your
  `REMOTE_HANDLER_SLURM_TEMPLATES` dir alongside your cluster header.
- `jaxrens_al.dashboard.run_layout` — reads a remote_handler run's artifacts
  (`AL_progress.json` + the `calchome` tree). **Reads files only; does not
  import remote_handler** — the dashboard is coupled to the orchestrator's
  output *format*, not its code.
- `jaxrens_al.dashboard.build_dashboard(calchome)` — assembles an alviz
  dashboard for a run. Panels are registered **only when their inputs exist**,
  so a fresh/stub run yields a small board and a full run a richer one; the
  grid auto-sizes.
- `jaxrens_al.dashboard.panels` — project panels. Each takes an alviz skeleton (an
  abstract `ParityPanel`/`HeatmapPanel`, or the base `Entry`) and equips it
  with computation that reads the run's artifacts. Generic `Extxyz*` readers
  belong in `alviz`. Current panels:
  - `DatasetGrowthPanel` — training-set size per iteration (`AL_progress`).
  - `NSSampledEnergyPanel` — per-iteration `ns_energy` distribution of the
    jaxrens-sampled configs.
  - `QuestsPCAPanel` — QUESTS descriptor PCA map of one iteration's NS
    candidates, highlighting the resampled ones (`quests_pca.json`).
  - `ResampledGridPanel` — the resampled configs scattered on a (NS
    iteration, replica/`run_id`) grid (`resampled_provenance.json`).
  - `MoveAcceptancePanel` — per-move MCMC acceptance rate vs NS iteration,
    from jaxrens' `<prefix>.acc_rates.h5` (read via its own reader).
  - `RESwapAcceptancePanel` — inter-replica-exchange swap acceptance per
    replica pair vs iteration, from jaxrens' `<prefix>.re_stats.h5`.
  - `AtomicEnvsPanel` — per-atom QUESTS-descriptor PCA of an iteration's
    sampled configs, coloured by species (quests + scikit-learn PCA).
  - `EquationOfStatePanel` — multi-phase EOS (3 sub-panels: E(V), H(P),
    ΔH(P)) with Birch-Murnaghan fits, comparing the iteration's model
    (jaxrens backend) against DFT `REF_energy`; one reference scan per phase.
  - `NeuralILParityPanel` (← `ParityPanel`) — energy/force parity from the
    trainer's `save_data` `.npz` bundles.
  - `NeuralILTrainingCurve` — validation E/F RMSE vs epoch, parsed from the
    trainer's `error_*.txt`.
  - `NSThermodynamicsHeatmap` (← `HeatmapPanel`) — heat capacity / mean
    volume vs temperature, via `jaxrens.postprocess.Monitor` on the NS
    `.energies` + `.final.checkpoint.h5` (NumPy postprocess — no JAX needed
    to build the dashboard).
- `jaxrens_al.run.reference_db` — builds a **starting/reference structure
  database** for iteration zero: fetch an element's experimentally observed
  phases from the Materials Project (`fetch_mp_elemental`), expand each into a
  cloud of rattled, volume-scaled supercells (`rattle_atoms`), and write an
  `extxyz` (`build_reference_database`, or the `jaxrens-al refdb` CLI). The
  Materials Project query needs the `[refdb]` extra (`mp_api` + `pymatgen`);
  the rattling half is pure numpy + ase.
- `examples/run_al.py` — concrete wiring of the AL loop (jaxrens sampler +
  VASP evaluator + NeuralIL trainer + dashboard); needs the `[run]` extra + a
  configured cluster. Run it with `STUB = True` first to exercise the wiring
  with no cluster. `examples/inputs/ns.yaml` is the base jaxrens NS config.

## Install

The private dependencies are pinned in `pyproject.toml` as **SSH** git URLs:

| dep | host | extra |
|-----|------|-------|
| `alviz` | GitLab (madsen group) | base |
| `remote_handler` | GitLab (madsen group) | `[run]` |
| `jaxrens` | GitHub (`nunglert/jaxrens`) | `[run]` |

An extra `[refdb]` (`mp_api` + `pymatgen`) is needed only to fetch starting
structures from the Materials Project via `jaxrens_al.run.reference_db`.

**Normal install (SSH keys)** — on any machine whose SSH keys are registered
with GitLab (and GitHub, for `jaxrens`) — your laptop, a cluster like musica —
the URLs resolve with zero config:
```bash
pip install -e ".[run,dev,refdb]"
```
SSH is the default precisely because humans have SSH keys; only CI is special
(below). Caveat: this pulls `alviz@main` (etc.) from the remote, so it reflects
what's **pushed** — if you have unpushed alviz work, use the editable sibling
for it (next).

**Local dev (editable siblings)** — to work against unpushed/local changes in
the siblings (or offline), install the checkouts editable and then this project
with `--no-deps` so pip doesn't fetch the URLs at all:
```bash
pip install -e ../alviz
pip install -e ../remote_handler        # only for the [run] extra
pip install -e ../jaxrens               # only for the [run] extra
pip install -e ".[dev]" --no-deps       # this project (deps already present)
```

**CI / GitLab** — the runners have no SSH key, so `before_script` rewrites the
SSH URLs to authenticated HTTPS at clone time via `git config insteadOf`:
`CI_JOB_TOKEN` for the GitLab deps, a masked `GH_TOKEN` for the GitHub one.

The design: `pyproject.toml` holds SSH URLs (the common case — humans/clusters
with SSH keys install with no setup), and the **one** environment without an SSH
key, CI, rewrites them to HTTPS+token. The file is never touched per-machine.

## Command-line interface

Everything is under one console script, `jaxrens-al` (also `python -m
jaxrens_al`), which dispatches subcommands:

```bash
jaxrens-al --version                 # installed version (setuptools-scm, from the git tag)
jaxrens-al dashboard path/to/calchome   # build a run dashboard (see "Visualize a run")
jaxrens-al refdb Si si_start.extxyz     # build a starting database (see below)
```

The version is derived from the git tag by **setuptools-scm** (mirroring
`jaxrens`): `pyproject.toml` declares `dynamic = ["version"]`, so a tagged
release (`git tag vX.Y.Z`) becomes the version and untagged checkouts report a
`0.1.devN+g<hash>` development version. It refreshes on reinstall. Notable
changes per version are tracked in [`CHANGELOG.md`](CHANGELOG.md).

## Build a starting database

Before the first AL iteration you need seed structures. `reference_db` turns an
element into a rattled, volume-scaled `extxyz` drawn from its Materials Project
phases (needs the `[refdb]` extra and an `MP_API_KEY`):

```bash
export MP_API_KEY=...
jaxrens-al refdb Si si_start.extxyz --target-cutoff 3 \
    --rattle-strengths 0.0 0.2 --n-vol 20 --vol-min 0.9 --vol-max 1.1
```

or programmatically:

```python
from jaxrens_al.run.reference_db import build_reference_database

configs = build_reference_database(
    "Si", target_cutoff=3.0, output="si_start.extxyz"
)
```

## Visualize a run

Plotting is decoupled from running the loop: `run_al.py` only produces the
`calchome` run dir, and a standalone CLI turns it into a PDF (re-runnable, like
the legacy `al_dashboard.py`).

The dashboard is **multipage — one page per AL iteration**, each page carrying
the same panel layout (parity, sampled-energy, evaluated E/V, NS
thermodynamics) computed on *that iteration's* artifacts, so the result reads as
a flip-book of the run. Each iteration is also written on its own under
`calchome/plots/Iteration<i>.pdf`, then stitched into the combined output via
`alviz.merge_pdfs` (pypdf). A panel appears on a page only when its inputs exist
for that iteration, so early iterations get a smaller page.

```bash
jaxrens-al dashboard path/to/calchome              # -> calchome/dashboard.pdf (+ plots/Iteration*.pdf)
jaxrens-al dashboard path/to/calchome -o out.pdf --force
jaxrens-al dashboard path/to/calchome --dry-run    # report per-iteration panel layout only
jaxrens-al dashboard path/to/calchome --only sampled_ns_energy neuralil_parity   # subset of panels
jaxrens-al dashboard path/to/calchome --exclude dataset_growth                   # all but these
```

From a checkout without installing, the thin wrapper is equivalent:
`python scripts/al_dashboard.py path/to/calchome`. A single iteration's page is
also available programmatically:

```python
from jaxrens_al.dashboard import build_iteration_dashboard

board = build_iteration_dashboard("path/to/calchome", 3, datapath="some/dir")
board.savefig("iteration3.pdf")
```

(The cross-iteration *summary* board — dataset growth etc. on one page — is
still available as `jaxrens_al.build_dashboard(calchome)`.)

## Test

The dashboard side is fully testable without a cluster. The suite mirrors the
code's layers — `test_run_layout.py` (artifact reading), `test_panels.py` (each
panel's `compute_data`), `test_dashboard.py` (assembly/selection/layout),
`test_cli.py` (CLI + multipage merge), `test_sampler.py` (the run-side sampler).
All synthetic run dirs come from one `make_calchome` factory in `conftest.py`.

```bash
pytest                 # everything installed here runs
pytest -m local        # dashboard side only (alviz + ase + pypdf; no producer)
pytest -m integration  # run-side sampler (needs remote_handler)
```

Dashboard-side files are marked `local`; `test_sampler.py` is `integration` and
also self-skips (via `importorskip`) when `remote_handler` is absent.
