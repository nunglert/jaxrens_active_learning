# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are cut from git tags via `setuptools-scm` (see the *Install* section
of `README.md`): tag a release with `git tag vX.Y.Z` and the package version
follows automatically. To release, rename the `[Unreleased]` section below to
the new `## [X.Y.Z] — YYYY-MM-DD`, add a fresh empty `[Unreleased]`, then tag.

## [Unreleased]

### Added
- **Reference databases for any compound, not just elements.**
  `reference_db.fetch_mp_structures` generalises the Materials Project fetch:
  the system spec is an element (`"Si"`), a formula (`"SiO2"`), a chemical
  system (`"Si-O"`, `"Li-Fe-O"`, `"*-*-O"`) or a list of symbols, dispatched to
  MP's `chemsys`/`formula` constraint by `_parse_system`. Since MP's `chemsys`
  matches materials of *exactly* those elements, `include_subsystems=True`
  expands a system over all its sub-systems (`Li-Fe-O` → also `Li-O`, `Fe`, …),
  putting the end members back in. New phase-count filters — a compound system
  holds orders of magnitude more polymorphs than an element, and each is
  multiplied by the rattle grid: `max_e_above_hull`, `only_stable`, `max_sites`,
  `max_per_composition`, `max_structures`, `elements`, `exclude_elements`,
  `num_elements` (plus any other `mp_api` filter via `**extra_query`). Results
  are ordered most-stable-first (ties by `mp_id`) so the caps keep the relevant
  phases and the output is reproducible, and each structure is tagged with
  `mp_id`, `spacegroup`, `formula`, `e_above_hull` and `is_stable`.
  `build_reference_database`, `reference_eval.build_eos_reference` and the
  `mp_elemental`/`eos` resolver kinds all take the general system and forward
  the filters (`mp_query_kwargs` picks them out of a loose param bag);
  `eos_phase_scans(label_formula=True)` puts the composition in each phase's
  file stem (`eos_SiO2_mp-6922.extxyz`), which `build_eos_reference` enables
  automatically for a multi-compound fetch. `jaxrens-al refdb` takes the system
  as its positional argument with a *Materials Project filters* flag group
  (`--include-subsystems`, `--max-e-above-hull`, `--only-stable`, `--max-sites`,
  `--max-per-composition`, `--max-structures`, `--num-elements`,
  `--exclude-elements`).
  **Backwards compatible by addition**: `fetch_mp_elemental(element)` remains
  the single-element entry point, `system=`/`element=` are interchangeable
  (`resolve_system`), a legacy `element=` may now itself hold a formula or
  chemsys (so a panel's `reference_element="Si-O"` works with no dashboard
  change), and single-element EOS file stems are unchanged.
- **`jaxrens-al dashboard --iterations I [I ...]`.** Renders only the named AL
  iterations instead of the whole run — the switch for trying a panel out, since
  the remote annotation / model-evaluation pass then covers just those
  iterations' trajectories rather than every iteration's. Negative indices count
  back from the last recorded iteration (`--iterations -1`); a positive index may
  reach an on-disk working dir `AL_progress.json` never recorded, whose
  `--sections` pages still render. The combined PDF defaults to
  `dashboard_it<I>.pdf`, so testing one iteration never clobbers the whole-run
  `dashboard.pdf`. Also settable as `iterations:` in `dashboard.yaml`.
- **Selected-configs map (`selection_map`), a curated default panel.** Plots the
  selector stage's picks on an (NS `iter` × `run_id`) grid, over the sampler's
  full candidate pool in grey — so "which part of the NS run, and which
  replicas, did the selector actually draw from" is visible per iteration (and,
  on the summary page, one colour per AL iteration). Self-gates on a recorded
  `selector_output`, so a classic (selector-free) run is unaffected. Reads only
  the extxyz headers, never building an `Atoms` per frame, since NS pools run to
  hundreds of MB.
- **`<g(r)>(T)` RDF expectation panel (`rdf_expectation`).** Pairs
  `remote_handler`'s built-in `rdf` processor (declared as a `ProcessingNeed`,
  run by the existing resolver — `rdf` is registered as a second processor kind)
  with each replica's jaxrens `Monitor`. The whole `(replica, T, r)` surface is
  cached in the panel's JSON; what gets plotted is a selectable scalar
  `reduction` of each curve: `peak_height` (default), `peak_position`, `order`
  (τ = ∫|g−1|dr/Δr), `coordination` (4πρ∫r²g dr, with ρ = N/`<V>`(T) reweighted
  in the same pass and the window defaulting to the first minimum after the
  first peak), or `mean` over an `[r_min, r_max]` window. Repeatable and
  param-keyed (`rdf_<reduction>_rc<cutoff>`); entries differing only in
  `reduction` share one annotation job.
- **`panels/ns_expectation.py`**, the shared seam behind every NS expectation
  panel: replica trajectory/Monitor discovery, the subsampled-trajectory
  hold-to-dead-points reconstruction (now also for vector-valued observables),
  a `reweight` that applies the NS weights to a whole curve in one matmul
  (verified against `Monitor.expectation`), and the P-on-x/T-on-y heatmap mixin
  the Steinhardt / thermodynamics / RDF panels previously each copied.

### Fixed
- **`ns_thermodynamics` with `quantity: volume` works again.** It read
  `Monitor.dead_volumes`, which `MonitorCollection.from_multi_run_directory`
  deliberately leaves `None` (an NPT run's stored energy is already the enthalpy
  `U + P*V`; the bare cell-volume series is stashed as `volume_trace`), so
  `<V>(T)` raised instead of plotting. It now reads `volume_trace` with
  `dead_volumes` as the fallback, and raises an explicit "NVT run has no volume"
  error when neither exists. A second `ns_thermodynamics` entry also no longer
  collides with the curated heat-capacity panel: the name is quantity-keyed
  (`ns_volume`), while `heat_capacity` keeps the historical `ns_thermodynamics`.

## [0.1.0] — 2026-07-21

### Added
- **Single `jaxrens-al` command-line entry point with `--version`.** The
  project now installs one console script, `jaxrens-al`
  (`jaxrens_al.__main__:main`, also runnable as `python -m jaxrens_al`), which
  dispatches subcommands. `jaxrens-al --version` reports the installed version;
  `jaxrens-al dashboard …` and `jaxrens-al refdb …` route to the existing
  dashboard and reference-database tools (each imported lazily, so `refdb`
  never pulls in the dashboard stack and vice versa). The `dashboard`
  subcommand keeps its own `ns` / `dft` / `training` / `init-config` sub-flows,
  e.g. `jaxrens-al dashboard ns <dir>`.
- **Git-tag-driven versioning via `setuptools-scm`.** `pyproject.toml` declares
  `dynamic = ["version"]` with a `[tool.setuptools_scm]` table (and a
  `fallback_version` for tagless / shallow clones); `jaxrens_al.__version__`
  reads the installed version back through `importlib.metadata`, mirroring the
  `jaxrens` package. A tagged release (`git tag vX.Y.Z`) becomes the version;
  untagged checkouts report a `0.1.devN+g<hash>` development version. CI now
  fetches full history + tags so the derived version is accurate. The reported
  version refreshes on reinstall, not on a bare `git pull`.

### Changed
- **The two console scripts were consolidated into `jaxrens-al` subcommands
  (breaking).** `jaxrens-al-dashboard` → `jaxrens-al dashboard` and
  `jaxrens-al-refdb` → `jaxrens-al refdb`; the old hyphenated scripts are
  removed. Update any wrappers, aliases, or CI invocations accordingly (the
  checkout scripts under `scripts/` and the Python `jaxrens_al.cli` /
  `jaxrens_al.run.reference_db` entry points are unchanged).

[Unreleased]: https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/madsen-s-research-group/jaxrens_active_learning/-/commits/main
