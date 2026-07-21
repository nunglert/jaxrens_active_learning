# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are cut from git tags via `setuptools-scm` (see the *Install* section
of `README.md`): tag a release with `git tag vX.Y.Z` and the package version
follows automatically. To release, rename the `[Unreleased]` section below to
the new `## [X.Y.Z] — YYYY-MM-DD`, add a fresh empty `[Unreleased]`, then tag.

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
