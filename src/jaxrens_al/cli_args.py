"""Reusable argparse argument groups shared by the dashboard entry points.

The standalone ``scripts/{ns,dft,training}_dashboard.py`` and the global
``jaxrens-al dashboard`` CLI expose overlapping flags -- output/cache options,
the NS temperature grid, the Steinhardt block, the reference-key options.
Defining each group once here keeps every consumer in step (a new Steinhardt
flag lands in all of them at once) and mirrors :mod:`jaxrens_al.dashboard.sections`,
which unifies the panel wiring those same flags feed. argparse only -- no
heavy imports, safe on the dashboard import path.

Each ``add_*`` helper adds its flags to an existing parser (some accept
defaults so a consumer can keep its natural values -- e.g. the scripts scan
Kelvin temperatures for eV models, the AL CLI reduced units).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_FORCE_HELP = "Recompute panel data even if a JSON cache exists."

# Config files auto-discovered in a run's calchome dir, in priority order. Each
# maps long-option names (``swap-pt``/``swap_pt``) to values; CLI flags still
# win. YAML is the project's native format; JSON/TOML need no extra dependency.
_CONFIG_BASENAMES = (
    "dashboard.yaml",
    "dashboard.yml",
    "dashboard.toml",
    "dashboard.json",
)


def find_config(calchome):
    """The dashboard config auto-discovered in ``calchome``, or None.

    Looks for ``dashboard.{yaml,yml,toml,json}`` (first match wins) so
    ``jaxrens-al dashboard <calchome>`` needs no flags when one is dropped
    beside the run.
    """
    calchome = Path(calchome)
    for name in _CONFIG_BASENAMES:
        p = calchome / name
        if p.is_file():
            return p
    return None


def load_config(path) -> dict:
    """Parse a dashboard config into ``{option: value}`` (argparse dests).

    Format is by extension: ``.yaml``/``.yml`` (lazy ``pyyaml``), ``.toml``
    (stdlib ``tomllib`` on 3.11+, else ``tomli``), ``.json`` (stdlib). Keys are
    normalized to argparse dests (``swap-pt`` -> ``swap_pt``). Raises on an
    unreadable/oddly-shaped file so a typo'd config fails loudly.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "Reading a YAML dashboard config needs pyyaml; install it or "
                "use a .json/.toml config instead."
            ) from exc
        data = yaml.safe_load(path.read_text()) or {}
    elif suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - py<3.11
            import tomli as tomllib
        data = tomllib.loads(path.read_text())
    elif suffix == ".json":
        data = json.loads(path.read_text() or "{}")
    else:
        raise ValueError(
            f"Unsupported dashboard config format {path.suffix!r} "
            "(use .yaml/.yml/.toml/.json)."
        )
    if not isinstance(data, dict):
        raise ValueError(
            f"Dashboard config {path} must be a mapping of option -> value."
        )
    return {str(k).replace("-", "_"): v for k, v in data.items()}


def add_config_arg(parser) -> None:
    """``--config PATH`` (else a ``dashboard.*`` beside the run is used)."""
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Dashboard config file (YAML/TOML/JSON) providing flag defaults; "
        "CLI flags still override it. Default: auto-detect "
        "<calchome>/dashboard.{yaml,yml,toml,json} if present.",
    )


def apply_config(
    parser, argv, args, logger, *, run_dir=None, positional="path"
):
    """Fold a config file's values in as defaults, return re-parsed ``args``.

    Resolves ``--config`` (else :func:`find_config` on the run dir), loads it,
    drops the non-overridable positional / ``config``, applies the rest via
    ``parser.set_defaults`` and re-parses -- so a config supplies defaults
    while any CLI flag still wins. An unknown key raises ``ValueError`` (with
    a did-you-mean hint) instead of being dropped: a misspelled setting would
    otherwise silently fall back to its default. No-op (returns ``args``
    unchanged) when no config is found.

    ``run_dir`` is the directory a ``dashboard.*`` config is auto-discovered in
    (default: the ``positional`` arg's value -- the AL CLI's calchome ``path``).
    Section subcommands pass their input dir + positional name (``output_dir``,
    ``results_dir``, ``config_files``) so a config drops beside the artifact. It
    may also be a *sequence* of dirs, tried in order (the NS subcommand passes
    the input dir and its parent, since the PDF defaults one level up -- so a
    config beside either is found).

    Beyond the flat argparse-dest keys, the config may carry the structured
    **panel** keys ``add`` / ``disable`` / ``panels`` (and an optional
    ``global:`` block that is merged into the flat keys). These are not argparse
    options; the raw (unparsed) mapping is stashed on the returned args as
    ``panel_config_raw`` for the caller to turn into panel specs (via the
    project's panel registry -- kept out of this argparse-only module).
    """
    if run_dir is None:
        run_dir = getattr(args, positional)
    run_dirs = (
        list(run_dir) if isinstance(run_dir, (list, tuple)) else [run_dir]
    )
    if args.config:
        path = Path(args.config)
    else:
        path = next((c for c in (find_config(d) for d in run_dirs) if c), None)
    if path is None:
        return args
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = load_config(path)
    cfg.pop(positional, None)
    cfg.pop("config", None)

    # Structured panel keys live outside the flat argparse namespace.
    panel_raw = {k: cfg.pop(k, None) for k in ("add", "disable", "panels")}
    # A ``global:`` block is an alias for the flat keys (both forms work); an
    # explicit flat key wins over the block.
    block = cfg.pop("global", None)
    if block is not None and not isinstance(block, dict):
        raise ValueError(
            f"{path}: 'global' must be a mapping of settings, got "
            f"{type(block).__name__}."
        )
    for k, v in (block or {}).items():
        cfg.setdefault(str(k).replace("-", "_"), v)

    valid = {a.dest for a in parser._actions} - {"help"}
    unknown = sorted(set(cfg) - valid)
    if unknown:
        raise ValueError(_unknown_keys_message(path, unknown, valid))
    parser.set_defaults(**cfg)
    logger.info("Applied dashboard config %s (%d setting(s)).", path, len(cfg))
    new_args = parser.parse_args(argv)
    if any(v is not None for v in panel_raw.values()):
        new_args.panel_config_raw = panel_raw
    return new_args


def _unknown_keys_message(path, unknown, valid) -> str:
    """Name each unknown config key, with the closest valid one if any."""
    import difflib

    def _show(key):  # long-option spelling, as the config files use it
        return key.replace("_", "-")

    lines = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(valid), n=1)
        hint = f" (did you mean '{_show(close[0])}'?)" if close else ""
        lines.append(f"  {_show(key)}{hint}")
    return (
        f"{path}: unknown config key(s):\n" + "\n".join(lines) + "\n"
        "Flat / `global:` keys must be options of this dashboard (see "
        "`jaxrens-al dashboard init-config`); panel settings go on an "
        "`add:` / `panels:` entry."
    )


def _config_value_repr(v) -> str:
    """Format a default value as a YAML scalar for :func:`dump_config`."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_config_value_repr(x) for x in v) + "]"
    s = str(v)
    reserved = {"true", "false", "null", "yes", "no", "on", "off"}
    if s == "" or s[0] in "!&*[]{}#>|%@`\"',:" or s.lower() in reserved:
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _short_help(help_text, limit: int = 60) -> str:
    """A one-line, length-capped gloss of an argparse help string."""
    if not help_text:
        return ""
    text = " ".join(help_text.split())
    for sep in (". ", " -- "):  # keep only the first sentence/clause
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def dump_config(parser, *, title: str | None = None) -> str:
    """Render a parser's options as a commented YAML config template.

    Walks the parser's *optional* actions (skipping the positional, ``--help``
    and ``--config``) and emits one ``long-option: default  # help`` line each,
    so the template is always in step with the live flags -- add a flag to a
    :mod:`jaxrens_al.cli_args` group and it appears here for free. Options whose
    default is ``None`` (no natural value) are emitted commented out; the rest
    reproduce the current default. The result loads cleanly via
    :func:`load_config` and every active key is a valid parser dest.
    """
    lines = []
    if title:
        lines.append(f"# {title} — default dashboard config.")
    lines.append(
        "# Keys are long-option names; set any to change its default "
        "(an explicit CLI flag still wins)."
    )
    lines.append("# Options with no natural default are listed commented out.")
    lines.append("")

    body = []
    for action in parser._actions:
        if not action.option_strings:  # positional (path/output_dir/...)
            continue
        if action.dest in ("help", "config"):
            continue
        if action.default is argparse.SUPPRESS:
            continue
        key = action.dest.replace("_", "-")
        help_txt = _short_help(action.help)
        if action.default is None:
            body.append((f"# {key}:", help_txt or "(unset)"))
        else:
            body.append(
                (f"{key}: {_config_value_repr(action.default)}", help_txt)
            )

    width = max((len(entry) for entry, _ in body), default=0)
    for entry, comment in body:
        lines.append(
            f"{entry.ljust(width)}  # {comment}" if comment else entry
        )
    return "\n".join(lines) + "\n"


def add_output_args(parser, *, force_help: str = _FORCE_HELP) -> None:
    """``-o/--output``, ``--datapath``, ``--force`` (shared by all four)."""
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PDF path (default: alongside the input).",
    )
    parser.add_argument(
        "--datapath",
        default=None,
        help="Where panels cache computed JSON (default: beside the output).",
    )
    parser.add_argument("--force", action="store_true", help=force_help)


def add_ref_key_args(parser) -> None:
    """``--energy-key`` / ``--forces-key`` (DFT/training reference keys)."""
    parser.add_argument(
        "--energy-key",
        default="REF_energy",
        help="info key of the reference energy (default: REF_energy).",
    )
    parser.add_argument(
        "--forces-key",
        default="REF_forces",
        help="array key of the reference forces (default: REF_forces).",
    )


def add_thermo_args(
    parser,
    *,
    k_B: float = 1.0,
    tmin: float = 0.05,
    tmax: float = 2.0,
    n_T: int = 50,
    swap_default: bool = False,
) -> None:
    """NS temperature grid + heatmap axis order (scripts + AL CLI).

    Defaults differ per consumer -- the scripts scan Kelvin for eV models
    (``k_B=8.617e-5``, ``tmin=100``, ...), the AL CLI reduced units -- so they
    are parameters here rather than baked in.
    """
    parser.add_argument(
        "--k-B",
        type=float,
        default=k_B,
        help="Boltzmann constant in the energy units of the .energies log "
        "(1.0 reduced; 8.617e-5 eV with T in K).",
    )
    parser.add_argument(
        "--tmin", type=float, default=tmin, help="Min NS temperature."
    )
    parser.add_argument(
        "--tmax", type=float, default=tmax, help="Max NS temperature."
    )
    parser.add_argument(
        "--n-T",
        type=int,
        default=n_T,
        help="Number of NS temperature points.",
    )
    parser.add_argument(
        "--swap-pt",
        action=argparse.BooleanOptionalAction,
        default=swap_default,
        help="Thermodynamics/Steinhardt heatmaps: pressure on x, temperature "
        "on y (use --no-swap-pt for T on x, P on y).",
    )


def add_processing_args(parser) -> None:
    """Property-agnostic knobs for the trajectory-processing (annotation) job.

    These govern *how* the single ``RemoteTrajectoryProcessor`` pass runs --
    where, how trajectories are sharded, how much runs concurrently -- not
    *what* it computes. Shared by any producer-backed panel, so they are
    configured once, globally, not per property. The panels that *need*
    annotation (e.g. a Steinhardt heatmap) are added via the config ``add:``
    list -- there are no per-panel CLI flags.
    """
    parser.add_argument(
        "--cluster",
        default="local",
        help="Where to run trajectory annotation: 'local' (default, "
        "scheduler-free on this machine) or the name/path of a remote_handler "
        "ClusterConfig JSON.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2000,
        help="Configs per shard when annotating via the processing machinery "
        "(default: 2000).",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=1,
        help="Max shards processed concurrently by the annotation job "
        "(default: 1).",
    )
    parser.add_argument(
        "--n-processors",
        type=int,
        default=1,
        help="CPUs requested per processing shard job (default: 1).",
    )
    parser.add_argument(
        "--relax-batch-size",
        type=int,
        default=32,
        help="Max structures relaxed per vmap kernel in the phase-diagram "
        "relax step (caps GPU memory; ~20-50 for MACE; default: 32).",
    )
    parser.add_argument(
        "--relax-shard-size",
        type=int,
        default=None,
        help="Override --shard-size for the phase-diagram relax/spacegroup job "
        "only (a GPU job, unlike the CPU annotation jobs); defaults to "
        "--shard-size when unset.",
    )


def relax_shard_size(args) -> int:
    """Shard size for the phase-diagram relax job: the ``--relax-shard-size``
    override if set, else the global ``--shard-size``.
    """
    override = getattr(args, "relax_shard_size", None)
    return override if override is not None else args.shard_size
