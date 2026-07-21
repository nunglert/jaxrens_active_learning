"""Top-level ``jaxrens-al`` command dispatcher.

The one console script for this project. It routes to the two sub-tools,
keeping their argument parsers (and everything below them) untouched::

    jaxrens-al dashboard <calchome> ...   -> jaxrens_al.cli:main
    jaxrens-al refdb ...                  -> jaxrens_al.run.reference_db:main
    jaxrens-al --version                  -> print the installed version

``dashboard`` further dispatches its own ``ns``/``dft``/``training``/
``init-config`` sub-flows (see :mod:`jaxrens_al.cli`), so e.g.
``jaxrens-al dashboard ns <dir>`` still works.

Also runnable as ``python -m jaxrens_al``.
"""

from __future__ import annotations

import sys

from jaxrens_al import __version__

_SUBCOMMANDS = ("dashboard", "refdb")

_USAGE = f"""\
usage: jaxrens-al [--version] <command> [<args>...]

jaxrens active-learning command-line interface (version {__version__}).

commands:
  dashboard   Build/inspect dashboards for an AL run or a raw artifact dir.
              (run 'jaxrens-al dashboard --help' for its own subcommands)
  refdb       Build a starting reference database (iteration-zero seeds).

options:
  -V, --version   Print the installed version and exit.
  -h, --help      Show this message and exit.
"""


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``jaxrens-al`` console script."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE, end="")
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd == "dashboard":
        from jaxrens_al.cli import main as dashboard_main

        return dashboard_main(rest)
    if cmd == "refdb":
        from jaxrens_al.run.reference_db import main as refdb_main

        return refdb_main(rest)

    print(
        f"jaxrens-al: unknown command {cmd!r}\n\n{_USAGE}",
        end="",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
