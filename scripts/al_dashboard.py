#!/usr/bin/env python
"""Thin wrapper to build an AL-run dashboard from a checkout (no install).

Mirrors the legacy ``jaxns/scripts/utility/al_dashboard.py`` invocation::

    python scripts/al_dashboard.py path/to/calchome [-o out.pdf] [--force]

When jaxrens_al is installed, prefer the console script
``jaxrens-al dashboard``. The real logic lives in :mod:`jaxrens_al.cli`.
"""

from jaxrens_al.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
