"""Render a dashboard for a *single* set of DFT-labelled configs.

The ``jaxrens-al-dashboard`` CLI is built for a finished *active-learning*
run: it reads ``AL_progress.json`` and renders cross-iteration progress. When
a loop dies after the DFT step but before recording a completed iteration --
``AL_progress.json`` then shows that iteration as unfinished, so the CLI skips
it -- the labelled configs are still on disk and worth looking at on their own.

This script wires the *same* project / alviz panels the AL dashboard uses
directly onto one extxyz of DFT-labelled configs (``REF_energy`` /
``REF_forces``, the keys ``remote_handler``'s evaluators write), so there is no
panel logic here, only wiring. The real logic lives in
:func:`jaxrens_al.dashboard.section_dashboards.run_dft`; when jaxrens_al is installed,
prefer the console subcommand ``jaxrens-al-dashboard dft``. Point it at the
evaluator output, e.g.::

    ~/miniconda3/envs/jaxrens_al/bin/python scripts/dft_dashboard.py \
        experiments/run6/.../evaluator_output/evaluator_it0_0.extxyz

Pass several files to pool them into one set. The ``ns_energy`` key (the
sampler's foundation-model energy, if present) is plotted against the DFT
reference as a parity panel -- "how far off was the sampler's surrogate". Flag
defaults can also come from a ``dashboard.{yaml,yml,toml,json}`` beside the
first input file (or ``--config PATH``); CLI flags still win.
"""

from jaxrens_al.dashboard.section_dashboards import run_dft

if __name__ == "__main__":
    raise SystemExit(run_dft())
