"""Render a dashboard for a *single* jaxrens NS run (no AL_progress.json).

The ``jaxrens-al-dashboard`` CLI is built for a finished *active-learning*
run: it reads ``AL_progress.json`` and shows cross-iteration progress
(dataset growth, parity, sampled-vs-DFT energies, EOS). When a loop only got
as far as the iteration-0 NS sampling -- e.g. it died pulling output back --
there is no ``AL_progress.json`` and no AL progress to show, so the CLI bails.

This script reuses the *same* project panels (``jaxrens_al.panels``) the AL
dashboard uses, wired directly onto one multi-run NS output directory (the
``<prefix>.runNN.energies`` layout) -- so there is no panel logic here, only
wiring. The real logic lives in :func:`jaxrens_al.section_dashboards.run_ns`;
when jaxrens_al is installed, prefer the equivalent console subcommand
``jaxrens-al-dashboard ns``. Run it in an env with jaxrens installed, e.g.::

    ~/miniconda3/envs/jaxrens_al/bin/python scripts/ns_dashboard.py \
        experiments/run4/calchome/phase1_mace/0/ns_sampler/results/output \
        --prefix mace_sn16 -o experiments/run4/ns_dashboard.pdf

Add NS ``<q_l>(T)`` Steinhardt heatmap(s) via a ``dashboard.*`` config ``add:``
list (one entry per l / r_cut) -- there is no CLI flag. The per-replica
trajectories are annotated by the real ``remote_handler`` processing pipeline
(``SteinhardtProcessor``); ``--cluster`` picks where it runs -- ``local``
(default, scheduler-free on this machine) or a ClusterConfig JSON for a
dedicated/existing postprocessing cluster. Same code path either way. Flag
defaults can also come from a ``dashboard.{yaml,yml,toml,json}`` beside the
output dir (or ``--config PATH``); CLI flags still win.
"""

from jaxrens_al.section_dashboards import run_ns

if __name__ == "__main__":
    raise SystemExit(run_ns())
