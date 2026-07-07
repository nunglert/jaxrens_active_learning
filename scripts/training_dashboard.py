"""Render a dashboard for a *single* training run (no AL_progress.json).

The ``jaxrens-al-dashboard`` CLI only renders iterations that
``AL_progress.json`` records as *finished*. When a loop trains a model but then
dies before completing the iteration (e.g. the readback fails), that training
is invisible to the AL dashboard even though its logs and model are on disk.

This script wires the *same* project panels the AL dashboard uses directly onto
one training ``results`` directory, so there is no panel logic here, only
wiring. It auto-detects the trainer by what is on disk (recursively):

* **MACE** (``*_train.txt`` JSON-lines log) -> :class:`MaceTrainingCurve`.
* **NeuralIL** (``error_*.txt`` logs and ``save_data`` ``.npz`` bundles)
  -> :class:`NeuralILTrainingCurve` + :class:`NeuralILParityPanel`.

Parity of the *trained model* against the DFT reference is shown too:

* **NeuralIL** reads the trainer's ``.npz`` bundle directly
  (:class:`NeuralILParityPanel`).
* **MACE** writes no such bundle, so :class:`ModelParityPanel` *re-scores* a
  labelled set (the validation split ``db_val.extxyz``, else ``db_train``) by
  loading the trained model through the jaxrens backend -- this is a real
  predicted-vs-reference parity, not the sampler-surrogate parity that
  ``dft_dashboard`` shows.

The model bundle and validation set are auto-detected under ``results_dir``;
override with ``--model`` / ``--valid-set`` / ``--backend``. If a
``db_train.extxyz`` training set is present, its energy/volume spread and
descriptor diversity are shown as well. The real logic lives in
:func:`jaxrens_al.section_dashboards.run_training`; when jaxrens_al is
installed, prefer the console subcommand ``jaxrens-al-dashboard training``.
Point at the training results dir, e.g.::

    ~/miniconda3/envs/jaxrens_al/bin/python scripts/training_dashboard.py \
        experiments/run6/calchome/phase1_mace/0/train/results

Flag defaults can also come from a ``dashboard.{yaml,yml,toml,json}`` beside
the results dir (or ``--config PATH``); CLI flags still win.
"""

from jaxrens_al.section_dashboards import run_training

if __name__ == "__main__":
    raise SystemExit(run_training())
