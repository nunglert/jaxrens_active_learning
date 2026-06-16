"""Assemble an alviz dashboard for a remote_handler active-learning run.

This is the heart of the composition root: it knows *where* remote_handler /
jaxrens put a run's artifacts (via :mod:`jaxrens_al.run_layout`) and feeds
those paths into panels -- generic ``Extxyz*`` readers from ``alviz.panels``
plus this project's own panels (``jaxrens_al.panels``). Neither alviz nor
remote_handler knows about the other; this module is the only bridge.

Panels are registered **only when their inputs exist**, so the same call
produces a 2-panel summary for a fresh/stub run and a fuller board once NS
postprocessing artifacts and NeuralIL training results are present. The grid
auto-sizes to the panels that have data.
"""

import math
from pathlib import Path

from alviz import Dashboard
from alviz.panels import ExtxyzSampledEnergyVolume

from jaxrens_al.panels import (
    DatasetGrowthPanel,
    NeuralILParityPanel,
    NeuralILTrainingCurve,
    NSSampledEnergyPanel,
    NSThermodynamicsHeatmap,
)
from jaxrens_al.run_layout import (
    evaluated_configs,
    evaluated_configs_per_iteration,
    iteration_dir,
    load_progress,
    ns_result_dir,
    ns_result_dirs,
    sampled_configs,
    sampled_configs_per_iteration,
    training_error_files,
    training_error_for_iteration,
    training_npz_files,
    training_npz_for_iteration,
)

# remote_handler labels ground-truth energies/forces with these keys.
REF_ENERGY_KEY = "REF_energy"
REF_FORCES_KEY = "REF_forces"


def _assemble_entries(calchome, datapath, ns_prefix, ns_k_B):
    """Build the list of panels whose inputs exist for this run."""
    progress = load_progress(calchome)
    entries = []

    # Project panel: training-set growth (always -- needs only AL_progress).
    entries.append(DatasetGrowthPanel(calchome=calchome, datapath=datapath))

    # Generic alviz panel: distribution of evaluated (DFT) energies/forces/
    # volumes. Fed the producer's reference keys -- alviz stays neutral.
    evald = evaluated_configs(calchome, progress)
    if evald:
        entries.append(
            ExtxyzSampledEnergyVolume(
                config_files=evald,
                datapath=datapath,
                energy_key=REF_ENERGY_KEY,
                forces_key=REF_FORCES_KEY,
            )
        )

    # Project panel: sampled NS-energy distribution across iterations.
    if sampled_configs(calchome, progress):
        entries.append(
            NSSampledEnergyPanel(calchome=calchome, datapath=datapath)
        )

    # Project panel: NeuralIL parity from the trainer's .npz bundles.
    npz = training_npz_files(calchome)
    if npz:
        entries.append(NeuralILParityPanel(npz_files=npz, datapath=datapath))

    # Project panel: NeuralIL training-error curve from error_*.txt.
    err = training_error_files(calchome)
    if err:
        entries.append(
            NeuralILTrainingCurve(error_files=err, datapath=datapath)
        )

    # Project panel: NS heat capacity vs T (needs the run's .energies +
    # .final.checkpoint.h5 collected back into the iteration dirs).
    ns_dirs = ns_result_dirs(calchome, prefix=ns_prefix)
    if ns_dirs:
        entries.append(
            NSThermodynamicsHeatmap(
                result_dirs=ns_dirs,
                quantity="heat_capacity",
                k_B=ns_k_B,
                prefix=ns_prefix,
                datapath=datapath,
            )
        )

    return entries


def build_dashboard(
    calchome,
    *,
    datapath=None,
    ncols: int = 2,
    width_per_panel: float = 4.0,
    height_per_panel: float = 4.0,
    ns_prefix: str = "ns",
    ns_k_B: float = 1.0,
    compute: bool = True,
) -> Dashboard:
    """Build a dashboard summarizing a remote_handler / jaxrens run.

    Args:
        calchome: Path to a run directory (holds ``AL_progress.json``).
        datapath: Where panels cache their computed JSON. Defaults to
            ``calchome/dashboard``.
        ncols: Number of columns; rows auto-size to the available panels.
        width_per_panel, height_per_panel: Panel sizing.
        ns_prefix: jaxrens ``out_file_prefix`` of the NS run files.
        ns_k_B: Boltzmann constant for the thermodynamics panel, in the energy
            units of the NS energies (1.0 reduced; 8.617e-5 eV with T in K).
        compute: If True, run the compute/plot lifecycle before returning.

    Returns:
        The assembled :class:`alviz.Dashboard`.
    """
    calchome = Path(calchome)
    datapath = Path(datapath) if datapath else calchome / "dashboard"
    datapath.mkdir(parents=True, exist_ok=True)

    entries = _assemble_entries(calchome, datapath, ns_prefix, ns_k_B)

    nrows = max(1, math.ceil(len(entries) / ncols))
    board = Dashboard(nrows, ncols, width_per_panel, height_per_panel)
    for idx, entry in enumerate(entries):
        row, col = divmod(idx, ncols)
        board.register_entry(entry, (str(row), str(col)))

    if compute:
        board.allocate_panels()
        board.compute_data()
        board.plot_panels()

    return board


# ---------------------------------------------------------------------------
# Per-iteration (multipage) assembly.
#
# The CLI renders one dashboard page per AL iteration. Each page reuses the
# same panels as the summary, but fed *only that iteration's* artifacts, so the
# pages share a layout and read as a flip-book of the run. As above, a panel is
# registered only when its inputs exist for that iteration -- early iterations
# (no NS postprocess / no training results yet) get a smaller page.
# ---------------------------------------------------------------------------


def _assemble_iteration_entries(
    calchome,
    datapath,
    index,
    eval_files,
    sampled_files,
    ns_prefix,
    ns_k_B,
    force_recompute,
):
    """Panels for a single AL iteration whose inputs exist."""
    it_dir = iteration_dir(calchome, index)
    common = dict(datapath=datapath, force_recompute=force_recompute)
    entries = []

    # Evaluated (DFT) energy/volume distribution for this iteration.
    if eval_files:
        entries.append(
            ExtxyzSampledEnergyVolume(
                config_files=eval_files,
                energy_key=REF_ENERGY_KEY,
                forces_key=REF_FORCES_KEY,
                **common,
            )
        )

    # Sampled NS-energy distribution for this iteration (single box).
    if sampled_files:
        entries.append(
            NSSampledEnergyPanel(
                config_files=sampled_files,
                iteration_index=index,
                **common,
            )
        )

    # NeuralIL parity from this iteration's training .npz bundles.
    npz = training_npz_for_iteration(it_dir)
    if npz:
        entries.append(NeuralILParityPanel(npz_files=npz, **common))

    # NeuralIL training-error curve from this iteration's error_*.txt.
    err = training_error_for_iteration(it_dir)
    if err:
        entries.append(NeuralILTrainingCurve(error_files=err, **common))

    # NS thermodynamics from this iteration's NS run.
    ns_dir = ns_result_dir(it_dir, prefix=ns_prefix)
    if ns_dir is not None:
        entries.append(
            NSThermodynamicsHeatmap(
                result_dirs=[ns_dir],
                quantity="heat_capacity",
                k_B=ns_k_B,
                prefix=ns_prefix,
                **common,
            )
        )

    return entries


def build_iteration_dashboard(
    calchome,
    index,
    *,
    datapath,
    eval_files=None,
    sampled_files=None,
    ncols: int = 2,
    width_per_panel: float = 4.0,
    height_per_panel: float = 4.0,
    ns_prefix: str = "ns",
    ns_k_B: float = 1.0,
    compute: bool = True,
    force_recompute: bool = False,
):
    """Build one dashboard page for a single AL iteration.

    Args:
        calchome: Path to the run directory.
        index: AL iteration index (its working dir is ``calchome/<index>``).
        datapath: Where this iteration's panels cache their JSON (must exist;
            use a per-iteration subdir to avoid clobbering other pages).
        eval_files / sampled_files: This iteration's evaluated/sampled config
            files. If None, they are looked up from ``AL_progress.json``.
        force_recompute: Recompute panel data even if a JSON cache exists.

    Returns:
        The assembled :class:`alviz.Dashboard`, or ``None`` if this iteration
        has no panels with data.
    """
    calchome = Path(calchome)
    if eval_files is None:
        eval_files = evaluated_configs_per_iteration(calchome)[index]
    if sampled_files is None:
        sampled_files = sampled_configs_per_iteration(calchome)[index]

    entries = _assemble_iteration_entries(
        calchome,
        Path(datapath),
        index,
        eval_files,
        sampled_files,
        ns_prefix,
        ns_k_B,
        force_recompute,
    )
    if not entries:
        return None

    nrows = max(1, math.ceil(len(entries) / ncols))
    board = Dashboard(nrows, ncols, width_per_panel, height_per_panel)
    for idx, entry in enumerate(entries):
        row, col = divmod(idx, ncols)
        board.register_entry(entry, (str(row), str(col)))

    if compute:
        board.allocate_panels()
        board.compute_data()
        board.plot_panels()
        board._fig.suptitle(f"AL iteration {index}")

    return board
