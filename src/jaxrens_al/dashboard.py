"""Assemble an alviz dashboard for a remote_handler active-learning run.

This is the heart of the composition root: it knows *where* remote_handler
puts a run's artifacts (via :mod:`jaxrens_al.run_layout`) and feeds those
paths into alviz panels -- generic ones from ``alviz.panels`` plus this
project's own panels. Neither alviz nor remote_handler knows about the
other; this module is the only place that bridges them.
"""

from pathlib import Path

from alviz import Dashboard
from alviz.panels import ExtxyzSampledEnergyVolume

from jaxrens_al.panels import DatasetGrowthPanel
from jaxrens_al.run_layout import evaluated_configs, load_progress

# remote_handler labels ground-truth energies/forces with these keys.
REF_ENERGY_KEY = "REF_energy"
REF_FORCES_KEY = "REF_forces"


def build_dashboard(
    calchome,
    *,
    datapath=None,
    width_per_panel: float = 4.0,
    height_per_panel: float = 4.0,
    compute: bool = True,
) -> Dashboard:
    """Build a dashboard summarizing a remote_handler run.

    Args:
        calchome: Path to a remote_handler run directory (holds
            ``AL_progress.json``).
        datapath: Where panels cache their computed JSON. Defaults to
            ``calchome/dashboard``.
        width_per_panel, height_per_panel: Panel sizing.
        compute: If True, run the compute/plot lifecycle before returning.

    Returns:
        The assembled :class:`alviz.Dashboard`.
    """
    calchome = Path(calchome)
    datapath = Path(datapath) if datapath else calchome / "dashboard"
    datapath.mkdir(parents=True, exist_ok=True)

    progress = load_progress(calchome)
    evald = evaluated_configs(calchome, progress)

    board = Dashboard(1, 2, width_per_panel, height_per_panel)

    # Generic alviz panel: distribution of evaluated (DFT) energies/forces/
    # volumes. Fed the producer's reference keys -- alviz stays neutral.
    board.register_entry(
        ExtxyzSampledEnergyVolume(
            config_files=evald,
            datapath=datapath,
            energy_key=REF_ENERGY_KEY,
            forces_key=REF_FORCES_KEY,
        ),
        ("0", "0"),
    )

    # Project-specific panel: how the training set grew across iterations.
    board.register_entry(
        DatasetGrowthPanel(calchome=calchome, datapath=datapath),
        ("0", "1"),
    )

    if compute:
        board.allocate_panels()
        board.compute_data()
        board.plot_panels()

    return board
