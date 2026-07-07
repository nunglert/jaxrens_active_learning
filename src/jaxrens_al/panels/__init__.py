"""Project-specific dashboard panels.

These subclass alviz's panel skeletons (``alviz.Entry`` or an abstract panel
like ``ParityPanel`` / ``HeatmapPanel``) and equip them with computation that
reads a remote_handler/jaxrens run's artifacts. Generic, reusable panels (the
``Extxyz*`` readers) belong in ``alviz.panels`` instead.
"""

# Importing each panel module above ran its ``register_panel`` call, so the
# panel registry is populated by importing this package. ``dft_config`` holds
# the specs for the generic Extxyz panels (no jaxrens class of their own).
from jaxrens_al.panels import dft_config  # noqa: F401
from jaxrens_al.panels.acceptance import (
    MoveAcceptancePanel,
    RESwapAcceptancePanel,
)
from jaxrens_al.panels.adaptation import (
    AdaptationAcceptancePanel,
    StepSizePanel,
)
from jaxrens_al.panels.environments import AtomicEnvsPanel
from jaxrens_al.panels.eos import EquationOfStatePanel
from jaxrens_al.panels.progress import DatasetGrowthPanel
from jaxrens_al.panels.quests import QuestsPCAPanel, ResampledGridPanel
from jaxrens_al.panels.sampling import NSSampledEnergyPanel
from jaxrens_al.panels.steinhardt import (
    SteinhardtDistributionPanel,
    SteinhardtExpectationHeatmap,
    SteinhardtHeatmapPanel,
    SteinhardtPanel,
)
from jaxrens_al.panels.thermodynamics import NSThermodynamicsHeatmap
from jaxrens_al.panels.traces import NSEnergyTracePanel
from jaxrens_al.panels.training import (
    MaceTrainingCurve,
    ModelParityPanel,
    NeuralILParityPanel,
    NeuralILTrainingCurve,
)

__all__ = [
    "AdaptationAcceptancePanel",
    "AtomicEnvsPanel",
    "DatasetGrowthPanel",
    "EquationOfStatePanel",
    "MaceTrainingCurve",
    "ModelParityPanel",
    "MoveAcceptancePanel",
    "NSEnergyTracePanel",
    "NSSampledEnergyPanel",
    "QuestsPCAPanel",
    "RESwapAcceptancePanel",
    "ResampledGridPanel",
    "SteinhardtDistributionPanel",
    "SteinhardtExpectationHeatmap",
    "SteinhardtHeatmapPanel",
    "SteinhardtPanel",
    "NeuralILParityPanel",
    "NeuralILTrainingCurve",
    "NSThermodynamicsHeatmap",
    "StepSizePanel",
]
