"""Project-specific dashboard panels.

These subclass alviz's panel skeletons (``alviz.Entry`` or an abstract panel
like ``ParityPanel`` / ``HeatmapPanel``) and equip them with computation that
reads a remote_handler/jaxrens run's artifacts. Generic, reusable panels (the
``Extxyz*`` readers) belong in ``alviz.panels`` instead.
"""

from jaxrens_al.panels.acceptance import (
    MoveAcceptancePanel,
    RESwapAcceptancePanel,
)
from jaxrens_al.panels.environments import AtomicEnvsPanel
from jaxrens_al.panels.eos import EquationOfStatePanel
from jaxrens_al.panels.progress import DatasetGrowthPanel
from jaxrens_al.panels.quests import QuestsPCAPanel, ResampledGridPanel
from jaxrens_al.panels.sampling import NSSampledEnergyPanel
from jaxrens_al.panels.thermodynamics import NSThermodynamicsHeatmap
from jaxrens_al.panels.training import (
    NeuralILParityPanel,
    NeuralILTrainingCurve,
)

__all__ = [
    "AtomicEnvsPanel",
    "DatasetGrowthPanel",
    "EquationOfStatePanel",
    "MoveAcceptancePanel",
    "NSSampledEnergyPanel",
    "QuestsPCAPanel",
    "RESwapAcceptancePanel",
    "ResampledGridPanel",
    "NeuralILParityPanel",
    "NeuralILTrainingCurve",
    "NSThermodynamicsHeatmap",
]
