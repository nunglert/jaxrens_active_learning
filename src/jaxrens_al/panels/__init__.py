"""Project-specific dashboard panels.

These subclass alviz's panel skeletons (``alviz.Entry`` or an abstract panel
like ``ParityPanel`` / ``HeatmapPanel``) and equip them with computation that
reads a remote_handler/jaxrens run's artifacts. Generic, reusable panels (the
``Extxyz*`` readers) belong in ``alviz.panels`` instead.
"""

from jaxrens_al.panels.progress import DatasetGrowthPanel
from jaxrens_al.panels.sampling import NSSampledEnergyPanel
from jaxrens_al.panels.thermodynamics import NSThermodynamicsHeatmap
from jaxrens_al.panels.training import (
    NeuralILParityPanel,
    NeuralILTrainingCurve,
)

__all__ = [
    "DatasetGrowthPanel",
    "NSSampledEnergyPanel",
    "NeuralILParityPanel",
    "NeuralILTrainingCurve",
    "NSThermodynamicsHeatmap",
]
