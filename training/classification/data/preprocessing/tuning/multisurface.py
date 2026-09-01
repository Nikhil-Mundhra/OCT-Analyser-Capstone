"""
data/preprocessing/tuning/multisurface.py

Re-export façade for JointMultiSurfaceOptimizer and BSplineRegularizer.
Delegates to core domain module data.preprocessing.multisurface.
"""

from data.preprocessing.multisurface import (
    BSplineRegularizer,
    JointMultiSurfaceOptimizer,
    MultiSurfaceConfig,
)

__all__ = [
    "BSplineRegularizer",
    "JointMultiSurfaceOptimizer",
    "MultiSurfaceConfig",
]
