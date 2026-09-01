"""Reusable space--time DWR bubble-projection adaptive solver.

The public entry point is :class:`BubbleProjectionAdaptiveSolver`.  A new
PDE normally only needs a small :class:`TransientDWRProblem` implementation;
the adaptive loop, bubble/cone localisation, Dörfler marking, refinement, and
ParaView output stay unchanged.
"""

from .parameters import BubbleProjectionOptions
from .problem import (
    BBMFiniteIntervalSolitaryWaveProblem,
    BBMSolitaryWaveProblem,
    BBMTravellingWaveProblem,
    HeatEquationProblem,
    TransientDWRProblem,
)
from .solver import BubbleProjectionAdaptiveSolver

__all__ = [
    "BubbleProjectionAdaptiveSolver",
    "BubbleProjectionOptions",
    "BBMFiniteIntervalSolitaryWaveProblem",
    "BBMSolitaryWaveProblem",
    "BBMTravellingWaveProblem",
    "HeatEquationProblem",
    "TransientDWRProblem",
]
