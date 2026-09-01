"""Semi-automatic goal-oriented adaptivity for nonstationary UFL problems."""

from .cli import add_common_arguments, config_from_arguments, parse_arguments
from .mesh import create_adaptive_l_shaped_mesh, create_adaptive_unit_square_mesh
from .options import NonstationaryDWRConfig
from .problem import GoalComponent, NonstationaryProblem
from .refine import refine_marked_interval_mesh, refine_marked_periodic_interval_mesh
from .solver import NonstationaryDWRAdaptiveSolver, NonstationaryDWRSolver

__all__ = [
    "NonstationaryDWRAdaptiveSolver",
    "NonstationaryDWRConfig",
    "NonstationaryDWRSolver",
    "NonstationaryProblem",
    "GoalComponent",
    "add_common_arguments",
    "config_from_arguments",
    "create_adaptive_l_shaped_mesh",
    "create_adaptive_unit_square_mesh",
    "parse_arguments",
    "refine_marked_interval_mesh",
    "refine_marked_periodic_interval_mesh",
]
