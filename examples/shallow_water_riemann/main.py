r"""Module: examples.shallow_water_riemann.main

Description: 2D shallow-water four-quadrant Riemann problem with transmissive boundaries.
    Choose the algorithm on the command line (flux, limiter, gradient, time integrator, CFL,
    mesh, ...), for example
    ``python examples/shallow_water_riemann/main.py --flux hllc --integrator euler``.
    Run with ``--help`` for all options. Distributed run:
    ``mpiexec -n 4 python examples/shallow_water_riemann/main.py``.

Author: Hui Zhang
Date: 2025-07-11
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.case_setup import setup_case_shallow_water  # noqa: E402
from src.driver import Problem, run_problem  # noqa: E402
from src.shallow_water_equations import ShallowWaterEquations  # noqa: E402

PROBLEM = Problem(
    name="shallow_water_riemann",
    description="2D shallow-water four-quadrant Riemann problem",
    make_equation=lambda: ShallowWaterEquations(g=9.81),
    setup=setup_case_shallow_water,
    default_mesh="data/river_structured.msh",
    t_end=20.0,
)

if __name__ == "__main__":
    run_problem(PROBLEM)
