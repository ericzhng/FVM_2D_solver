r"""Module: examples.euler_riemann.main

Description: 2D Euler four-quadrant Riemann problem with transmissive boundaries. Choose the
    algorithm on the command line (flux, limiter, gradient, time integrator, CFL, mesh, ...),
    for example ``python examples/euler_riemann/main.py --flux hllc --limiter venkatakrishnan``.
    Run with ``--help`` for all options. Distributed run:
    ``mpiexec -n 4 python examples/euler_riemann/main.py``.

Author: Hui Zhang
Date: 2025-07-11
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.case_setup import setup_case_euler  # noqa: E402
from src.driver import Problem, run_problem  # noqa: E402
from src.euler_equations import EulerEquations  # noqa: E402

PROBLEM = Problem(
    name="euler_riemann",
    description="2D Euler four-quadrant Riemann problem",
    make_equation=lambda: EulerEquations(gamma=1.4),
    setup=setup_case_euler,
    default_mesh="data/euler_mesh.msh",
    t_end=0.25,
)

if __name__ == "__main__":
    run_problem(PROBLEM)
