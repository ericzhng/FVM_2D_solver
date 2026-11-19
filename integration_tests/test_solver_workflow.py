r"""Module: integration_tests.test_solver_workflow

Description: End-to-end test of the serial solver loop: with ``save_every`` the stored history keeps
    the total simulated time and the same final state as storing every step.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.euler_equations import EulerEquations  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver import solve  # noqa: E402


def cons(r, u, v, p, gamma=1.4):
    return np.array([r, r * u, r * v, p / (gamma - 1) + 0.5 * r * (u * u + v * v)])


def test_solve_save_every_keeps_total_time_and_final_state():
    m = Mesh.unit_square(4)
    eq = EulerEquations(1.4)
    U0 = np.tile(cons(1.0, 0.3, 0.0, 1.0), (m.nelem, 1))
    bc = {k: {"type": "transmissive"} for k in m.boundary_tag_map}
    full, dts_full = solve(m, U0, bc, eq, t_end=0.5, save_every=1)
    sparse, dts_sparse = solve(m, U0, bc, eq, t_end=0.5, save_every=3)
    assert len(sparse) < len(full)
    assert sum(dts_sparse) == pytest.approx(0.5) and sum(dts_full) == pytest.approx(0.5)
    np.testing.assert_allclose(sparse[-1], full[-1])
