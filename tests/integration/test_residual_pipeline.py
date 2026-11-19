r"""Module: tests.integration.test_residual_pipeline

Description: Integration tests of the spatial residual pipeline (gradient, limiter, Riemann flux,
    per-cell gather) on small in-memory meshes: free-stream preservation for every flux and
    limiter, and global conservation of mass and energy in a closed wall domain.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np
import pytest

from src.boundary import build_face_bcs, create_numba_bcs
from src.euler_equations import EulerEquations
from src.mesh import Mesh
from src.residual import compute_residual


def cons(r, u, v, p, gamma=1.4):
    return np.array([r, r * u, r * v, p / (gamma - 1) + 0.5 * r * (u * u + v * v)])


@pytest.mark.parametrize("flux", ["roe", "hllc"])
@pytest.mark.parametrize("limiter", ["barth_jespersen", "venkatakrishnan"])
def test_uniform_flow_has_zero_residual(flux, limiter):
    m = Mesh.unit_square()
    eq = EulerEquations(1.4)
    U = np.tile(cons(1.0, 0.3, -0.2, 1.0), (m.nelem, 1))
    bcs = create_numba_bcs(
        {k: {"type": "transmissive"} for k in m.boundary_tag_map}, m.boundary_tag_map
    )
    bc_type, bc_value = build_face_bcs(m, bcs)
    R = compute_residual(
        U, m.fvm_geometry(), eq, bc_type, bc_value, limiter, flux, over_relaxation=1.0
    )
    assert np.abs(R).max() < 1e-12


def test_wall_domain_conserves_mass_and_energy():
    """Sum of V*R over a closed (wall) domain telescopes to ~0 for mass/energy."""
    m = Mesh.unit_square(6)
    eq = EulerEquations(1.4)
    x, y = m.fvm_geometry().centroids.T
    U = np.stack(
        [
            1.0 + 0.3 * np.sin(4 * x) * np.cos(3 * y),
            0.2 * np.ones_like(x),
            -0.1 * np.ones_like(x),
            3.0 + 0.2 * np.cos(5 * x * y),
        ],
        axis=1,
    )
    bcs = create_numba_bcs({k: {"type": "wall"} for k in m.boundary_tag_map}, m.boundary_tag_map)
    bc_type, bc_value = build_face_bcs(m, bcs)
    R = compute_residual(U, m.fvm_geometry(), eq, bc_type, bc_value, flux="hllc")
    total = (m.cell_volumes[:, None] * R).sum(axis=0)
    assert abs(total[0]) < 1e-12 and abs(total[3]) < 1e-12
