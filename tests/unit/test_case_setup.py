r"""Module: tests.unit.test_case_setup

Description: Unit tests for the initial conditions of the demo Riemann problems: shapes,
    quadrant values, physical admissibility and boundary dictionaries.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np
import pytest

from src.case_setup import setup_case_euler, setup_case_shallow_water
from src.mesh import Mesh


def quadrant_cells(mesh):
    """One cell index in each quadrant (top-right, top-left, bottom-left, bottom-right)."""
    x, y = mesh.cell_centroids[:, 0], mesh.cell_centroids[:, 1]
    mid_x, mid_y = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    masks = (
        (x >= mid_x) & (y >= mid_y),
        (x < mid_x) & (y >= mid_y),
        (x < mid_x) & (y < mid_y),
        (x >= mid_x) & (y < mid_y),
    )
    return [int(np.flatnonzero(m)[0]) for m in masks]


def test_euler_case_has_four_quadrants_with_expected_primitives():
    mesh = Mesh.unit_square(6)
    U, bc = setup_case_euler(mesh)
    assert U.shape == (mesh.nelem, 4)
    rho = U[:, 0]
    u, v = U[:, 1] / rho, U[:, 2] / rho
    p = 0.4 * (U[:, 3] - 0.5 * rho * (u**2 + v**2))
    np.testing.assert_allclose(p, 1.0)
    expected = [(1.0, -0.75, -0.5), (2.0, -0.75, 0.5), (1.0, 0.75, 0.5), (3.0, 0.75, -0.5)]
    for cell, (r, ux, vy) in zip(quadrant_cells(mesh), expected):
        assert (rho[cell], u[cell], v[cell]) == pytest.approx((r, ux, vy))
    assert set(bc) == {"top", "bottom", "left", "right"}
    assert all(v["type"] == "transmissive" for v in bc.values())


def test_shallow_water_case_is_admissible_and_momentum_is_depth_times_velocity():
    mesh = Mesh.unit_square(6)
    U, bc = setup_case_shallow_water(mesh)
    assert U.shape == (mesh.nelem, 3)
    assert (U[:, 0] > 0).all()
    tr, tl, bl, br = quadrant_cells(mesh)
    np.testing.assert_allclose(U[tr], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(U[tl], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(U[bl], [1.5, -0.75, 0.0])
    np.testing.assert_allclose(U[br], [0.5, 0.0, 0.25])
    assert set(bc) == {"top", "bottom", "left", "right"}


def test_shallow_water_split_follows_the_domain_not_a_fixed_coordinate():
    """Scaling the mesh by 100 must give the same quadrant layout."""
    small, big = Mesh.unit_square(6), Mesh.unit_square(6)
    big.cell_centroids = big.cell_centroids * 100.0
    np.testing.assert_array_equal(
        setup_case_shallow_water(small)[0], setup_case_shallow_water(big)[0]
    )
