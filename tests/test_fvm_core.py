"""Regression tests for the FVM core (mesh connectivity, BCs, fluxes, residual).

Uses a tiny in-memory triangle mesh so it runs quickly even when another test
module has disabled numba's JIT.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.boundary import build_face_bcs, create_numba_bcs  # noqa: E402
from src.euler_equations import EulerEquations  # noqa: E402
from src.limiters import limiter_code  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.residual import compute_residual  # noqa: E402


def make_grid_mesh(n=4):
    """Unit square split into ``2 n^2`` triangles; boundary groups left/right/bottom/top."""
    xs = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(xs, xs)
    coords = np.zeros(((n + 1) ** 2, 3))
    coords[:, 0], coords[:, 1] = X.ravel(), Y.ravel()
    idx = lambda i, j: j * (n + 1) + i  # noqa: E731
    tris = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [[a, b, c], [a, c, d]]
    m = Mesh()
    m.node_tags = np.arange(1, len(coords) + 1, dtype=np.uint64)
    m.node_coords = coords
    m.elem_conn = np.array(tris, dtype=np.uint64) + 1
    m.elem_tags = np.arange(1, len(tris) + 1)
    m.nnode, m.nelem, m.dim = len(coords), len(tris), 2
    m.analyze_mesh()

    ii, jj = np.nonzero(m.cell_neighbors == -1)
    mid = m.face_midpoints[ii, jj]
    tags = np.select(
        [np.isclose(mid[:, 0], 0), np.isclose(mid[:, 0], 1), np.isclose(mid[:, 1], 0)],
        [1, 2, 3],
        4,
    )
    m.boundary_faces_nodes = np.sort(m.elem_faces[ii, jj], axis=1)
    m.boundary_faces_tags = tags
    m.boundary_tag_map = {"left": 1, "right": 2, "bottom": 3, "top": 4}
    return m


def cons(r, u, v, p, gamma=1.4):
    return np.array([r, r * u, r * v, p / (gamma - 1) + 0.5 * r * (u * u + v * v)])


def test_neighbors_are_symmetric_and_interior_faces_counted():
    m = make_grid_mesh()
    nb = m.cell_neighbors
    for i in range(m.nelem):
        for j in range(nb.shape[1]):
            if nb[i, j] != -1:
                assert i in nb[nb[i, j]]
    # 4 sides * n cells each
    assert (nb == -1).sum() == 4 * 4


def test_face_bcs_cover_every_boundary_face_regardless_of_node_order():
    m = make_grid_mesh()
    bcs = create_numba_bcs(
        {
            "left": {"type": "inlet", "velocity": [1.0, 0.0]},
            "right": {"type": "outlet", "pressure": 2.0},
            "bottom": {"type": "wall"},
            "top": {"type": "transmissive"},
        },
        m.boundary_tag_map,
    )
    bc_type, bc_value = build_face_bcs(m, bcs)
    free = m.cell_neighbors == -1
    assert (bc_type[~free] == 0).all()
    assert set(np.unique(bc_type[free])) == {1, 2, 3, 4}
    # left faces have x-midpoint 0 -> inlet with the requested velocity
    left = free & np.isclose(m.face_midpoints[..., 0], 0)
    assert (bc_type[left] == 1).all() and (bc_value[left][:, 0] == 1.0).all()


@pytest.mark.parametrize("flux", ["roe", "hllc"])
@pytest.mark.parametrize("limiter", ["barth_jespersen", "minmod"])
def test_uniform_flow_has_zero_residual(flux, limiter):
    m = make_grid_mesh()
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


def test_roe_pure_shear_is_upwinded_exactly():
    """A pure shear jump moving right must give the left physical flux."""
    eq = EulerEquations(1.4)
    n = np.array([1.0, 0.0])
    UL, UR = cons(1, 2, 0, 1), cons(1, 2, 1, 1)
    np.testing.assert_allclose(
        eq.roe_flux(UL, UR, n), eq._compute_flux(UL, n), atol=1e-12
    )


def test_boundary_condition_does_not_mutate_interior_state():
    eq = EulerEquations(1.4)
    U = cons(1.0, 0.5, 0.0, 1.0)
    before = U.copy()
    eq.apply_boundary_condition(U, np.array([1.0, 0.0]), 1, np.array([2.0, 0.0, 0.0]))
    eq.apply_boundary_condition(U, np.array([1.0, 0.0]), 2, np.array([5.0, 0.0, 0.0]))
    np.testing.assert_array_equal(U, before)


def test_unknown_limiter_raises():
    with pytest.raises(ValueError):
        limiter_code("nope")
