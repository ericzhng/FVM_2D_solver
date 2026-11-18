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
from src.reconstruction import compute_gradients_gaussian, compute_gradients_lsq  # noqa: E402
from src.residual import compute_residual  # noqa: E402
from src.shallow_water_equations import ShallowWaterEquations  # noqa: E402
from src.solver import solve  # noqa: E402


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
@pytest.mark.parametrize("limiter", ["barth_jespersen", "venkatakrishnan"])
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
    np.testing.assert_allclose(eq.roe_flux(UL, UR, n), eq._compute_flux(UL, n), atol=1e-12)


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


def _transmissive_bcs(m):
    bcs = create_numba_bcs(
        {k: {"type": "transmissive"} for k in m.boundary_tag_map}, m.boundary_tag_map
    )
    return build_face_bcs(m, bcs)


def _skewed_mesh(n=8, seed=0):
    m = make_grid_mesh(n)
    rng = np.random.default_rng(seed)
    c = m.node_coords
    inner = (c[:, 0] > 1e-9) & (c[:, 0] < 1 - 1e-9) & (c[:, 1] > 1e-9) & (c[:, 1] < 1 - 1e-9)
    c[inner, :2] += rng.uniform(-0.3 / n, 0.3 / n, (inner.sum(), 2))
    m.analyze_mesh()
    return m


def test_lsq_gradient_exact_for_linear_field_on_skewed_mesh():
    m = _skewed_mesh()
    geo = m.fvm_geometry()
    U = (2 * geo.centroids[:, 0] - 3 * geo.centroids[:, 1] + 1)[:, None]
    bt, bv = _transmissive_bcs(m)
    g = compute_gradients_lsq(geo, U, EulerEquations(1.4), bt, bv)
    interior = (geo.neighbors != -1).all(axis=1)
    np.testing.assert_allclose(
        g[interior, 0, :], np.broadcast_to([2.0, -3.0], (interior.sum(), 2)), atol=1e-10
    )
    # Green-Gauss is not exact here; LSQ is the reason it is the default.
    gg = compute_gradients_gaussian(geo, U, 0.0)
    assert np.abs(gg[interior, 0, :] - [2.0, -3.0]).max() > 1e-3


def test_face_table_is_consistent():
    m = make_grid_mesh()
    geo = m.fvm_geometry()
    nf_u = geo.owner_cell.size
    n_boundary = (geo.neighbors == -1).sum()
    assert 2 * nf_u - n_boundary == geo.neighbors.size  # interior faces counted twice
    # every unique face has exactly one owner entry; interior faces one non-owner
    owners = np.zeros(nf_u, int)
    others = np.zeros(nf_u, int)
    np.add.at(owners, geo.face_id[geo.face_sign == 1], 1)
    np.add.at(others, geo.face_id[geo.face_sign == -1], 1)
    assert (owners == 1).all() and (others <= 1).all()
    # the owner's stored location matches the face_id table
    assert (geo.face_id[geo.owner_cell, geo.owner_local] == np.arange(nf_u)).all()


def test_wall_domain_conserves_mass_and_energy():
    """Sum of V*R over a closed (wall) domain telescopes to ~0 for mass/energy."""
    m = make_grid_mesh(6)
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


def test_euler_inlet_outlet_ghost_states_are_physical():
    eq = EulerEquations(1.4)
    U = cons(1.5, 0.5, 0.0, 2.0)
    n = np.array([1.0, 0.0])
    ghost = eq.apply_boundary_condition(U, n, 1, np.array([2.0, 0.5, 0.0]))
    rho, u, v, p = eq._cons_to_prim(ghost)
    assert (rho, u, v, p) == pytest.approx((1.5, 2.0, 0.5, 2.0))
    ghost = eq.apply_boundary_condition(U, n, 2, np.array([0.7, 0.0, 0.0]))
    rho, u, v, p = eq._cons_to_prim(ghost)
    assert (rho, u, v, p) == pytest.approx((1.5, 0.5, 0.0, 0.7))


def test_shallow_water_inlet_outlet_ghost_states():
    sw = ShallowWaterEquations(9.81)
    U = np.array([2.0, 1.0, 0.0])  # h=2, u=0.5
    n = np.array([1.0, 0.0])
    h, u, v = sw._cons_to_prim(sw.apply_boundary_condition(U, n, 1, np.array([1.5, 0.0, 0.0])))
    assert (h, u, v) == pytest.approx((2.0, 1.5, 0.0))
    h, u, v = sw._cons_to_prim(sw.apply_boundary_condition(U, n, 2, np.array([0.0, 0.8, 0.0])))
    assert (h, u, v) == pytest.approx((0.8, 0.5, 0.0))
    np.testing.assert_array_equal(U, [2.0, 1.0, 0.0])  # not mutated


def test_area_centroid_of_trapezoid():
    m = Mesh()
    m.node_tags = np.arange(1, 5, dtype=np.uint64)
    m.node_coords = np.array([[0, 0, 0], [4, 0, 0], [3, 2, 0], [1, 2, 0]], float)
    m.elem_conn = np.array([[1, 2, 3, 4]], dtype=np.uint64)
    m.elem_tags = np.array([1])
    m.nnode, m.nelem, m.dim = 4, 1, 2
    m.analyze_mesh()
    np.testing.assert_allclose(m.cell_centroids[0, :2], [2.0, 8.0 / 9.0])


def test_solve_save_every_keeps_total_time_and_final_state():
    m = make_grid_mesh(4)
    eq = EulerEquations(1.4)
    U0 = np.tile(cons(1.0, 0.3, 0.0, 1.0), (m.nelem, 1))
    bc = {k: {"type": "transmissive"} for k in m.boundary_tag_map}
    full, dts_full = solve(m, U0, bc, eq, t_end=0.5, save_every=1)
    sparse, dts_sparse = solve(m, U0, bc, eq, t_end=0.5, save_every=3)
    assert len(sparse) < len(full)
    assert sum(dts_sparse) == pytest.approx(0.5) and sum(dts_full) == pytest.approx(0.5)
    np.testing.assert_allclose(sparse[-1], full[-1])


def test_near_wall_gradient_honours_reflection_ghost():
    """A wall-consistent linear field must have exact gradients in wall cells.

    The field is even in y for rho, rho*u, E and odd for rho*v (as a reflecting
    wall at y=0 implies). A zero-gradient virtual neighbour gets d(rho*v)/dy
    wrong by 50%; the reflected ghost state recovers it exactly.
    """
    m = make_grid_mesh(8)
    geo = m.fvm_geometry()
    x, y = geo.centroids.T
    U = np.stack([1 + 0.3 * x, 0.5 + 0.2 * x, 0.4 * y, 3 + 0.1 * x], axis=1)
    exact = np.array([[0.3, 0.0], [0.2, 0.0], [0.0, 0.4], [0.1, 0.0]])
    bcs = create_numba_bcs(
        {
            "bottom": {"type": "wall"},
            "left": {"type": "transmissive"},
            "right": {"type": "transmissive"},
            "top": {"type": "transmissive"},
        },
        m.boundary_tag_map,
    )
    bt, bv = build_face_bcs(m, bcs)
    free = geo.neighbors == -1
    wall_only = (free.sum(axis=1) > 0) & ((free & (bt != 3)).sum(axis=1) == 0)
    assert wall_only.sum() > 0
    g = compute_gradients_lsq(geo, U, EulerEquations(1.4), bt, bv)
    for i in np.nonzero(wall_only)[0]:
        np.testing.assert_allclose(g[i], exact, atol=1e-10)


def test_limiter_with_wall_ghosts_stays_in_range():
    """Limiter values stay in [0, 1] when bounds include reflected wall ghosts."""
    from src.reconstruction import compute_limiters

    m = make_grid_mesh(8)
    geo = m.fvm_geometry()
    x, y = geo.centroids.T
    U = np.stack([1 + 0.3 * x, 0.5 + 0.2 * x, 0.4 * y, 3 + 0.1 * x], axis=1)
    bcs = create_numba_bcs(
        {k: {"type": "wall"} for k in m.boundary_tag_map}, m.boundary_tag_map
    )
    bt, bv = build_face_bcs(m, bcs)
    eq = EulerEquations(1.4)
    g = compute_gradients_lsq(geo, U, eq, bt, bv)
    phi = compute_limiters(geo, U, g, 0, 1.0, eq, bt, bv)
    assert np.isfinite(phi).all() and phi.min() >= 0.0 and phi.max() <= 1.0
