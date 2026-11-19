r"""Module: tests.unit.test_mesh_geometry

Description: Unit tests for mesh geometry on the gmsh-free unit-square mesh: areas, closed cells,
    face lengths, boundary groups and quality.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np
import pytest

from src.mesh import Mesh


@pytest.mark.parametrize("n", [1, 3, 8])
def test_unit_square_counts_and_total_area(n):
    mesh = Mesh.unit_square(n)
    assert mesh.nelem == 2 * n * n
    assert mesh.dim == 2
    assert mesh.cell_volumes.sum() == pytest.approx(1.0)
    assert (mesh.cell_volumes > 0).all()


def test_every_cell_is_closed(square_mesh):
    """Sum of (outward normal x face length) around a cell vanishes (divergence theorem)."""
    m = square_mesh
    closure = (m.face_normals[..., :2] * m.face_areas[..., None]).sum(axis=1)
    np.testing.assert_allclose(closure, 0.0, atol=1e-12)


def test_normals_are_unit_and_point_outward(square_mesh):
    m = square_mesh
    n = m.face_normals[..., :2]
    np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0)
    r = (m.face_midpoints - m.cell_centroids[:, None, :])[..., :2]
    assert ((r * n).sum(-1) > 0).all()


def test_boundary_length_is_the_perimeter(square_mesh):
    boundary = square_mesh.cell_neighbors == -1
    assert square_mesh.face_areas[boundary].sum() == pytest.approx(4.0)


def test_each_boundary_group_is_one_unit_long(square_mesh):
    m = square_mesh
    boundary = m.cell_neighbors == -1
    x, y = m.face_midpoints[..., 0], m.face_midpoints[..., 1]
    sides = {"left": x, "right": x - 1.0, "bottom": y, "top": y - 1.0}
    for side, offset in sides.items():
        assert side in m.boundary_tag_map
        assert m.face_areas[boundary & np.isclose(offset, 0.0)].sum() == pytest.approx(1.0)


def test_mesh_quality_is_finite_and_at_least_one(square_mesh):
    q = square_mesh.get_mesh_quality()
    assert np.isfinite(q).all() and (q >= 1.0).all()
