r"""Module: tests.unit.test_boundary

Description: Unit tests for boundary-condition conversion: the dictionary to structured array
    (types and parameters) and the per-face resolution on a mesh (inlet, wall default).

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np

from src.boundary import BC_TYPE_MAP, build_face_bcs, create_numba_bcs
from src.mesh import Mesh


def test_create_numba_bcs_maps_types_and_parameters():
    bc = {
        "a": {"type": "inlet", "velocity": [1.5, -2.0]},
        "b": {"type": "outlet", "pressure": 3.0, "water_depth": 0.7},
        "c": {"type": "wall", "slip": True},
        "d": {"type": "transmissive"},
    }
    arr = create_numba_bcs(bc, {"a": 1, "b": 2, "c": 3, "d": 4})
    assert list(arr["tag_id"]) == [1, 2, 3, 4]
    expected = [BC_TYPE_MAP[k] for k in ("inlet", "outlet", "wall", "transmissive")]
    assert list(arr["bc_type"]) == expected
    np.testing.assert_allclose(arr["vars"][0], [1.5, -2.0, 0.0])
    np.testing.assert_allclose(arr["vars"][1], [3.0, 0.7, 0.0])
    assert arr["vars"][2, 0] == 1.0  # slip wall
    assert not arr["vars"][3].any()


def test_unknown_group_gets_tag_minus_one_and_missing_type_defaults_to_wall():
    arr = create_numba_bcs({"nowhere": {}}, {"left": 1})
    assert arr["tag_id"][0] == -1
    assert arr["bc_type"][0] == BC_TYPE_MAP["wall"]


def test_build_face_bcs_resolves_groups_and_defaults_to_wall():
    mesh = Mesh.unit_square(4)
    bc = {"left": {"type": "inlet", "velocity": [1.0, 0.0]}, "right": {"type": "transmissive"}}
    bc_type, bc_value = build_face_bcs(mesh, create_numba_bcs(bc, mesh.boundary_tag_map))

    boundary = mesh.cell_neighbors == -1
    assert (bc_type[~boundary] == 0).all()  # interior faces carry no BC
    mid_x = mesh.face_midpoints[..., 0]
    left = boundary & np.isclose(mid_x, 0.0)
    right = boundary & np.isclose(mid_x, 1.0)
    rest = boundary & ~left & ~right
    assert (bc_type[left] == BC_TYPE_MAP["inlet"]).all()
    np.testing.assert_allclose(bc_value[left][:, 0], 1.0)
    assert (bc_type[right] == BC_TYPE_MAP["transmissive"]).all()
    assert (bc_type[rest] == BC_TYPE_MAP["wall"]).all()  # not listed -> wall
