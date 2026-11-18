import numpy as np

# --- Sample BC Dictionary-based Definition ---
# bc_dict: Dict[str, Dict[str, Union[str, float, bool]]] = {
#     "inlet_channel": {"type": "inlet", "velocity": [1.5, 0.0]},
#     "outlet_weir": {"type": "outlet", "pressure": 0.0},
#     "downstream_boundary": {"type": "outlet", "water_depth": 0.5},
#     "solid_walls": {"type": "wall", "slip": False},
#     "open_boundary": {"type": "transmissive"},
# }


# Mapping for Boundary Condition types
BC_TYPE_MAP = {
    "inlet": 1,
    "outlet": 2,
    "wall": 3,
    "transmissive": 4,
}


def create_numba_bcs(
    bc_dict,
    boundary_tag_map,
):
    """
    Converts the human-readable dictionary into a NumPy structured array
    that can be passed to Numba.
    """
    # Define the structure (dtype) of our array.
    # We use generic 'param' fields. The meaning of each param depends on the 'bc_type'.
    # For example, for an inlet, param1=vx, param2=vy, param3=vz.
    # For an outlet, param1=pressure, param2=water_depth.
    # For a wall, param1=slip (0.0 for False, 1.0 for True).
    bc_dtype = np.dtype(
        [
            ("tag_id", np.int32),
            ("bc_type", np.int32),
            ("vars", (np.float64, 3)),
        ]
    )

    bc_list = []
    for tag_name, props in bc_dict.items():
        tag_id = boundary_tag_map.get(tag_name, -1)

        bc_type_str = props.get("type", "wall")
        bc_type_id = BC_TYPE_MAP.get(bc_type_str, 3)

        # Initialize params to a default (e.g., NaN or 0)
        vars_val = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Populate params based on BC type
        if bc_type_id == 1:  # inlet
            velocity = np.asarray(props.get("velocity", [0.0, 0.0]), dtype=np.float64)
            vars_val[: velocity.size] = velocity  # accepts 2D or 3D velocities
        elif bc_type_id == 2:  # outlet
            vars_val[0] = props.get("pressure", 0.0)
            # h = props.get("water_depth", 0.0)
        elif bc_type_id == 3:  # Wall
            vars_val[0] = 1.0 if props.get("slip", False) else 0.0

        # For transmissive, params can remain 0 as it often needs no values.
        bc_list.append((tag_id, bc_type_id, vars_val))

    return np.array(bc_list, dtype=bc_dtype)


def build_face_bcs(mesh, bcs_array):
    """Resolve the boundary condition of every cell face once, before time stepping.

    Replaces the per-face linear searches the flux kernel used to perform on every
    residual evaluation. Boundary faces whose physical group has no entry in
    ``bcs_array`` (or no group at all) default to a wall.

    Args:
        mesh: Analysed :class:`src.mesh.Mesh`.
        bcs_array: Output of :func:`create_numba_bcs`.

    Returns:
        ``(bc_type, bc_value)`` with shapes ``(nelem, nfaces)`` (int32, ``0`` on
        interior faces) and ``(nelem, nfaces, 3)`` (float64).
    """
    nelem, nfaces = mesh.cell_neighbors.shape
    bc_type = np.zeros((nelem, nfaces), dtype=np.int32)
    bc_value = np.zeros((nelem, nfaces, 3), dtype=np.float64)

    # Boundary faces are stored with sorted node tags; use the same key.
    face_tag = {
        tuple(nodes): tag
        for nodes, tag in zip(
            mesh.boundary_faces_nodes.tolist(), mesh.boundary_faces_tags.tolist()
        )
    }
    bc_by_tag = {int(row["tag_id"]): row for row in bcs_array}

    ii, jj = np.nonzero(mesh.cell_neighbors == -1)
    sorted_faces = np.sort(mesh.elem_faces[ii, jj], axis=1).tolist()
    for i, j, key in zip(ii, jj, sorted_faces):
        row = bc_by_tag.get(face_tag.get(tuple(key)))
        if row is None:
            bc_type[i, j] = BC_TYPE_MAP["wall"]
        else:
            bc_type[i, j] = row["bc_type"]
            bc_value[i, j] = row["vars"]
    return bc_type, bc_value
