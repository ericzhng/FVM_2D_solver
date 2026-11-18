from typing import NamedTuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class FVMGeometry(NamedTuple):
    """Contiguous 2D arrays the numba kernels need (all indexed ``[cell, face]``).

    Attributes:
        neighbors: Neighbour cell index, ``-1`` on boundary faces.
        normals: Outward unit normals, ``(nelem, nfaces, 2)``.
        areas: Face lengths.
        midpoints: Face midpoints, ``(nelem, nfaces, 2)``.
        face_r: ``midpoint - cell centroid``, ``(nelem, nfaces, 2)``.
        centroids: Cell centroids, ``(nelem, 2)``.
        volumes: Cell areas.
        weights: Linear interpolation weight of the *own* cell at the face.
        nonorth: Scalar non-orthogonal gradient-correction coefficient.
        lsq: Least-squares gradient coefficients, ``(nelem, nfaces, 2)``:
            ``grad_i = sum_j lsq[i, j] * (U_nb(j) - U_i)`` over interior faces.
        face_id: Unique-face index of every ``[cell, face]``.
        face_sign: ``+1`` if the cell owns the face, ``-1`` otherwise.
        owner_cell: Owning cell of each unique face.
        owner_local: Local face index (within the owner) of each unique face.
    """

    neighbors: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    midpoints: np.ndarray
    face_r: np.ndarray
    centroids: np.ndarray
    volumes: np.ndarray
    weights: np.ndarray
    nonorth: np.ndarray
    lsq: np.ndarray
    face_id: np.ndarray
    face_sign: np.ndarray
    owner_cell: np.ndarray
    owner_local: np.ndarray


class Mesh:
    """
    A class to represent a computational mesh for 1D, 2D, or 3D simulations,
    providing geometric and connectivity information for Finite Volume Methods.
    """

    def __init__(self):
        """
        Initializes the Mesh object with empty attributes.
        """
        self.dim = 0
        self.nelem = 0
        self.nnode = 0

        # raw data
        self.node_tags = np.array([])
        self.elem_tags = np.array([])
        self.node_coords = np.array([])
        self.elem_conn = np.array([])

        # derived
        self.cell_volumes = np.array([])
        self.cell_centroids = np.array([])
        self.face_normals = np.array([])
        self.face_tangentials = np.array([])
        self.face_areas = np.array([])
        self.boundary_faces_nodes = np.array([])
        self.boundary_faces_tags = np.array([])
        self.boundary_tag_map = {}
        self.cell_neighbors = np.array([])
        self.elem_faces = np.array([])

    def read_mesh(self, mesh_file):
        """
        Reads the mesh file using gmsh, determines the highest dimension,
        and extracts node and element information.

        Args:
            mesh_file (str): Path to the mesh file (e.g., .msh).
        """
        import gmsh  # lazy: geometry code stays usable without gmsh installed

        gmsh.initialize()
        try:
            gmsh.open(mesh_file)
            self._extract_from_gmsh(gmsh)
        finally:
            gmsh.finalize()

    def _extract_from_gmsh(self, gmsh):
        """Pull nodes, main-dimension elements and boundary groups from gmsh."""
        self.node_tags, self.node_coords, _ = gmsh.model.mesh.getNodes()
        self.node_coords = np.array(self.node_coords).reshape(-1, 3)
        self.nnode = len(self.node_tags)

        elem_types, elem_tags, node_connectivity = gmsh.model.mesh.getElements()
        max_dim = 0
        main_elem_type_idx = -1
        for i, e_type in enumerate(elem_types):
            _, dim, _, _, _, _ = gmsh.model.mesh.getElementProperties(e_type)
            if dim > max_dim:
                max_dim = dim
                main_elem_type_idx = i

        self.dim = max_dim

        if main_elem_type_idx != -1:
            main_elem_type = elem_types[main_elem_type_idx]
            self.elem_tags = elem_tags[main_elem_type_idx]

            _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(
                main_elem_type
            )
            self.elem_conn = np.array(node_connectivity[main_elem_type_idx]).reshape(
                -1, num_nodes
            )
            self.nelem = len(self.elem_tags)

        self._get_boundary_info(gmsh)

    def analyze_mesh(self):
        """
        Analyzes the mesh to compute all geometric and connectivity properties
        required for a Finite Volume Method solver.
        """
        if self.nelem == 0:
            raise RuntimeError("Mesh data has not been read. Call read_mesh() first.")

        # Create a mapping from node tags to their 0-based index.
        max_tag = np.max(self.node_tags)
        self.node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
        self.node_tag_map[self.node_tags] = np.arange(self.nnode, dtype=np.int32)

        self._compute_cell_centroids()
        self._compute_mesh_properties()
        self._compute_cell_volumes()
        if self.dim == 2:
            self._compute_reconstruction_geometry()
            self._compute_lsq_coefficients()

    def _get_boundary_info(self, gmsh):
        """
        Extracts boundary faces and their corresponding physical group tags.
        """
        boundary_dim = self.dim - 1
        if boundary_dim < 0:
            return

        all_boundary_faces_nodes = []
        all_boundary_faces_tags = []

        physical_groups = gmsh.model.getPhysicalGroups(dim=boundary_dim)
        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            self.boundary_tag_map[name] = tag
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for entity in entities:
                b_elem_types, b_elem_tags, b_node_tags = gmsh.model.mesh.getElements(
                    dim, entity
                )
                for i, elem_type in enumerate(b_elem_types):
                    _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(
                        elem_type
                    )

                    faces_nodes = np.array(b_node_tags[i]).reshape(-1, num_nodes)
                    # Sort nodes within each face to ensure consistent representation
                    faces_nodes.sort(axis=1)
                    all_boundary_faces_nodes.append(faces_nodes)
                    all_boundary_faces_tags.extend([tag] * len(faces_nodes))

        if all_boundary_faces_nodes:
            self.boundary_faces_nodes = np.vstack(all_boundary_faces_nodes)
            self.boundary_faces_tags = np.array(all_boundary_faces_tags)

    def _compute_cell_centroids(self):
        """Computes the centroid of each element using vectorized operations."""
        # Use the map to convert element connectivity from tags to indices.
        elem_node_indices = self.node_tag_map[self.elem_conn]

        # Gather all node coordinates for all elements.
        elem_nodes_coords = self.node_coords[elem_node_indices]

        if self.dim == 2:
            # Area centroid of the polygon (equals the vertex mean for triangles).
            x, y = elem_nodes_coords[:, :, 0], elem_nodes_coords[:, :, 1]
            xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
            cross = x * yn - xn * y
            area6 = 3.0 * cross.sum(axis=1)
            good = np.abs(area6) > 1e-30
            safe = np.where(good, area6, 1.0)
            cen = np.mean(elem_nodes_coords, axis=1)
            cen[:, 0] = np.where(good, ((x + xn) * cross).sum(axis=1) / safe, cen[:, 0])
            cen[:, 1] = np.where(good, ((y + yn) * cross).sum(axis=1) / safe, cen[:, 1])
            self.cell_centroids = cen
        else:
            self.cell_centroids = np.mean(elem_nodes_coords, axis=1)

    def _compute_cell_volumes(self):
        """Computes the volume/area of each element."""
        if self.dim == 1:
            elem_node_indices = self.node_tag_map[self.elem_conn]
            elem_nodes_coords = self.node_coords[elem_node_indices]
            self.cell_volumes = np.linalg.norm(
                elem_nodes_coords[:, 1, :] - elem_nodes_coords[:, 0, :], axis=1
            )
        elif self.dim == 2:
            elem_node_indices = self.node_tag_map[self.elem_conn]
            elem_nodes_coords = self.node_coords[elem_node_indices]
            x = elem_nodes_coords[:, :, 0]
            y = elem_nodes_coords[:, :, 1]
            self.cell_volumes = 0.5 * np.abs(
                np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
            )
        elif self.dim == 3:
            self.cell_volumes = np.zeros(self.nelem)
            for i in range(self.nelem):
                volume = 0.0
                for j, face_nodes in enumerate(self.elem_faces[i]):
                    node_indices_face = np.array(
                        [self.node_tag_map[tag] for tag in face_nodes]
                    )
                    face_midpoint = np.mean(self.node_coords[node_indices_face], axis=0)
                    face_normal = self.face_normals[i, j]
                    face_area = self.face_areas[i, j]
                    volume += np.dot(face_midpoint, face_normal) * face_area
                self.cell_volumes[i] = volume / 3.0

    def _compute_mesh_properties(self):
        """
        Computes cell neighbors and face properties (normals, tangentials, areas) using vectorized operations.
        """
        face_definitions = {
            "tet": [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],
            "hex": [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ],
            "wedge": [[0, 1, 2], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
        }

        num_nodes_per_elem = self.elem_conn.shape[1]

        if self.dim == 2:
            faces_per_elem = num_nodes_per_elem
            face_nodes_def = [
                [i, (i + 1) % num_nodes_per_elem] for i in range(num_nodes_per_elem)
            ]
        elif self.dim == 3:
            if num_nodes_per_elem == 4:
                face_nodes_def = face_definitions["tet"]
            elif num_nodes_per_elem == 8:
                face_nodes_def = face_definitions["hex"]
            elif num_nodes_per_elem == 6:
                face_nodes_def = face_definitions["wedge"]
            else:
                raise NotImplementedError(
                    f"3D elements with {num_nodes_per_elem} nodes are not supported."
                )
            faces_per_elem = len(face_nodes_def)
        else:
            faces_per_elem = 0

        self.cell_neighbors = -np.ones((self.nelem, faces_per_elem), dtype=int)
        self.face_normals = np.zeros((self.nelem, faces_per_elem, 3))
        self.face_tangentials = np.zeros((self.nelem, faces_per_elem, 3))
        self.face_areas = np.zeros((self.nelem, faces_per_elem))
        self.face_midpoints = np.zeros((self.nelem, faces_per_elem, 3))
        self.face_to_cell_distances = np.zeros((self.nelem, faces_per_elem, 2))

        if faces_per_elem == 0:
            return

        # Vectorized extraction of face nodes
        face_nodes_def_arr = np.array(face_nodes_def)
        all_faces_nodes = self.elem_conn[:, face_nodes_def_arr]
        self.elem_faces = all_faces_nodes

        (
            self.cell_neighbors,
            self.face_id,
            self.face_sign,
            self.owner_cell,
            self.owner_local,
        ) = self._build_face_table(all_faces_nodes)

        # Vectorized computation of face midpoints and distances
        face_node_indices = self.node_tag_map[self.elem_faces]
        face_node_coords = self.node_coords[face_node_indices]
        self.face_midpoints = np.mean(face_node_coords, axis=2)

        cell_centroids_reshaped = self.cell_centroids[:, np.newaxis, :]
        d_i = np.linalg.norm(self.face_midpoints - cell_centroids_reshaped, axis=2)

        valid_neighbors_mask = self.cell_neighbors != -1
        neighbor_indices = np.maximum(self.cell_neighbors, 0)
        neighbor_centroids = self.cell_centroids[neighbor_indices]
        d_j_all = np.linalg.norm(self.face_midpoints - neighbor_centroids, axis=2)
        d_j = np.where(valid_neighbors_mask, d_j_all, 0)

        self.face_to_cell_distances = np.stack([d_i, d_j], axis=-1)

        # Vectorized computation of face normals and areas
        if self.dim == 2:
            p1 = face_node_coords[:, :, 0, :]
            p2 = face_node_coords[:, :, 1, :]
            delta = p2 - p1
            dx, dy = delta[:, :, 0], delta[:, :, 1]

            self.face_areas = np.sqrt(dx**2 + dy**2)

            # Avoid division by zero for normals and tangents
            length = self.face_areas
            inv_length = np.divide(
                1.0, length, where=length > 1e-9, out=np.zeros_like(length)
            )

            self.face_normals[:, :, 0] = dy * inv_length
            self.face_normals[:, :, 1] = -dx * inv_length

            self.face_tangentials[:, :, 0] = dx * inv_length
            self.face_tangentials[:, :, 1] = dy * inv_length

            # Ensure normals point outwards
            dot_product = np.einsum(
                "ijk,ijk->ij",
                self.face_normals,
                self.face_midpoints - cell_centroids_reshaped,
            )
            correction_mask = (dot_product < 0)[:, :, np.newaxis]
            self.face_normals = np.where(
                correction_mask, -self.face_normals, self.face_normals
            )

        elif self.dim == 3:
            # The 3D case is more complex due to variable nodes per face (tris/quads).
            # A loop is more straightforward here and likely not the main bottleneck
            # compared to the previous implementation.
            for i in range(self.nelem):
                for j, face_nodes in enumerate(self.elem_faces[i]):
                    node_indices = self.node_tag_map[face_nodes]
                    nodes = self.node_coords[node_indices]

                    if len(nodes) >= 3:
                        if len(nodes) == 3:  # Triangular face
                            v_diag1 = nodes[1] - nodes[0]
                            v_diag2 = nodes[2] - nodes[0]
                            normal = np.cross(v_diag1, v_diag2)
                            area = np.linalg.norm(normal) / 2.0
                        elif len(nodes) == 4:  # Quadrilateral face
                            # Calculate normal and area using diagonals
                            v_diag1 = nodes[2] - nodes[0]  # P0 to P2
                            v_diag2 = nodes[3] - nodes[1]  # P1 to P3
                            normal = np.cross(v_diag1, v_diag2)
                            area = np.linalg.norm(normal) / 2.0
                        else:
                            # Handle other polygon types or raise an error
                            raise NotImplementedError(
                                f"Face with {len(nodes)} nodes not supported for 3D area calculation."
                            )
                        self.face_areas[i, j] = area

                        if area > 1e-9:
                            normal /= 2.0 * area
                            tangent = v_diag1 / np.linalg.norm(v_diag1)
                        else:
                            normal = np.zeros(3)
                            tangent = np.zeros(3)

                        face_midpoint = self.face_midpoints[i, j]
                        if np.dot(normal, face_midpoint - self.cell_centroids[i]) < 0:
                            normal = -normal

                        self.face_normals[i, j] = normal
                        self.face_tangentials[i, j] = tangent

    @staticmethod
    def _build_face_table(elem_faces):
        """Match identical faces between cells and number the unique faces.

        Args:
            elem_faces: Node tags per face, shape ``(nelem, nfaces, nodes_per_face)``.

        Returns:
            ``(neighbors, face_id, face_sign, owner_cell, owner_local)``:
            ``neighbors`` is ``(nelem, nfaces)`` with ``-1`` on boundary faces;
            ``face_id``/``face_sign`` map every ``[cell, face]`` to its unique
            face and to ``+1`` (owner = lowest cell index) or ``-1``;
            ``owner_cell``/``owner_local`` locate each unique face.

        Raises:
            ValueError: If a face is shared by more than two cells.
        """
        nelem, nfaces, nper = elem_faces.shape
        keys = np.sort(elem_faces.reshape(-1, nper), axis=1)
        _, face_id, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        if counts.max() > 2:
            raise ValueError("Non-manifold mesh: a face is shared by more than 2 cells.")
        face_id = face_id.reshape(-1)
        order = np.argsort(face_id, kind="stable")  # ascending flat index per face
        sorted_id = face_id[order]
        paired = sorted_id[1:] == sorted_id[:-1]
        first, second = order[:-1][paired], order[1:][paired]
        neighbors = np.full(nelem * nfaces, -1, dtype=int)
        neighbors[first] = second // nfaces
        neighbors[second] = first // nfaces

        starts = np.r_[0, np.flatnonzero(sorted_id[1:] != sorted_id[:-1]) + 1]
        owner_flat = order[starts]
        face_sign = np.where(owner_flat[face_id] == np.arange(nelem * nfaces), 1, -1)
        return (
            neighbors.reshape(nelem, nfaces),
            face_id.reshape(nelem, nfaces),
            face_sign.reshape(nelem, nfaces),
            owner_flat // nfaces,
            owner_flat % nfaces,
        )

    def _compute_lsq_coefficients(self):
        """Weighted least-squares gradient coefficients (exact for linear fields).

        Boundary faces add a mirrored virtual neighbour with zero difference
        (zero-gradient ghost), which keeps corner cells well conditioned.
        """
        nb = np.maximum(self.cell_neighbors, 0)
        has_nb = self.cell_neighbors != -1
        d_real = (self.cell_centroids[nb] - self.cell_centroids[:, None, :])[..., :2]
        d_virt = 2.0 * (self.face_midpoints - self.cell_centroids[:, None, :])[..., :2]
        d = np.where(has_nb[..., None], d_real, d_virt)
        dist2 = np.maximum((d**2).sum(axis=2), 1e-300)
        w2 = 1.0 / dist2  # (1/|d|)^2
        A = np.einsum("ij,ijk,ijl->ikl", w2, d, d)
        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] ** 2
        det = np.where(np.abs(det) > 1e-300, det, 1.0)
        Ainv = np.empty_like(A)
        Ainv[:, 0, 0], Ainv[:, 1, 1] = A[:, 1, 1] / det, A[:, 0, 0] / det
        Ainv[:, 0, 1] = Ainv[:, 1, 0] = -A[:, 0, 1] / det
        coef = np.einsum("ikl,ijl->ijk", Ainv, w2[..., None] * d)
        self.lsq_coef = np.where(has_nb[..., None], coef, 0.0)

    def _compute_reconstruction_geometry(self):
        """Precompute the static coefficients used by the gradient kernel.

        Sets ``face_weights`` (own-cell interpolation weight) and
        ``face_nonorth`` (scalar non-orthogonal correction coefficient).
        """
        d_i, d_j = self.face_to_cell_distances[..., 0], self.face_to_cell_distances[..., 1]
        total = d_i + d_j
        self.face_weights = np.where(
            total > 1e-9, d_j / np.where(total > 1e-9, total, 1.0), 0.5
        )

        has_nb = self.cell_neighbors != -1
        nb = np.maximum(self.cell_neighbors, 0)
        d = self.cell_centroids[nb] - self.cell_centroids[:, None, :]
        d_norm = np.linalg.norm(d, axis=2)
        n_norm = np.linalg.norm(self.face_normals, axis=2)
        safe_d = np.where(d_norm > 1e-9, d_norm, 1.0)
        safe_n = np.where(n_norm > 1e-9, n_norm, 1.0)
        e = d / safe_d[..., None]
        k = self.face_normals / safe_n[..., None]
        d_dot_k = np.einsum("ijk,ijk->ij", d, k)
        e_dot_k = np.einsum("ijk,ijk->ij", e, k)
        valid = has_nb & (d_norm > 1e-9) & (np.abs(d_dot_k) > 1e-9)
        # (e - k (e.k)) . d / (d.k)  ==  (|d| - (e.k)(k.d)) / (d.k)
        safe_dk = np.where(valid, d_dot_k, 1.0)
        self.face_nonorth = np.where(valid, (d_norm - e_dot_k * d_dot_k) / safe_dk, 0.0)

    def fvm_geometry(self):
        """Bundle the 2D arrays the numba kernels use (see :class:`FVMGeometry`)."""
        if self.dim != 2:
            raise NotImplementedError("The FVM kernels currently support 2D meshes.")
        mid = np.ascontiguousarray(self.face_midpoints[..., :2])
        cen = np.ascontiguousarray(self.cell_centroids[:, :2])
        return FVMGeometry(
            neighbors=np.ascontiguousarray(self.cell_neighbors),
            normals=np.ascontiguousarray(self.face_normals[..., :2]),
            areas=np.ascontiguousarray(self.face_areas),
            midpoints=mid,
            face_r=mid - cen[:, None, :],
            centroids=cen,
            volumes=np.ascontiguousarray(self.cell_volumes),
            weights=np.ascontiguousarray(self.face_weights),
            nonorth=np.ascontiguousarray(self.face_nonorth),
            lsq=np.ascontiguousarray(self.lsq_coef),
            face_id=np.ascontiguousarray(self.face_id),
            face_sign=np.ascontiguousarray(self.face_sign),
            owner_cell=np.ascontiguousarray(self.owner_cell),
            owner_local=np.ascontiguousarray(self.owner_local),
        )

    def get_mesh_quality(self, metric="aspect_ratio"):
        """
        Computes mesh quality for each element using vectorized operations.
        """
        if self.dim == 1:
            return np.ones(self.nelem)
        if self.nelem == 0:
            return np.array([])

        quality = np.zeros(self.nelem)

        if self.dim == 2:
            # Vectorized calculation for 2D elements
            elem_node_indices = self.node_tag_map[self.elem_conn]
            elem_nodes_coords = self.node_coords[elem_node_indices]

            rolled_nodes = np.roll(elem_nodes_coords, -1, axis=1)
            edge_lengths = np.linalg.norm(elem_nodes_coords - rolled_nodes, axis=2)

            min_edge_lengths = np.min(edge_lengths, axis=1)
            max_edge_lengths = np.max(edge_lengths, axis=1)

            # Avoid division by zero
            quality = np.divide(
                max_edge_lengths,
                min_edge_lengths,
                out=np.full(self.nelem, float("inf")),
                where=min_edge_lengths > 1e-9,
            )

        elif self.dim == 3:
            # Vectorized calculation for 3D elements
            # This is more complex as it involves edges of faces.
            face_node_indices = self.node_tag_map[self.elem_faces]
            face_nodes_coords = self.node_coords[
                face_node_indices
            ]  # (nelem, nfaces, nnodes_per_face, 3)

            # Roll nodes within each face to compute edge vectors
            rolled_face_nodes = np.roll(face_nodes_coords, -1, axis=2)
            edge_vectors = face_nodes_coords - rolled_face_nodes

            # Calculate lengths of all edges for all faces of all elements
            face_edge_lengths = np.linalg.norm(
                edge_vectors, axis=3
            )  # (nelem, nfaces, nnodes_per_face)

            # Reshape to group all edge lengths for each element
            all_edge_lengths = face_edge_lengths.reshape(self.nelem, -1)

            min_edge_lengths = np.min(all_edge_lengths, axis=1)
            max_edge_lengths = np.max(all_edge_lengths, axis=1)

            # Avoid division by zero
            quality = np.divide(
                max_edge_lengths,
                min_edge_lengths,
                out=np.full(self.nelem, float("inf")),
                where=min_edge_lengths > 1e-9,
            )

        return quality

    def summary(self):
        """
        Prints a summary of the mesh information.
        """
        print("\n--- Mesh Summary ---")
        print(f"Mesh Dimension: {self.dim}D")
        print(f"Number of Nodes: {self.nnode}")
        print(f"Number of Elements: {self.nelem}")

        quality = self.get_mesh_quality()
        if self.nelem > 0:
            print(f"Element Type: {self.elem_conn.shape[1]}-node elements")
            avg_quality = np.mean(quality)
            print(f"Average Mesh Quality (Aspect Ratio): {avg_quality:.4f}")

        num_boundary_sets = len(self.boundary_tag_map)
        print(f"Number of Boundary Face Sets: {num_boundary_sets}")
        if num_boundary_sets > 0:
            for name, tag in self.boundary_tag_map.items():
                count = np.sum(self.boundary_faces_tags == tag)
                print(f"  - Boundary '{name}' (tag {tag}): {count} faces")
        print("--------------------\n")

    def get_mesh_data(self):
        """
        Returns all the computed mesh data in a dictionary.
        """
        return {
            "dimension": self.dim,
            "node_tags": self.node_tags,
            "node_coords": self.node_coords,
            "elem_tags": self.elem_tags,
            "elem_conn": self.elem_conn,
            "cell_volumes": self.cell_volumes,
            "cell_centroids": self.cell_centroids,
            "cell_neighbors": self.cell_neighbors,
            "boundary_faces_nodes": self.boundary_faces_nodes,
            "boundary_faces_tags": self.boundary_faces_tags,
            "boundary_tag_map": self.boundary_tag_map,
            "face_areas": self.face_areas,
            "face_normals": self.face_normals,
            "face_tangentials": self.face_tangentials,
        }


def plot_mesh(mesh: Mesh):
    """
    Visualizes the computational mesh, including element and node labels, and face normals.

    This function is useful for debugging and verifying the mesh structure.

    Args:
        mesh (Mesh): The mesh object to visualize.
    """
    if mesh.nelem == 0:
        raise ValueError("Possibly mesh has not been read. Call read_mesh() first")

    fig, ax = plt.subplots(figsize=(12, 12))

    text_flag = mesh.nelem <= 2000

    # Create a mapping from node tags to their 0-based index.
    max_tag = np.max(mesh.node_tags)
    node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
    node_tag_map[mesh.node_tags] = np.arange(mesh.nnode, dtype=np.int32)

    # Plot elements and their labels
    for i, elem_nodes_tags in enumerate(mesh.elem_conn):
        node_indices = [node_tag_map[tag] for tag in elem_nodes_tags]
        # node_indices = [
        #     np.where(mesh.node_tags == tag)[0][0] for tag in elem_nodes_tags
        # ]
        nodes = mesh.node_coords[np.array(node_indices)]
        polygon = Polygon(nodes[:, :2], edgecolor="b", facecolor="none", lw=0.5)
        ax.add_patch(polygon)
        if text_flag:
            ax.text(
                mesh.cell_centroids[i, 0],
                mesh.cell_centroids[i, 1],
                f"{i} (A={mesh.cell_volumes[i]:.2f})",
                color="blue",
                fontsize=8,
                ha="center",
            )

    # Plot node labels
    if text_flag:
        for i, coord in enumerate(mesh.node_coords):
            ax.text(
                coord[0],
                coord[1],
                str(mesh.node_tags[i]),
                color="red",
                fontsize=8,
                ha="center",
            )

    # Plot face normals
    if text_flag:
        for i in range(mesh.nelem):
            for j, _ in enumerate(mesh.elem_faces[i]):
                midpoint = mesh.face_midpoints[i, j]
                normal = mesh.face_normals[i, j]
                face_to_cell_distances = mesh.face_to_cell_distances[i, j][0]

                # Scale for visibility
                normal_scaled = normal * face_to_cell_distances * 0.5

                # Plot normal vector
                ax.quiver(
                    midpoint[0],
                    midpoint[1],
                    normal_scaled[0],
                    normal_scaled[1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="green",
                    width=0.003,
                )

    ax.set_aspect("equal", "box")
    ax.set_title("Mesh Visualization")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(False)
    plt.show(block=True)


if __name__ == "__main__":
    try:
        mesh_file = "./data/euler_mesh.msh"

        # New workflow
        mesh = Mesh()
        mesh.read_mesh(mesh_file)
        mesh.analyze_mesh()

        mesh.summary()

        mesh_data = mesh.get_mesh_data()
        print("\n--- Mesh Data Export ---")
        print(f"First 5 node coordinates:\n{mesh_data['node_coords'][:5]}")
        print(f"First 5 element connectivities:\n{mesh_data['elem_conn'][:5]}")

    except FileNotFoundError:
        print(f"Error: Mesh file not found at {mesh_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

    if mesh.dim == 2:
        plot_mesh(mesh)
