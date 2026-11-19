r"""Module: utility.benchmarks.bench_centroids

Description: Compares three ways to compute cell centroids (vertex means) from the saved mesh
    arrays in data/mesh_arrays/: a loop with a linear node-tag search, a loop with a dense
    tag-to-index map, and a vectorised version. It checks that all three give the same
    centroids, then prints the timings. Run it as a script
    (``python utility/benchmarks/bench_centroids.py``).

Author: Hui Zhang
Date: 2025-07-25
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import os
import time

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mesh_arrays")


def load_mesh():
    """Load the saved mesh arrays and build the node-tag to index map."""
    node_tags = np.load(os.path.join(DATA_DIR, "node_tags.npy"))
    elem_conn = np.load(os.path.join(DATA_DIR, "elem_conn.npy"))
    node_coords = np.load(os.path.join(DATA_DIR, "node_coords.npy"))

    # Create a mapping from node tags to their 0-based index.
    max_tag = np.max(node_tags)
    node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
    node_tag_map[node_tags] = np.arange(len(node_tags), dtype=np.int32)
    return node_tags, elem_conn, node_coords, node_tag_map


def centroids_old(node_tags, elem_conn, node_coords):
    """Loop over elements, finding each node by a linear search of the tags."""
    cell_centroids = np.zeros((elem_conn.shape[0], 3))
    for i, elem_nodes_tags in enumerate(elem_conn):
        node_indices = [np.where(node_tags == tag)[0][0] for tag in elem_nodes_tags]
        nodes = node_coords[np.array(node_indices)]
        cell_centroids[i] = np.mean(nodes, axis=0)
    return cell_centroids


def centroids_loop(elem_conn, node_coords, node_tag_map):
    """Loop over elements, looking each node up in the dense tag map."""
    cell_centroids = np.zeros((elem_conn.shape[0], 3))
    for i, elem_nodes_tags in enumerate(elem_conn):
        node_indices = [node_tag_map[tag] for tag in elem_nodes_tags]
        nodes = node_coords[node_indices]
        cell_centroids[i] = np.mean(nodes, axis=0)
    return cell_centroids


def centroids_vectorized(elem_conn, node_coords, node_tag_map):
    """Computes the centroid of each element using vectorized operations."""
    # Use the map to convert element connectivity from tags to indices.
    elem_node_indices = node_tag_map[elem_conn]

    # Gather all node coordinates for all elements.
    elem_nodes_coords = node_coords[elem_node_indices]

    # Compute the mean over the nodes for each element to get the centroids.
    return np.mean(elem_nodes_coords, axis=1)


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def main():
    node_tags, elem_conn, node_coords, node_tag_map = load_mesh()

    fast, t_fast = timed(centroids_vectorized, elem_conn, node_coords, node_tag_map)
    loop, t_loop = timed(centroids_loop, elem_conn, node_coords, node_tag_map)
    old, t_old = timed(centroids_old, node_tags, elem_conn, node_coords)

    # All implementations must agree, otherwise the timing comparison is meaningless.
    np.testing.assert_allclose(loop, fast)
    np.testing.assert_allclose(old, fast)

    print(f"Vectorised          : {t_fast:.6f} seconds")
    print(f"Loop + dense map    : {t_loop:.6f} seconds")
    print(f"Loop + linear search: {t_old:.6f} seconds")
    print(f"{elem_conn.shape[0]} elements; all three agree.")


if __name__ == "__main__":
    main()
