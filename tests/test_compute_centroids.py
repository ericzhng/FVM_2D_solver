r"""Module: tests.test_compute_centroids

Description: Runs three cell-centroid implementations (fastest, current and old loop) on the
    saved test mesh arrays in tests/*.npy. The current test checks that they run; it does not
    yet assert that they agree.

Author: Hui Zhang
Date: 2025-07-25
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def mesh_data():
    """Load the saved test-mesh arrays and build the node-tag to index map."""
    node_tags = np.load("tests/node_tags.npy")
    elem_conn = np.load("tests/elem_conn.npy")
    node_coords = np.load("tests/node_coords.npy")
    nnode = len(node_tags)
    nelem = elem_conn.shape[0]

    # Create a mapping from node tags to their 0-based index.
    max_tag = np.max(node_tags)
    node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
    node_tag_map[node_tags] = np.arange(nnode, dtype=np.int32)

    return SimpleNamespace(
        node_tags=node_tags,
        elem_conn=elem_conn,
        node_coords=node_coords,
        nnode=nnode,
        nelem=nelem,
        node_tag_map=node_tag_map,
    )


def _centroids_old(d):
    """Computes the centroid of each element."""
    cell_centroids = np.zeros((d.nelem, 3))
    for i, elem_nodes_tags in enumerate(d.elem_conn):
        node_indices = [np.where(d.node_tags == tag)[0][0] for tag in elem_nodes_tags]
        nodes = d.node_coords[np.array(node_indices)]
        cell_centroids[i] = np.mean(nodes, axis=0)


def _centroids(d):
    """Computes the centroid of each element."""
    cell_centroids = np.zeros((d.nelem, 3))

    for i, elem_nodes_tags in enumerate(d.elem_conn):
        node_indices = [d.node_tag_map[tag] for tag in elem_nodes_tags]
        nodes = d.node_coords[node_indices]
        cell_centroids[i] = np.mean(nodes, axis=0)


def _centroids_fastest(d):
    """Computes the centroid of each element using vectorized operations."""
    # Use the map to convert element connectivity from tags to indices.
    elem_node_indices = d.node_tag_map[d.elem_conn]

    # Gather all node coordinates for all elements.
    elem_nodes_coords = d.node_coords[elem_node_indices]

    # Compute the mean over the nodes for each element to get the centroids.
    cell_centroids = np.mean(elem_nodes_coords, axis=1)


def test_compare(mesh_data):
    """All three centroid implementations run on the saved mesh."""
    _centroids_fastest(mesh_data)
    _centroids(mesh_data)
    _centroids_old(mesh_data)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
