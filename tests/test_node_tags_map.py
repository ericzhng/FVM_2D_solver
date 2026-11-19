r"""Module: tests.test_node_tags_map

Description: Timing comparison of two ways to map gmsh node tags to array indices: a dense
    lookup array and a dictionary, on the saved node tags. It prints timings and asserts
    nothing.

Author: Hui Zhang
Date: 2025-07-25
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import time

import numpy as np
import pytest


@pytest.fixture
def node_tags():
    """Node tags of the saved test mesh."""
    return np.load("tests/node_tags.npy")


def test_compare(node_tags):
    """Time the dense-array and dictionary tag maps (prints timings, asserts nothing)."""
    nnode = len(node_tags)

    start_time_block1 = time.perf_counter()

    max_tag = np.max(node_tags)
    node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
    node_tag_map[node_tags] = np.arange(nnode, dtype=np.int32)

    end_time_block1 = time.perf_counter()
    elapsed_time_block1 = end_time_block1 - start_time_block1
    print(f"\nTime taken for Block 1: {elapsed_time_block1:.6f} seconds")

    start_time_block2 = time.perf_counter()

    node_tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

    end_time_block2 = time.perf_counter()
    elapsed_time_block2 = end_time_block2 - start_time_block2
    print(f"\nTime taken for Block 2: {elapsed_time_block2:.6f} seconds")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
