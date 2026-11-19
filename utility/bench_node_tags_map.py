r"""Module: utility.bench_node_tags_map

Description: Timing comparison of two ways to map gmsh node tags to array indices: a dense
    lookup array and a dictionary, on the saved node tags in tests/node_tags.npy. It first
    checks that both maps give the same index for every tag, then prints the timings. Run it
    as a script (``python utility/bench_node_tags_map.py``).

Author: Hui Zhang
Date: 2025-07-25
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import os
import time

import numpy as np

NODE_TAGS_FILE = os.path.join(os.path.dirname(__file__), "..", "tests", "node_tags.npy")


def main():
    node_tags = np.load(NODE_TAGS_FILE)
    nnode = len(node_tags)

    start_time_block1 = time.perf_counter()

    max_tag = np.max(node_tags)
    node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
    node_tag_map[node_tags] = np.arange(nnode, dtype=np.int32)

    elapsed_time_block1 = time.perf_counter() - start_time_block1

    start_time_block2 = time.perf_counter()

    node_tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

    elapsed_time_block2 = time.perf_counter() - start_time_block2

    # Both maps must agree, otherwise the timing comparison is meaningless.
    assert all(node_tag_to_index[tag] == node_tag_map[tag] for tag in node_tags)

    print(f"Dense array map : {elapsed_time_block1:.6f} seconds")
    print(f"Dictionary map  : {elapsed_time_block2:.6f} seconds")
    print(f"{nnode} nodes; both maps agree.")


if __name__ == "__main__":
    main()
