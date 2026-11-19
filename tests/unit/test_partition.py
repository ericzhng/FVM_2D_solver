r"""Module: tests.unit.test_partition

Description: Unit tests for the recursive coordinate bisection partitioner: every cell is
    assigned to exactly one part and the parts are balanced.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np
import pytest

from src.mesh import Mesh
from src.partition import partition_rcb


@pytest.mark.parametrize("nparts", [2, 3, 5, 8])
def test_rcb_partition_is_balanced_and_complete(nparts):
    m = Mesh.unit_square(10)
    part = partition_rcb(m.cell_centroids, nparts)
    counts = np.bincount(part, minlength=nparts)
    assert counts.sum() == m.nelem and (counts > 0).all()
    assert counts.max() - counts.min() <= 2
