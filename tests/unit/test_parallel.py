r"""Module: tests.unit.test_parallel

Description: Unit tests for the communicator layer: the serial communicator does nothing and
    get_comm returns a single-rank communicator outside mpiexec.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np

from src.parallel import SerialComm, get_comm


def test_serial_comm_is_identity():
    c = SerialComm()
    assert (c.rank, c.size) == (0, 1)
    assert c.allreduce_min(2.5) == 2.5
    assert c.alltoall([7]) == [7]
    assert c.gather("x") == ["x"]
    U = np.arange(4.0)
    c.exchange(U, {}, {})
    np.testing.assert_array_equal(U, np.arange(4.0))


def test_get_comm_returns_single_rank_outside_mpiexec():
    c = get_comm()
    assert (c.rank, c.size) == (0, 1)
