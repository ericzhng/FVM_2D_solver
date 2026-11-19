r"""Module: tests.integration.test_distributed_threads

Description: Integration tests of the domain decomposition and the distributed solver with
    thread-simulated ranks (``ThreadComm``): every cell is owned exactly once, halo plans
    are symmetric, and distributed results equal the serial run for 2 to 4 ranks. JIT is
    disabled (see conftest.py) because numba's default threading layer is not safe for concurrent
    parallel regions from several Python threads.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import threading

import numpy as np
import pytest

from src.boundary import build_face_bcs, create_numba_bcs
from src.euler_equations import EulerEquations
from src.mesh import Mesh
from src.partition import build_domain
from src.solver import solve

# --- In-process stand-in for MPI: ranks are threads sharing a barrier ---------
# Implements the communicator interface of src.parallel so decomposition, halo
# exchange and the distributed solver run without an MPI runtime.


class _Shared:
    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.mail = {}


class ThreadComm:
    def __init__(self, shared, rank):
        self._s = shared
        self.rank = rank
        self.size = shared.size

    def allreduce_min(self, x):
        s = self._s
        s.slots[self.rank] = x
        s.barrier.wait()
        out = min(s.slots)
        s.barrier.wait()
        return out

    def alltoall(self, objs):
        s = self._s
        for q, o in enumerate(objs):
            s.mail[(self.rank, q)] = o
        s.barrier.wait()
        out = [s.mail[(q, self.rank)] for q in range(self.size)]
        s.barrier.wait()
        return out

    def gather(self, obj):
        s = self._s
        s.slots[self.rank] = obj
        s.barrier.wait()
        out = list(s.slots) if self.rank == 0 else None
        s.barrier.wait()
        return out

    def exchange(self, U, send, recv):
        s = self._s
        for q, idx in send.items():
            s.mail[("halo", self.rank, q)] = U[idx].copy()
        s.barrier.wait()
        for q, idx in recv.items():
            U[idx] = s.mail[("halo", q, self.rank)]
        s.barrier.wait()


def run_ranks(size, fn):
    """Run ``fn(comm)`` on ``size`` thread-ranks; return the list of results."""
    shared = _Shared(size)
    results = [None] * size
    errors = []

    def target(rank):
        try:
            results[rank] = fn(ThreadComm(shared, rank))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)
            shared.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


BC = {
    "left": {"type": "wall"},
    "right": {"type": "transmissive"},
    "bottom": {"type": "wall"},
    "top": {"type": "transmissive"},
}


def cons(r, u, v, p, gamma=1.4):
    return np.array([r, r * u, r * v, p / (gamma - 1) + 0.5 * r * (u * u + v * v)])


def initial_state(mesh):
    """Discontinuity plus a smooth perturbation so every rank sees real dynamics."""
    x, y = mesh.cell_centroids[:, 0], mesh.cell_centroids[:, 1]
    rows = []
    for xi, yi in zip(x, y):
        in_blast = xi < 0.5 and yi < 0.6
        scale = 1.0 + 0.02 * np.sin(7 * xi + 3 * yi)
        rows.append(cons(scale * (1.0 if in_blast else 0.5), 0.2, -0.1, 1.8 if in_blast else 1.0))
    return np.array(rows)


def test_domains_cover_mesh_and_halo_plan_is_symmetric():
    m = Mesh.unit_square(8)
    geo = m.fvm_geometry()
    bt, bv = build_face_bcs(m, create_numba_bcs(BC, m.boundary_tag_map))
    size = 4

    doms = run_ranks(size, lambda comm: build_domain(geo, bt, bv, comm))
    owned = np.concatenate([d.global_ids[d.owned_idx] for d in doms])
    assert sorted(owned) == list(range(m.nelem))  # every cell owned exactly once
    for d in doms:
        assert (np.diff(d.global_ids) > 0).all()  # ascending global ids
        # every neighbour of an owned cell is present locally (ring 1)
        nb = d.geo.neighbors[d.owned_idx]
        full = geo.neighbors[d.global_ids[d.owned_idx]]
        assert ((nb >= 0) == (full >= 0)).all()
    for r, d in enumerate(doms):  # what r sends to q is what q expects from r
        for q, rows in d.send.items():
            expected = doms[q].global_ids[doms[q].recv[r]]
            np.testing.assert_array_equal(d.global_ids[rows], expected)


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("integration", ["rk2", "euler"])
def test_distributed_solve_matches_serial(size, integration):
    m = Mesh.unit_square(8)
    eq = EulerEquations(1.4)
    U0 = initial_state(m)
    kw = dict(
        t_end=0.15, flux_type="hllc", limiter_type="venkatakrishnan", time_integration=integration
    )

    serial, dts_serial = solve(m, U0, BC, eq, **kw)

    results = run_ranks(size, lambda comm: solve(m, U0, BC, eq, comm=comm, **kw))
    hist, dts = results[0]
    assert all(r[0] == [] for r in results[1:])  # only rank 0 receives states
    assert len(hist) == len(serial)
    np.testing.assert_allclose(dts, dts_serial, rtol=1e-12)
    np.testing.assert_allclose(hist[-1], serial[-1], rtol=0, atol=1e-11)
