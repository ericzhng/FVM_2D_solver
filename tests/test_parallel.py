"""Domain decomposition and distributed solver tests (thread-simulated ranks)."""

import os
import sys

os.environ["NUMBA_DISABLE_JIT"] = "1"  # see thread_comm.py

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from test_fvm_core import cons, make_grid_mesh  # noqa: E402
from thread_comm import run_ranks  # noqa: E402

from src.boundary import build_face_bcs, create_numba_bcs  # noqa: E402
from src.euler_equations import EulerEquations  # noqa: E402
from src.partition import build_domain, partition_rcb  # noqa: E402
from src.solver import solve  # noqa: E402

BC = {
    "left": {"type": "wall"},
    "right": {"type": "transmissive"},
    "bottom": {"type": "wall"},
    "top": {"type": "transmissive"},
}


def initial_state(mesh):
    """Discontinuity plus a smooth perturbation so every rank sees real dynamics."""
    x, y = mesh.cell_centroids[:, 0], mesh.cell_centroids[:, 1]
    rows = []
    for xi, yi in zip(x, y):
        in_blast = xi < 0.5 and yi < 0.6
        scale = 1.0 + 0.02 * np.sin(7 * xi + 3 * yi)
        rows.append(cons(scale * (1.0 if in_blast else 0.5), 0.2, -0.1, 1.8 if in_blast else 1.0))
    return np.array(rows)


@pytest.mark.parametrize("nparts", [2, 3, 5, 8])
def test_rcb_partition_is_balanced_and_complete(nparts):
    m = make_grid_mesh(10)
    part = partition_rcb(m.cell_centroids, nparts)
    counts = np.bincount(part, minlength=nparts)
    assert counts.sum() == m.nelem and (counts > 0).all()
    assert counts.max() - counts.min() <= 2


def test_domains_cover_mesh_and_halo_plan_is_symmetric():
    m = make_grid_mesh(8)
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
    m = make_grid_mesh(8)
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
