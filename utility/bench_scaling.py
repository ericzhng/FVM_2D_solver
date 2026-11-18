"""Strong-scaling benchmark and parallel-vs-serial check (no gmsh needed).

Run serially, with numba threads, or with MPI ranks::

    python utility/bench_scaling.py --n 400 --steps 40
    mpiexec -n 8 python utility/bench_scaling.py --n 400 --steps 40 --check

Use ``NUMBA_NUM_THREADS=1`` when running many ranks per node (one thread per
rank) to avoid oversubscribing cores. ``--check`` also solves serially on every
rank and compares the assembled MPI result with it (slow for big meshes).
"""

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.euler_equations import EulerEquations  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.parallel import get_comm  # noqa: E402
from src.solver import solve  # noqa: E402


def initial_state(mesh, gamma=1.4):
    x, y = mesh.cell_centroids[:, 0], mesh.cell_centroids[:, 1]
    blast = (x - 0.4) ** 2 + (y - 0.5) ** 2 < 0.04
    rho = np.where(blast, 1.0, 0.4) * (1.0 + 0.02 * np.sin(9 * x + 5 * y))
    p = np.where(blast, 2.0, 1.0)
    u, v = 0.2, -0.1
    E = p / (gamma - 1) + 0.5 * rho * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, E], axis=1)


def run(mesh, U0, comm, steps, flux, limiter, dt, quiet=True):
    bc = {k: {"type": "transmissive"} for k in mesh.boundary_tag_map}
    sink = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        return solve(
            mesh,
            U0,
            bc,
            EulerEquations(1.4),
            t_end=steps * dt,
            flux_type=flux,
            limiter_type=limiter,
            use_adaptive_dt=False,
            dt_initial=dt,
            comm=comm,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="grid is n x n squares (2 n^2 cells)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--flux", default="roe", choices=["roe", "hllc"])
    ap.add_argument("--limiter", default="barth_jespersen")
    ap.add_argument("--dt", type=float, default=2e-4)
    ap.add_argument("--check", action="store_true", help="compare with a serial solve")
    args = ap.parse_args()

    comm = get_comm()
    mesh = Mesh.unit_square(args.n)
    U0 = initial_state(mesh)

    run(mesh, U0, comm, 2, args.flux, args.limiter, args.dt)  # warm-up / compile
    comm.allreduce_min(0.0)  # cheap barrier
    t0 = time.perf_counter()
    hist, _ = run(mesh, U0, comm, args.steps, args.flux, args.limiter, args.dt)
    comm.allreduce_min(0.0)
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        threads = os.environ.get("NUMBA_NUM_THREADS", "all")
        print(
            f"cells={mesh.nelem} ranks={comm.size} numba_threads={threads} "
            f"steps={args.steps}: {elapsed:.3f} s total, "
            f"{1e3 * elapsed / args.steps:.2f} ms/step "
            f"({mesh.nelem * args.steps / elapsed / 1e6:.2f} M cell-steps/s)"
        )

    if args.check:
        from src.parallel import SerialComm

        ref, _ = run(mesh, U0, SerialComm(), args.steps, args.flux, args.limiter, args.dt)
        if comm.rank == 0:
            err = np.abs(hist[-1] - ref[-1]).max()
            print(f"max |parallel - serial| = {err:.3e}")


if __name__ == "__main__":
    main()
