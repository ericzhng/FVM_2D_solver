r"""Module: utility.checks.mpi_parity_check

Description: Real-MPI parity check: the distributed solve must equal the serial solve. It covers
    walls and open boundaries, both fluxes and integrators, the adaptive CFL step (allreduce)
    and gathered output. Run it under mpiexec, for example
    ``mpiexec -n 4 python utility/checks/mpi_parity_check.py``. It exits non-zero on mismatch.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.euler_equations import EulerEquations  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.parallel import MPIComm  # noqa: E402
from src.solver import solve  # noqa: E402

BC = {
    "left": {"type": "wall"},
    "right": {"type": "transmissive"},
    "bottom": {"type": "wall"},
    "top": {"type": "transmissive"},
}


def initial_state(mesh, gamma=1.4):
    x, y = mesh.cell_centroids[:, 0], mesh.cell_centroids[:, 1]
    blast = (x < 0.5) & (y < 0.6)
    rho = np.where(blast, 1.0, 0.5) * (1.0 + 0.02 * np.sin(7 * x + 3 * y))
    p = np.where(blast, 1.8, 1.0)
    u, v = 0.2, -0.1
    E = p / (gamma - 1) + 0.5 * rho * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, E], axis=1)


def quiet_solve(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return solve(*args, **kwargs)


def main():
    comm = MPIComm()
    mesh = Mesh.unit_square(24)
    U0 = initial_state(mesh)
    eq = EulerEquations(1.4)
    worst = 0.0
    ok = True
    for flux in ("roe", "hllc"):
        for integ in ("rk2", "euler"):
            kw = dict(
                t_end=0.12,
                flux_type=flux,
                limiter_type="venkatakrishnan",
                time_integration=integ,
                save_every=3,
            )
            ref_hist, ref_dt = quiet_solve(mesh, U0, BC, eq, **kw)
            hist, dts = quiet_solve(mesh, U0, BC, eq, comm=comm, **kw)
            if comm.rank == 0:
                err = max(np.abs(a - b).max() for a, b in zip(hist, ref_hist))
                same_len = len(hist) == len(ref_hist)
                dt_err = np.abs(np.array(dts) - np.array(ref_dt)).max()
                worst = max(worst, err)
                good = same_len and err < 1e-11 and dt_err < 1e-12
                ok &= good
                print(
                    f"P={comm.size} {flux}/{integ}: states={len(hist)} "
                    f"max|dU|={err:.1e} max|ddt|={dt_err:.1e} -> {'OK' if good else 'FAIL'}"
                )
            else:
                ok &= hist == []  # only rank 0 receives states
    ok = comm.allreduce_min(1.0 if ok else 0.0) == 1.0
    if comm.rank == 0:
        print("PARITY", "PASSED" if ok else "FAILED", f"(worst |dU| = {worst:.1e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
