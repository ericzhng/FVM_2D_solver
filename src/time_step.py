"""Time-step selection (CFL) and explicit time integrators."""

import numba
import numpy as np
from numba import prange


@numba.njit(parallel=True)
def _min_cell_dt(volumes, U, equation):
    """Per-cell stable step ``sqrt(V) / max_eigenvalue`` reduced to its minimum."""
    dt_cell = np.empty(U.shape[0])
    for i in prange(U.shape[0]):
        dt_cell[i] = np.sqrt(volumes[i]) / equation.max_eigenvalue(U[i])
    return dt_cell.min()


def calculate_adaptive_dt(volumes, U, equation, cfl_number):
    """Return the CFL-limited time step.

    Args:
        volumes: Cell areas, shape ``(nelem,)``.
        U: Conservative state, shape ``(nelem, nvars)``.
        equation: Equation object providing ``max_eigenvalue(U_i)``.
        cfl_number: Courant number.

    Returns:
        ``cfl * min_i(sqrt(V_i) / lambda_max,i)``.
    """
    return cfl_number * _min_cell_dt(volumes, U, equation)


def step_euler(residual_fn, U, dt):
    """First-order forward Euler: ``U - dt R(U)``."""
    return U - dt * residual_fn(U)


def step_rk2(residual_fn, U, dt):
    """Second-order SSP Runge-Kutta (Heun)."""
    U_star = U - dt * residual_fn(U)
    return 0.5 * (U + U_star - dt * residual_fn(U_star))


INTEGRATORS = {"euler": step_euler, "rk2": step_rk2}
