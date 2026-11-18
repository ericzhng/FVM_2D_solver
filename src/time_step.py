"""Time-step selection (CFL) and explicit time integrators."""

import numba
import numpy as np
from src.numba_compat import prange


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


@numba.njit(parallel=True)
def _advance(U, R, dt):
    """``U - dt * R`` in one parallel pass (avoids numpy temporaries)."""
    out = np.empty_like(U)
    for i in prange(U.shape[0]):
        for k in range(U.shape[1]):
            out[i, k] = U[i, k] - dt * R[i, k]
    return out


@numba.njit(parallel=True)
def _rk2_combine(U, U_star, R_star, dt):
    """``0.5 * (U + U_star - dt * R_star)`` in one parallel pass."""
    out = np.empty_like(U)
    for i in prange(U.shape[0]):
        for k in range(U.shape[1]):
            out[i, k] = 0.5 * ((U[i, k] + U_star[i, k]) - dt * R_star[i, k])
    return out


def step_euler(residual_fn, U, dt):
    """First-order forward Euler: ``U - dt R(U)``."""
    return _advance(U, residual_fn(U), dt)


def step_rk2(residual_fn, U, dt):
    """Second-order SSP Runge-Kutta (Heun)."""
    U_star = _advance(U, residual_fn(U), dt)
    return _rk2_combine(U, U_star, residual_fn(U_star), dt)


INTEGRATORS = {"euler": step_euler, "rk2": step_rk2}
