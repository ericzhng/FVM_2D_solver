r"""Module: tests.unit.test_time_step

Description: Unit tests for the CFL time step and the explicit time integrators (exactness on
    simple ODEs and the formal order of accuracy).

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np
import pytest

from src.euler_equations import EulerEquations
from src.time_step import INTEGRATORS, calculate_adaptive_dt, step_euler, step_rk2


def test_integrator_registry():
    assert INTEGRATORS == {"euler": step_euler, "rk2": step_rk2}


def test_euler_step_is_u_minus_dt_residual():
    U = np.array([[1.0, 2.0], [3.0, 4.0]])
    R = np.array([[0.5, -1.0], [2.0, 0.0]])
    np.testing.assert_allclose(step_euler(lambda _: R, U, 0.1), U - 0.1 * R)


def test_rk2_is_exact_for_constant_residual():
    U = np.arange(6.0).reshape(3, 2)
    R = np.ones_like(U)
    np.testing.assert_allclose(step_rk2(lambda _: R, U, 0.2), U - 0.2 * R)


def test_rk2_matches_heun_taylor_polynomial():
    """For dU/dt = -U (residual = U) the step multiplies by ``1 - dt + dt^2 / 2``."""
    U, dt = np.array([[2.0]]), 0.3
    np.testing.assert_allclose(step_rk2(lambda u: u, U, dt), U * (1 - dt + dt**2 / 2))


@pytest.mark.parametrize("name,order", [("euler", 1), ("rk2", 2)])
def test_integrators_converge_at_their_order(name, order):
    step = INTEGRATORS[name]

    def error(n):
        U, dt = np.array([[1.0]]), 1.0 / n
        for _ in range(n):
            U = step(lambda u: u, U, dt)
        return abs(U[0, 0] - np.exp(-1.0))

    rate = np.log2(error(20) / error(40))
    assert rate == pytest.approx(order, abs=0.15)


def test_adaptive_dt_is_cfl_times_sqrt_volume_over_wave_speed():
    eq = EulerEquations(1.4)
    rho, p = 1.0, 1.0
    U = np.tile([rho, 0.0, 0.0, p / 0.4], (5, 1))
    volumes = np.array([0.04, 0.09, 0.01, 0.04, 0.04])
    c = np.sqrt(1.4 * p / rho)
    assert calculate_adaptive_dt(volumes, U, eq, 0.5) == pytest.approx(0.5 * 0.1 / c)
