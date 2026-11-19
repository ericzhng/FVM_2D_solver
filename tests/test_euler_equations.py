r"""Module: tests.test_euler_equations

Description: Unit tests for the Euler flux kernels. They check flux consistency F(U, U),
    upwinding of supersonic flow, agreement of Roe and HLLC on a jump, and agreement of the
    optimised kernels with ``tests/reference_fluxes.py`` on random states. JIT is disabled at
    import.

Author: Hui Zhang
Date: 2025-07-25
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# Disable Numba JIT compilation for testing
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from reference_fluxes import ReferenceEuler  # noqa: E402

from src.euler_equations import EulerEquations  # noqa: E402


@pytest.fixture
def setup():
    """Common data for the flux tests: one left/right state pair and a normal."""
    gamma = 1.40
    return SimpleNamespace(
        q_left=np.array([1.0, 0.1, 0.0, 2.5], dtype=np.float64),
        q_right=np.array([0.125, 0.0, 0.0, 0.25], dtype=np.float64),
        gamma=gamma,
        normal=np.array([0.0, 1.0]),
        euler_equation=EulerEquations(gamma),
    )


def test_flux_consistency(setup):
    """F(U, U) must equal the physical flux for every solver."""
    U = setup.q_left
    exact = setup.euler_equation._compute_flux(U, setup.normal)
    for name in ("roe_flux", "hllc_flux"):
        flux = getattr(setup.euler_equation, name)(U, U, setup.normal)
        np.testing.assert_allclose(flux, exact, atol=1e-12, err_msg=name)


def test_supersonic_flux_is_upwinded(setup):
    """With both states moving supersonically right, the flux is F(U_L)."""
    gamma = setup.gamma
    rho, u, p = 1.0, 5.0, 1.0  # Mach ~ 4.2
    UL = np.array([rho, rho * u, 0.0, p / (gamma - 1) + 0.5 * rho * u * u])
    UR = 0.9 * UL
    n = np.array([1.0, 0.0])
    exact = setup.euler_equation._compute_flux(UL, n)
    for name in ("roe_flux", "hllc_flux"):
        flux = getattr(setup.euler_equation, name)(UL, UR, n)
        np.testing.assert_allclose(flux, exact, rtol=1e-9, err_msg=name)


def test_solvers_agree_on_sod_interface(setup):
    """Roe and HLLC should be close (same physics) across a Sod-type jump."""
    roe = setup.euler_equation.roe_flux(setup.q_left, setup.q_right, setup.normal)
    hllc = setup.euler_equation.hllc_flux(setup.q_left, setup.q_right, setup.normal)
    np.testing.assert_allclose(roe, hllc, atol=0.15)


def test_scalar_fluxes_match_reference_implementation(setup):
    """The allocation-free kernels agree with the plain array formulation."""
    ref = ReferenceEuler(setup.gamma)
    rng = np.random.default_rng(3)

    def state():
        r = rng.uniform(0.1, 3.0)
        u, v = rng.normal(0.0, 1.5, 2)
        p = rng.uniform(0.05, 3.0)
        return np.array(
            [r, r * u, r * v, p / (setup.gamma - 1) + 0.5 * r * (u * u + v * v)]
        )

    for _ in range(500):
        UL, UR = state(), state()
        th = rng.uniform(0.0, 2.0 * np.pi)
        n = np.array([np.cos(th), np.sin(th)])
        for name in ("roe_flux", "hllc_flux"):
            np.testing.assert_allclose(
                getattr(setup.euler_equation, name)(UL, UR, n),
                getattr(ref, name)(UL, UR, n),
                atol=1e-11,
                err_msg=name,
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
