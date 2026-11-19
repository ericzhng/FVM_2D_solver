r"""Module: tests.unit.test_shallow_water_equations

Description: Unit tests for the shallow-water equations: Roe and HLLC flux consistency,
    antisymmetry, upwinding in supercritical flow and the maximum wave speed.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np
import pytest

from src.shallow_water_equations import ShallowWaterEquations

G = 9.81
NORMALS = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.6, 0.8])]


@pytest.fixture(scope="module")
def eq():
    return ShallowWaterEquations(G)


def cons(h, u, v):
    return np.array([h, h * u, h * v])


@pytest.mark.parametrize("flux", ["roe_flux", "hllc_flux"])
@pytest.mark.parametrize("n", NORMALS)
def test_flux_is_consistent_with_the_physical_flux(eq, flux, n):
    U = cons(1.3, 0.4, -0.2)
    np.testing.assert_allclose(getattr(eq, flux)(U, U, n), eq._compute_flux(U, n), atol=1e-12)


@pytest.mark.parametrize("flux", ["roe_flux", "hllc_flux"])
@pytest.mark.parametrize("n", NORMALS)
def test_flux_is_antisymmetric_under_swapping_sides(eq, flux, n):
    UL, UR = cons(2.0, 0.5, 0.1), cons(1.0, -0.3, 0.2)
    f = getattr(eq, flux)
    np.testing.assert_allclose(f(UL, UR, n), -f(UR, UL, -n), atol=1e-10)


@pytest.mark.parametrize("flux", ["roe_flux", "hllc_flux"])
def test_supersonic_flow_is_fully_upwinded(eq, flux):
    n = np.array([1.0, 0.0])
    UL, UR = cons(1.0, 10.0, 0.0), cons(0.8, 9.0, 0.0)  # Froude ~ 3 > 1, flowing right
    np.testing.assert_allclose(getattr(eq, flux)(UL, UR, n), eq._compute_flux(UL, n), rtol=1e-10)


def test_max_eigenvalue_is_speed_plus_celerity(eq):
    assert eq.max_eigenvalue(cons(2.0, 3.0, 4.0)) == pytest.approx(5.0 + np.sqrt(G * 2.0))
    assert eq.max_eigenvalue(cons(1.0, 0.0, 0.0)) == pytest.approx(np.sqrt(G))
