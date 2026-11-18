import unittest
import numpy as np
import sys
import os

# Disable Numba JIT compilation for testing
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from src.euler_equations import EulerEquations
from reference_fluxes import ReferenceEuler


class TestEulerEquations(unittest.TestCase):
    """
    Test suite for the euler_equations module.
    """

    def setUp(self):
        """
        Set up common data for the tests.
        """
        # Example data for a single cell/interface
        self.q_left = np.array([1.0, 0.1, 0.0, 2.5]).astype(np.float64)
        self.q_right = np.array([0.125, 0.0, 0.0, 0.25]).astype(np.float64)
        self.gamma = 1.40
        self.normal = np.array([0.0, 1.0])
        self.euler_equation = EulerEquations(self.gamma)

    def test_flux_consistency(self):
        """F(U, U) must equal the physical flux for every solver."""
        U = self.q_left
        exact = self.euler_equation._compute_flux(U, self.normal)
        for name in ("roe_flux", "hllc_flux"):
            flux = getattr(self.euler_equation, name)(U, U, self.normal)
            np.testing.assert_allclose(flux, exact, atol=1e-12, err_msg=name)

    def test_supersonic_flux_is_upwinded(self):
        """With both states moving supersonically right, the flux is F(U_L)."""
        gamma = self.gamma
        rho, u, p = 1.0, 5.0, 1.0  # Mach ~ 4.2
        UL = np.array([rho, rho * u, 0.0, p / (gamma - 1) + 0.5 * rho * u * u])
        UR = 0.9 * UL
        n = np.array([1.0, 0.0])
        exact = self.euler_equation._compute_flux(UL, n)
        for name in ("roe_flux", "hllc_flux"):
            flux = getattr(self.euler_equation, name)(UL, UR, n)
            np.testing.assert_allclose(flux, exact, rtol=1e-9, err_msg=name)

    def test_solvers_agree_on_sod_interface(self):
        """Roe and HLLC should be close (same physics) across a Sod-type jump."""
        roe = self.euler_equation.roe_flux(self.q_left, self.q_right, self.normal)
        hllc = self.euler_equation.hllc_flux(self.q_left, self.q_right, self.normal)
        np.testing.assert_allclose(roe, hllc, atol=0.15)


    def test_scalar_fluxes_match_reference_implementation(self):
        """The allocation-free kernels agree with the plain array formulation."""
        ref = ReferenceEuler(self.gamma)
        rng = np.random.default_rng(3)

        def state():
            r = rng.uniform(0.1, 3.0)
            u, v = rng.normal(0.0, 1.5, 2)
            p = rng.uniform(0.05, 3.0)
            return np.array(
                [r, r * u, r * v, p / (self.gamma - 1) + 0.5 * r * (u * u + v * v)]
            )

        for _ in range(500):
            UL, UR = state(), state()
            th = rng.uniform(0.0, 2.0 * np.pi)
            n = np.array([np.cos(th), np.sin(th)])
            for name in ("roe_flux", "hllc_flux"):
                np.testing.assert_allclose(
                    getattr(self.euler_equation, name)(UL, UR, n),
                    getattr(ref, name)(UL, UR, n),
                    atol=1e-11,
                    err_msg=name,
                )


if __name__ == "__main__":
    unittest.main()
