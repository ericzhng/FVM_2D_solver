"""Spatial residual: MUSCL face states + Riemann flux summed per cell.

``compute_residual`` is the only entry point the time integrators need. It
chains gradient -> limiter -> flux loop and returns
``R_i = 1/V_i * sum_f F_f A_f``, so the explicit update is ``U - dt * R``.
"""

import numba
import numpy as np
from numba import prange

from src.limiters import limiter_code
from src.mesh import FVMGeometry
from src.reconstruction import compute_gradients_gaussian, compute_limiters

FLUX_CODES = {"roe": 0, "hllc": 1}


def flux_code(name: str) -> int:
    """Map a Riemann-solver name to its integer code.

    Raises:
        ValueError: If ``name`` is not in ``FLUX_CODES``.
    """
    try:
        return FLUX_CODES[name]
    except KeyError:
        raise ValueError(
            f"Unknown flux '{name}'. Choose from {sorted(FLUX_CODES)}."
        ) from None


@numba.njit(parallel=True)
def _flux_residual(geo, U, gradients, limiters, equation, flux_id, bc_type, bc_value):
    """Sum numerical fluxes over the faces of each cell (cell-parallel)."""
    nelem, nvars = U.shape
    nfaces = geo.neighbors.shape[1]
    residual = np.zeros((nelem, nvars))
    for i in prange(nelem):
        if geo.volumes[i] <= 1e-12:
            continue
        U_L = np.empty(nvars)
        U_R = np.empty(nvars)
        normal = np.empty(2)
        for j in range(nfaces):
            nb = geo.neighbors[i, j]
            normal[0] = geo.normals[i, j, 0]
            normal[1] = geo.normals[i, j, 1]
            mx = geo.midpoints[i, j, 0]
            my = geo.midpoints[i, j, 1]

            # Limited linear reconstruction onto the face from the inside cell.
            rx = mx - geo.centroids[i, 0]
            ry = my - geo.centroids[i, 1]
            for k in range(nvars):
                dU = gradients[i, k, 0] * rx + gradients[i, k, 1] * ry
                U_L[k] = U[i, k] + limiters[i, k] * dU

            if nb != -1:
                rx = mx - geo.centroids[nb, 0]
                ry = my - geo.centroids[nb, 1]
                for k in range(nvars):
                    dU = gradients[nb, k, 0] * rx + gradients[nb, k, 1] * ry
                    U_R[k] = U[nb, k] + limiters[nb, k] * dU
            else:
                # Ghost state from the boundary condition. Copy into U_R: the
                # BC may return U_L itself and U_R is reused for the next face.
                U_R[:] = equation.apply_boundary_condition(
                    U_L, normal, bc_type[i, j], bc_value[i, j]
                )

            if flux_id == 0:
                flux = equation.roe_flux(U_L, U_R, normal)
            else:
                flux = equation.hllc_flux(U_L, U_R, normal)

            for k in range(nvars):
                residual[i, k] += flux[k] * geo.areas[i, j]
        for k in range(nvars):
            residual[i, k] /= geo.volumes[i]
    return residual


def compute_residual(
    U: np.ndarray,
    geo: FVMGeometry,
    equation,
    bc_type: np.ndarray,
    bc_value: np.ndarray,
    limiter: str = "barth_jespersen",
    flux: str = "roe",
    over_relaxation: float = 1.2,
) -> np.ndarray:
    """Evaluate the second-order finite-volume residual ``R(U)``.

    Args:
        U: Conservative state, shape ``(nelem, nvars)``.
        geo: Static mesh arrays (``mesh.fvm_geometry()``).
        equation: Jit-compiled equation object (Euler / shallow water).
        bc_type: Per-face boundary type codes, shape ``(nelem, nfaces)``.
        bc_value: Per-face boundary parameters, shape ``(nelem, nfaces, 3)``.
        limiter: Name of the slope limiter (see ``src.limiters``).
        flux: Name of the Riemann solver, ``"roe"`` or ``"hllc"``.
        over_relaxation: Non-orthogonal correction scale for the gradients.

    Returns:
        Residual array, shape ``(nelem, nvars)``.
    """
    gradients = compute_gradients_gaussian(geo, U, over_relaxation)
    limiters = compute_limiters(geo, U, gradients, limiter_code(limiter))
    return _flux_residual(
        geo, U, gradients, limiters, equation, flux_code(flux), bc_type, bc_value
    )
