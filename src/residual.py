"""Spatial residual: MUSCL face states + Riemann flux summed per cell.

``compute_residual`` is the only entry point the time integrators need. It
chains gradient -> limiter -> per-face flux -> per-cell gather and returns
``R_i = 1/V_i * sum_f F_f A_f`` (each face flux is evaluated once), so the
explicit update is ``U - dt * R``.
"""

import numba
import numpy as np
from numba import prange

from src.limiters import limiter_code
from src.mesh import FVMGeometry
from src.reconstruction import (
    compute_gradients_gaussian,
    compute_gradients_lsq,
    compute_limiters,
)

FLUX_CODES = {"roe": 0, "hllc": 1}
GRADIENT_METHODS = ("least_squares", "green_gauss")


def flux_code(name: str) -> int:
    """Map a Riemann-solver name to its integer code.

    Raises:
        ValueError: If ``name`` is not in ``FLUX_CODES``.
    """
    try:
        return FLUX_CODES[name]
    except KeyError:
        raise ValueError(f"Unknown flux '{name}'. Choose from {sorted(FLUX_CODES)}.") from None


@numba.njit(parallel=True)
def _face_fluxes(geo, U, gradients, limiters, equation, flux_id, bc_type, bc_value):
    """Numerical flux times face length for every *unique* face (face-parallel).

    The flux is evaluated once per face from the owner cell's outward normal;
    the neighbour cell receives its negative in :func:`_gather_residual`.
    """
    nfaces_u = geo.owner_cell.shape[0]
    nvars = U.shape[1]
    face_flux = np.empty((nfaces_u, nvars))
    for f in prange(nfaces_u):
        i = geo.owner_cell[f]
        j = geo.owner_local[f]
        nb = geo.neighbors[i, j]
        normal = np.empty(2)
        normal[0] = geo.normals[i, j, 0]
        normal[1] = geo.normals[i, j, 1]
        mx = geo.midpoints[i, j, 0]
        my = geo.midpoints[i, j, 1]

        # Limited linear reconstruction onto the face from the owner cell.
        U_L = np.empty(nvars)
        rx = mx - geo.centroids[i, 0]
        ry = my - geo.centroids[i, 1]
        for k in range(nvars):
            dU = gradients[i, k, 0] * rx + gradients[i, k, 1] * ry
            U_L[k] = U[i, k] + limiters[i, k] * dU

        if nb != -1:
            U_R = np.empty(nvars)
            rx = mx - geo.centroids[nb, 0]
            ry = my - geo.centroids[nb, 1]
            for k in range(nvars):
                dU = gradients[nb, k, 0] * rx + gradients[nb, k, 1] * ry
                U_R[k] = U[nb, k] + limiters[nb, k] * dU
        else:
            # Ghost state from the boundary condition.
            U_R = equation.apply_boundary_condition(U_L, normal, bc_type[i, j], bc_value[i, j])

        if flux_id == 0:
            flux = equation.roe_flux(U_L, U_R, normal)
        else:
            flux = equation.hllc_flux(U_L, U_R, normal)
        for k in range(nvars):
            face_flux[f, k] = flux[k] * geo.areas[i, j]
    return face_flux


@numba.njit(parallel=True)
def _gather_residual(geo, face_flux):
    """``R_i = 1/V_i * sum_f sign_if * F_f`` (conflict-free gather per cell)."""
    nelem = geo.neighbors.shape[0]
    nfaces = geo.neighbors.shape[1]
    nvars = face_flux.shape[1]
    residual = np.zeros((nelem, nvars))
    for i in prange(nelem):
        if geo.volumes[i] <= 1e-12:
            continue
        for j in range(nfaces):
            f = geo.face_id[i, j]
            s = geo.face_sign[i, j]
            for k in range(nvars):
                residual[i, k] += s * face_flux[f, k]
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
    gradient: str = "least_squares",
    over_relaxation: float = 0.0,
    venkat_k: float = 1.0,
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
        gradient: ``"least_squares"`` (exact for linear fields on any mesh) or
            ``"green_gauss"``.
        over_relaxation: Non-orthogonal correction scale, Green-Gauss only. The
            correction is heuristic and measured to *increase* the error on
            skewed meshes, so it is off by default.
        venkat_k: Constant ``K`` of the Venkatakrishnan limiter.

    Returns:
        Residual array, shape ``(nelem, nvars)``.
    """
    if gradient == "least_squares":
        gradients = compute_gradients_lsq(geo, U)
    elif gradient == "green_gauss":
        gradients = compute_gradients_gaussian(geo, U, over_relaxation)
    else:
        raise ValueError(f"Unknown gradient method '{gradient}'. Choose from {GRADIENT_METHODS}.")
    limiters = compute_limiters(geo, U, gradients, limiter_code(limiter), venkat_k)
    face_flux = _face_fluxes(
        geo, U, gradients, limiters, equation, flux_code(flux), bc_type, bc_value
    )
    return _gather_residual(geo, face_flux)
