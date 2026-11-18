"""Second-order reconstruction: cell gradients (least squares or Green-Gauss) and limiting.

All kernels are cell-parallel numba loops that read the static geometry from a
:class:`src.mesh.FVMGeometry` bundle and write one entry per cell.
"""

import numba
import numpy as np
from numba import prange

from src.limiters import apply_limiter

_EPS = 1e-9
_BLOCK = 256  # cells per parallel task; scratch buffers are allocated once per block


@numba.njit(parallel=True)
def compute_gradients_gaussian(geo, U, over_relaxation):
    """Compute cell-centred gradients with the Green-Gauss theorem.

    ``grad(U)_i = 1/V_i * sum_f U_f n_f A_f``. Interior face values are a
    distance-weighted average of the two cells plus an optional non-orthogonal
    correction; boundary faces use the interior value.

    Args:
        geo: Static mesh arrays (see ``Mesh.fvm_geometry``).
        U: Conservative state, shape ``(nelem, nvars)``.
        over_relaxation: Scale of the non-orthogonal correction (0 disables).

    Returns:
        Gradients, shape ``(nelem, nvars, 2)``.
    """
    nelem, nvars = U.shape
    nfaces = geo.neighbors.shape[1]
    gradients = np.zeros((nelem, nvars, 2))
    for i in prange(nelem):
        if geo.volumes[i] <= _EPS:
            continue
        for j in range(nfaces):
            nb = geo.neighbors[i, j]
            nx = geo.normals[i, j, 0]
            ny = geo.normals[i, j, 1]
            area = geo.areas[i, j]
            w = geo.weights[i, j]
            c = over_relaxation * geo.nonorth[i, j]
            for k in range(nvars):
                if nb != -1:
                    u_f = w * U[i, k] + (1.0 - w) * U[nb, k]
                    u_f += c * (U[nb, k] - U[i, k])
                else:
                    u_f = U[i, k]
                gradients[i, k, 0] += u_f * nx * area
                gradients[i, k, 1] += u_f * ny * area
        for k in range(nvars):
            gradients[i, k, 0] /= geo.volumes[i]
            gradients[i, k, 1] /= geo.volumes[i]
    return gradients


@numba.njit(parallel=True)
def compute_gradients_lsq(geo, U, equation, bc_type, bc_value):
    """Compute gradients with weighted least squares (exact for linear fields).

    The coefficients are geometry-only and precomputed by the mesh. On boundary
    faces the neighbour value is the boundary-condition ghost state built from
    the cell value, so near-wall gradients respect the BC (e.g. a reflected
    normal velocity at a wall).

    Args:
        geo: Static mesh arrays.
        U: Conservative state, shape ``(nelem, nvars)``.
        equation: Equation object providing ``apply_boundary_condition``.
        bc_type: Per-face boundary type codes.
        bc_value: Per-face boundary parameters.

    Returns:
        Gradients, shape ``(nelem, nvars, 2)``.
    """
    nelem, nvars = U.shape
    nfaces = geo.neighbors.shape[1]
    gradients = np.zeros((nelem, nvars, 2))
    ncells = geo.grad_cells.shape[0]
    nblocks = (ncells + _BLOCK - 1) // _BLOCK
    for blk in prange(nblocks):
        normal = np.empty(2)
        for c in range(blk * _BLOCK, min((blk + 1) * _BLOCK, ncells)):
            i = geo.grad_cells[c]
            for j in range(nfaces):
                nb = geo.neighbors[i, j]
                if nb != -1:
                    for k in range(nvars):
                        dU = U[nb, k] - U[i, k]
                        gradients[i, k, 0] += geo.lsq[i, j, 0] * dU
                        gradients[i, k, 1] += geo.lsq[i, j, 1] * dU
                else:
                    normal[0] = geo.normals[i, j, 0]
                    normal[1] = geo.normals[i, j, 1]
                    ghost = equation.apply_boundary_condition(
                        U[i], normal, bc_type[i, j], bc_value[i, j]
                    )
                    for k in range(nvars):
                        dU = ghost[k] - U[i, k]
                        gradients[i, k, 0] += geo.lsq[i, j, 0] * dU
                        gradients[i, k, 1] += geo.lsq[i, j, 1] * dU
    return gradients


@numba.njit(parallel=True)
def compute_limiters(geo, U, gradients, limiter_code, venkat_k, equation, bc_type, bc_value):
    """Compute the bound-based limiter ``phi`` per cell and variable.

    The gradient is scaled so the value extrapolated to every face midpoint
    stays within the min/max of the cell and its neighbours.

    Args:
        geo: Static mesh arrays.
        U: Conservative state, shape ``(nelem, nvars)``.
        gradients: Output of a ``compute_gradients_*`` kernel.
        limiter_code: Integer code from :mod:`src.limiters`.
        venkat_k: Venkatakrishnan constant ``K`` (``eps2 = (K sqrt(V))^3``).
        equation, bc_type, bc_value: Used to include boundary ghost states in the
            min/max bounds (so a wall's reflected value does not clip the gradient).

    Returns:
        Limiter values in ``[0, 1]``, shape ``(nelem, nvars)``.
    """
    nelem, nvars = U.shape
    nfaces = geo.neighbors.shape[1]
    limiters = np.ones((nelem, nvars))
    ncells = geo.grad_cells.shape[0]
    nblocks = (ncells + _BLOCK - 1) // _BLOCK
    for blk in prange(nblocks):
        u_hi = np.empty(nvars)
        u_lo = np.empty(nvars)
        normal = np.empty(2)
        for c in range(blk * _BLOCK, min((blk + 1) * _BLOCK, ncells)):
            i = geo.grad_cells[c]
            eps2 = (venkat_k * np.sqrt(geo.volumes[i])) ** 3
            # Bounds from the cell, its neighbours and its boundary ghost states.
            for k in range(nvars):
                u_hi[k] = U[i, k]
                u_lo[k] = U[i, k]
            for j in range(nfaces):
                nb = geo.neighbors[i, j]
                if nb != -1:
                    for k in range(nvars):
                        u_hi[k] = max(u_hi[k], U[nb, k])
                        u_lo[k] = min(u_lo[k], U[nb, k])
                else:
                    normal[0] = geo.normals[i, j, 0]
                    normal[1] = geo.normals[i, j, 1]
                    ghost = equation.apply_boundary_condition(
                        U[i], normal, bc_type[i, j], bc_value[i, j]
                    )
                    for k in range(nvars):
                        u_hi[k] = max(u_hi[k], ghost[k])
                        u_lo[k] = min(u_lo[k], ghost[k])
            for k in range(nvars):
                u_i = U[i, k]
                phi = 1.0
                for j in range(nfaces):
                    diff = (
                        gradients[i, k, 0] * geo.face_r[i, j, 0]
                        + gradients[i, k, 1] * geo.face_r[i, j, 1]
                    )
                    if abs(diff) > _EPS:
                        bound = u_hi[k] if diff > 0 else u_lo[k]
                        phi = min(phi, apply_limiter(limiter_code, bound - u_i, diff, eps2))
                limiters[i, k] = phi
    return limiters
