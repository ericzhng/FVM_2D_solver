"""Second-order reconstruction: cell gradients (least squares or Green-Gauss) and limiting.

All kernels are cell-parallel numba loops that read the static geometry from a
:class:`src.mesh.FVMGeometry` bundle and write one entry per cell.
"""

import numba
import numpy as np
from numba import prange

from src.limiters import apply_limiter

_EPS = 1e-9


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
def compute_gradients_lsq(geo, U):
    """Compute gradients with weighted least squares (exact for linear fields).

    The coefficients are geometry-only and precomputed by the mesh, so this is a
    single pass over the neighbours.

    Args:
        geo: Static mesh arrays.
        U: Conservative state, shape ``(nelem, nvars)``.

    Returns:
        Gradients, shape ``(nelem, nvars, 2)``.
    """
    nelem, nvars = U.shape
    nfaces = geo.neighbors.shape[1]
    gradients = np.zeros((nelem, nvars, 2))
    for i in prange(nelem):
        for j in range(nfaces):
            nb = geo.neighbors[i, j]
            if nb == -1:
                continue
            for k in range(nvars):
                dU = U[nb, k] - U[i, k]
                gradients[i, k, 0] += geo.lsq[i, j, 0] * dU
                gradients[i, k, 1] += geo.lsq[i, j, 1] * dU
    return gradients


@numba.njit(parallel=True)
def compute_limiters(geo, U, gradients, limiter_code, venkat_k):
    """Compute the bound-based limiter ``phi`` per cell and variable.

    The gradient is scaled so the value extrapolated to every face midpoint
    stays within the min/max of the cell and its neighbours.

    Args:
        geo: Static mesh arrays.
        U: Conservative state, shape ``(nelem, nvars)``.
        gradients: Output of a ``compute_gradients_*`` kernel.
        limiter_code: Integer code from :mod:`src.limiters`.
        venkat_k: Venkatakrishnan constant ``K`` (``eps2 = (K sqrt(V))^3``).

    Returns:
        Limiter values in ``[0, 1]``, shape ``(nelem, nvars)``.
    """
    nelem, nvars = U.shape
    nfaces = geo.neighbors.shape[1]
    limiters = np.ones((nelem, nvars))
    for i in prange(nelem):
        eps2 = (venkat_k * np.sqrt(geo.volumes[i])) ** 3
        for k in range(nvars):
            u_i = U[i, k]
            u_max = u_i
            u_min = u_i
            for j in range(nfaces):
                nb = geo.neighbors[i, j]
                if nb != -1:
                    u_max = max(u_max, U[nb, k])
                    u_min = min(u_min, U[nb, k])
            phi = 1.0
            for j in range(nfaces):
                diff = (
                    gradients[i, k, 0] * geo.face_r[i, j, 0]
                    + gradients[i, k, 1] * geo.face_r[i, j, 1]
                )
                if abs(diff) > _EPS:
                    bound = u_max if diff > 0 else u_min
                    phi = min(phi, apply_limiter(limiter_code, bound - u_i, diff, eps2))
            limiters[i, k] = phi
    return limiters
