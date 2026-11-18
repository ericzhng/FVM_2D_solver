"""Second-order reconstruction: Green-Gauss gradients and slope limiting.

Both kernels are cell-parallel numba loops that read the static geometry from a
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
def compute_limiters(geo, U, gradients, limiter_code):
    """Compute the Barth-Jespersen style limiter ``phi`` per cell and variable.

    The gradient is scaled so the value extrapolated to every face midpoint
    stays within the min/max of the cell and its neighbours.

    Args:
        geo: Static mesh arrays.
        U: Conservative state, shape ``(nelem, nvars)``.
        gradients: Output of :func:`compute_gradients_gaussian`.
        limiter_code: Integer code from :mod:`src.limiters`.

    Returns:
        Limiter values in ``[0, 1]`` (``[0, 2]`` for superbee), shape
        ``(nelem, nvars)``.
    """
    nelem, nvars = U.shape
    nfaces = geo.neighbors.shape[1]
    limiters = np.ones((nelem, nvars))
    for i in prange(nelem):
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
                    phi = min(phi, apply_limiter(limiter_code, (bound - u_i) / diff))
            limiters[i, k] = phi
    return limiters
