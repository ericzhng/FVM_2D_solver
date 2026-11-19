r"""Module: src.limiters

Description: Bound-based slope limiters for the MUSCL reconstruction. For each face the
    gradient-extrapolated change is compared with the largest admissible change toward the
    cell's neighbour bounds. ``barth_jespersen`` is sharp but non-differentiable.
    ``venkatakrishnan`` is a smooth variant. ``minmod`` is a deprecated alias of
    ``barth_jespersen``. Limiters are selected by integer code so numba compiles a single
    kernel.

Example:
    >>> from src.limiters import limiter_code
    >>> limiter_code("minmod") == limiter_code("barth_jespersen")
    True
    >>> limiter_code("venkatakrishnan")
    1

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numba

BARTH_JESPERSEN = 0
VENKATAKRISHNAN = 1

LIMITER_CODES = {
    "barth_jespersen": BARTH_JESPERSEN,
    "venkatakrishnan": VENKATAKRISHNAN,
    "minmod": BARTH_JESPERSEN,  # deprecated alias (was numerically identical)
}


def limiter_code(name: str) -> int:
    """Return the integer code of a limiter name.

    Raises:
        ValueError: If ``name`` is unknown (no silent fallback).
    """
    try:
        return LIMITER_CODES[name]
    except KeyError:
        raise ValueError(
            f"Unknown limiter '{name}'. Choose from {sorted(LIMITER_CODES)}."
        ) from None


@numba.njit(inline="always")
def apply_limiter(code, dmax, diff, eps2):
    """Limiter value for admissible change ``dmax`` and extrapolated change ``diff``.

    Args:
        code: Limiter code.
        dmax: ``U_bound - U_i`` (same sign as ``diff``).
        diff: Extrapolated change at the face (non-zero).
        eps2: Venkatakrishnan smoothing parameter ``(K h)^3`` (ignored otherwise).
    """
    a = abs(dmax)
    d = abs(diff)
    if code == BARTH_JESPERSEN:
        return min(1.0, a / d)
    # The raw Venkatakrishnan expression can reach ~1.17 when d << a: clip to 1.
    return min(1.0, (a * a + eps2 + 2.0 * a * d) / (a * a + 2.0 * d * d + a * d + eps2))
