"""Slope-limiter functions for the MUSCL reconstruction.

Each limiter maps the ratio ``r = (U_bound - U_i) / (U_face_extrap - U_i)``
(always >= 0 in the Barth-Jespersen formulation used here) to a limiter value
``phi``. Limiters are selected through an integer code so numba compiles a single
kernel instead of one specialisation per function object.
"""

import numba

BARTH_JESPERSEN = 0
MINMOD = 1
SUPERBEE = 2

LIMITER_CODES = {
    "barth_jespersen": BARTH_JESPERSEN,
    "minmod": MINMOD,
    "superbee": SUPERBEE,
}


def limiter_code(name: str) -> int:
    """Return the integer code of a limiter name.

    Args:
        name: One of ``LIMITER_CODES``.

    Returns:
        The integer code understood by :func:`apply_limiter`.

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
def apply_limiter(code, r):
    """Evaluate the limiter ``code`` at ratio ``r``."""
    if code == BARTH_JESPERSEN:
        return min(1.0, r)
    if code == MINMOD:
        return max(0.0, min(1.0, r))
    # SUPERBEE
    return max(0.0, max(min(2.0 * r, 1.0), min(r, 2.0)))
