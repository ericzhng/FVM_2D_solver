"""Slope limiters for the MUSCL reconstruction.

Both limiters are *bound-based* (Barth-Jespersen family): for every face the
gradient-extrapolated change ``diff = grad . r_face`` is compared with the
largest admissible change ``dmax = U_bound - U_i`` (``U_bound`` is the max/min
over the cell and its neighbours, chosen by the sign of ``diff``). Both have the
same sign, and the limiter returns ``phi`` in ``[0, 1]``.

* ``barth_jespersen``: ``min(1, dmax / diff)`` - sharp, but non-differentiable
  (can stall convergence to steady state).
* ``venkatakrishnan``: smooth variant; ``eps2 = (K h)^3`` switches the limiter
  off in nearly-uniform regions.

Classic ratio limiters (minmod, superbee) are *not* meaningful in this
formulation: ``r = dmax / diff >= 0`` made minmod identical to Barth-Jespersen
and let superbee exceed 1, violating the bound. They were removed;
``"minmod"`` is kept as a deprecated alias of Barth-Jespersen.

Limiters are chosen by integer code so numba compiles a single kernel.
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
