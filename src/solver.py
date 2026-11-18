"""Explicit finite-volume time-marching loop."""

import time
from functools import partial

from src.boundary import build_face_bcs, create_numba_bcs
from src.mesh import Mesh
from src.residual import compute_residual
from src.time_step import INTEGRATORS, calculate_adaptive_dt


def solve(
    mesh: Mesh,
    U,
    bc_dict,
    equation,
    t_end,
    limiter_type="barth_jespersen",
    flux_type="roe",
    gradient="least_squares",
    over_relaxation=0.0,
    venkat_k=1.0,
    use_adaptive_dt=True,
    cfl=0.5,
    dt_initial=0.01,
    time_integration="rk2",
    save_every=1,
):
    """Advance ``U`` to ``t_end`` with a second-order MUSCL finite-volume scheme.

    Args:
        mesh: Analysed mesh.
        U: Initial conservative state, shape ``(nelem, nvars)``.
        bc_dict: Boundary conditions keyed by physical-group name.
        equation: Equation system (e.g. ``EulerEquations``).
        t_end: Final time.
        limiter_type: ``"barth_jespersen"`` or ``"venkatakrishnan"``.
        flux_type: ``"roe"`` or ``"hllc"``.
        gradient: ``"least_squares"`` or ``"green_gauss"``.
        over_relaxation: Green-Gauss non-orthogonal correction scale (off by default).
        venkat_k: Venkatakrishnan limiter constant.
        use_adaptive_dt: Use the CFL condition; otherwise ``dt_initial``.
        cfl: CFL number for adaptive stepping.
        dt_initial: Fixed time step when not adaptive.
        time_integration: Key of ``time_step.INTEGRATORS`` (``"rk2"`` / ``"euler"``).
        save_every: Store the state every ``save_every`` steps (the final state is
            always stored) to bound memory on long runs.

    Returns:
        ``(history, dt_history)``: stored states (including the initial one) and
        the simulated time elapsed since the previous stored state.
    """
    if time_integration not in INTEGRATORS:
        raise NotImplementedError(
            f"Time integration '{time_integration}' is not supported; "
            f"choose from {sorted(INTEGRATORS)}."
        )
    step = INTEGRATORS[time_integration]

    # Everything static is resolved once, outside the time loop.
    geo = mesh.fvm_geometry()
    bc_type, bc_value = build_face_bcs(mesh, create_numba_bcs(bc_dict, mesh.boundary_tag_map))
    residual = partial(
        compute_residual,
        geo=geo,
        equation=equation,
        bc_type=bc_type,
        bc_value=bc_value,
        limiter=limiter_type,
        flux=flux_type,
        gradient=gradient,
        over_relaxation=over_relaxation,
        venkat_k=venkat_k,
    )

    history = [U.copy()]
    dt_history = []
    t, n, dt = 0.0, 0, dt_initial
    t_saved = 0.0

    while t_end - t > 1e-12 * t_end:
        start = time.perf_counter()
        if use_adaptive_dt:
            dt = calculate_adaptive_dt(geo.volumes, U, equation, cfl)
        dt = min(dt, t_end - t)

        U = step(residual, U, dt)
        t += dt
        n += 1

        done = t_end - t <= 1e-12 * t_end
        if n % save_every == 0 or done:
            history.append(U.copy())
            dt_history.append(t - t_saved)
            t_saved = t
        print(
            f"Time: {t:.4f}s / {t_end:.4f}s, dt = {dt:.4f}s, Step: {n}, "
            f"Loop Time: {time.perf_counter() - start:.4f}s"
        )

    return history, dt_history
