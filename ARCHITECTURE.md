# Architecture

A 2D cell-centred finite-volume solver for hyperbolic conservation laws
(compressible Euler, shallow water) on unstructured meshes, using a
second-order MUSCL scheme with an explicit Runge-Kutta time integrator.
Hot loops are numba-compiled and parallel over cells.

## Data flow

```
 .msh file                                        main.py
    |                                                |
    v                                                v
 mesh.Mesh.read_mesh  (gmsh, lazy import)      case_setup.setup_case_*
    |                                           -> U0 (nelem, nvars), bc_dict
    v                                                |
 mesh.Mesh.analyze_mesh                              |
   centroids, volumes, neighbours,                   |
   normals, areas, interpolation weights             |
    |                                                |
    +-------------------> solver.solve <-------------+
                              |
        once, before the loop |  geo      = mesh.fvm_geometry()
                              |  bc_type, bc_value = boundary.build_face_bcs(...)
                              v
                     ┌── time loop ──────────────────────────────┐
                     │ dt = time_step.calculate_adaptive_dt      │
                     │ U  = INTEGRATORS[method](residual, U, dt) │
                     └───────────────┬───────────────────────────┘
                                     v
                          residual.compute_residual(U)
                     reconstruction.compute_gradients_gaussian
                       -> reconstruction.compute_limiters (limiters.py)
                       -> residual._flux_residual
                              -> equation.roe_flux / hllc_flux
                                 equation.apply_boundary_condition
                                     |
                                     v
                           visualization.* (history of U)
```

## Modules (`src/`)

| Module | Responsibility |
| --- | --- |
| `mesh.py` | `Mesh`: reads gmsh files, builds all geometry/connectivity (vectorised). `FVMGeometry`: the contiguous 2D arrays handed to numba kernels. `plot_mesh` for debugging. `gmsh` is imported lazily, so geometry works (and is testable) without it. |
| `boundary.py` | Boundary-condition vocabulary (`BC_TYPE_MAP`). `create_numba_bcs` turns the user dict into a structured array; `build_face_bcs` resolves it **once** into per-face `bc_type` / `bc_value` arrays. |
| `limiters.py` | Slope-limiter functions, selected by integer code. |
| `reconstruction.py` | Green-Gauss gradients and the limiter kernel (cell-parallel). |
| `residual.py` | `compute_residual(U, geo, ...)`: gradient → limiter → MUSCL face states → Riemann flux → per-cell sum. The only spatial-discretisation entry point. |
| `euler_equations.py`, `shallow_water_equations.py` | numba `jitclass` equation systems with a common interface: `max_eigenvalue`, `roe_flux`, `hllc_flux`, `apply_boundary_condition`. |
| `time_step.py` | CFL time step and the explicit integrators (`INTEGRATORS`: `euler`, `rk2`). |
| `solver.py` | `solve(...)`: wires the pieces together and runs the loop. |
| `case_setup.py` | Initial conditions and BC dictionaries for the demo cases. |
| `visualization.py` | Matplotlib snapshots and animation. |

`main.py` is the script entry point; `utility/` holds gmsh mesh generators;
`tests/` is pytest.

## Conventions

- State `U` is `(nelem, nvars)`; per-face arrays are `[cell, local_face, ...]`.
- Local face `j` of a cell joins nodes `j` and `j+1`. `cell_neighbors == -1`
  marks a boundary face.
- Normals are outward unit vectors; fluxes are multiplied by face length, and
  `R_i = 1/V_i * sum_f F_f A_f`, so the update is `U - dt * R`.
- BC type codes: `1` inlet, `2` outlet, `3` wall, `4` transmissive; interior
  faces carry `0`.
- Everything static is computed once (`fvm_geometry`, `build_face_bcs`); the
  time loop only touches `U`.
- Numba kernels take the `FVMGeometry` named tuple, not the `Mesh` object, and
  choose limiters/fluxes by integer code to avoid re-compilation.

## Extending

- **New equation system**: a `jitclass` with the four methods above; no solver
  changes needed.
- **New limiter / flux / integrator**: add an entry to `LIMITER_CODES` +
  `apply_limiter`, `FLUX_CODES` + a branch in `_flux_residual`, or
  `INTEGRATORS`.
- **New BC type**: extend `BC_TYPE_MAP`, `create_numba_bcs` and each equation's
  `apply_boundary_condition`.

## Known limitations and follow-ups

- Solver kernels are 2D only (`fvm_geometry` raises for other dimensions).
- Cell "centroid" is the vertex mean; for non-triangular cells it is not the
  area centroid.
- `minmod` is mathematically identical to `barth_jespersen` in this
  formulation (`r >= 0`), and `superbee` can return `phi > 1`, which breaks the
  min/max bound the limiter is built to enforce. Treat it as experimental.
- Each interior face is evaluated twice (once per adjacent cell). A face-based
  loop would roughly halve Riemann-solver work but needs a conflict-free
  accumulation scheme.
- `history` stores every step; large runs should subsample.
- `euler_equations.py` still carries the experimental `hllc_flux_change` /
  `roe_flux_change`; they are unused by the solver.
- The inlet BC overrides *momentum* with the given velocity (not
  `rho * velocity`); outlet overrides density with `pressure`. Both are
  placeholders, not physically consistent conditions.
- The non-orthogonal gradient correction is a heuristic, scaled by
  `over_relaxation`; `main.py` runs with `1.0`.

## Testing

`python -m pytest tests`. `tests/test_fvm_core.py` covers neighbour search,
per-face BC resolution, free-stream preservation (zero residual for uniform
flow), Roe shear-wave upwinding and BC non-mutation on a small in-memory mesh.
