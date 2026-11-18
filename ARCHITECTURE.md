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
                     reconstruction.compute_gradients_lsq  (or _gaussian)
                       -> reconstruction.compute_limiters (limiters.py)
                       -> residual._face_fluxes   (once per unique face)
                              -> equation.roe_flux / hllc_flux
                                 equation.apply_boundary_condition
                       -> residual._gather_residual (per cell, +/- sign)
                                     |
                                     v
                           visualization.* (history of U)
```

## Modules (`src/`)

| Module | Responsibility |
| --- | --- |
| `mesh.py` | `Mesh`: reads gmsh files, builds all geometry/connectivity (vectorised): area centroids, unique-face table, least-squares coefficients. `FVMGeometry`: the contiguous 2D arrays handed to numba kernels. `plot_mesh` for debugging. `gmsh` is imported lazily, so geometry works (and is testable) without it. |
| `boundary.py` | Boundary-condition vocabulary (`BC_TYPE_MAP`). `create_numba_bcs` turns the user dict into a structured array; `build_face_bcs` resolves it **once** into per-face `bc_type` / `bc_value` arrays. |
| `limiters.py` | Bound-based slope limiters (Barth-Jespersen, Venkatakrishnan), selected by integer code. |
| `reconstruction.py` | Least-squares (default) and Green-Gauss gradients, plus the limiter kernel (cell-parallel). |
| `residual.py` | `compute_residual(U, geo, ...)`: gradient → limiter → MUSCL face states → Riemann flux once per unique face → per-cell gather. The only spatial-discretisation entry point. |
| `euler_equations.py`, `shallow_water_equations.py` | numba `jitclass` equation systems with a common interface: `max_eigenvalue`, `roe_flux`, `hllc_flux`, `apply_boundary_condition`. |
| `time_step.py` | CFL time step and the explicit integrators (`INTEGRATORS`: `euler`, `rk2`). |
| `solver.py` | `solve(...)`: wires the pieces together and runs the loop (serial or distributed via `comm=`). |
| `partition.py` | Domain decomposition: RCB partitioner, two-ring halos, local geometry/face tables, halo exchange plan (`Domain`). |
| `parallel.py` | Communicator interface: `SerialComm`, `MPIComm` (mpi4py, lazy import), `get_comm()`. |
| `case_setup.py` | Initial conditions and BC dictionaries for the demo cases. |
| `visualization.py` | Matplotlib snapshots and animation. |

`main.py` is the script entry point; `utility/` holds gmsh mesh generators and
`bench_scaling.py` (benchmark / parallel-vs-serial check);
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

## Parallel execution

Two levels, which can be combined:

- **Shared memory (numba threads).** Kernels are `prange` loops over blocks of
  cells/faces; scratch buffers are allocated once per block (allocation
  serialises threads). Control with `NUMBA_NUM_THREADS`.
- **Distributed memory (MPI).** `mpiexec -n P python main.py` (needs `mpi4py`,
  see `requirements-mpi.txt`). Every rank builds the global mesh, then
  `partition.build_domain` keeps owned cells + two halo rings. A residual
  evaluation does **one** halo exchange of `U` (non-blocking, point-to-point);
  a time step does two (RK2) plus one `allreduce(min)` for the CFL step.
  Gradients/limiters run on owned + ring 1, fluxes only on faces touching an
  owned cell, so redundant work is small (180k cells on 64 ranks: <=5% extra
  gradient work, <=2.5% extra flux work, <=16 messages of <25 KB per rank).
  Local cells keep ascending global order, so a shared face is evaluated from
  the same side as in a serial run: **results are bitwise identical to serial**
  for any rank count (verified for 2-6 ranks).

With many ranks per node use one numba thread per rank
(`NUMBA_NUM_THREADS=1`) to avoid oversubscription. `solve(..., comm=comm)`
returns assembled global states on rank 0 only.

`python utility/bench_scaling.py --n 400 --steps 40 [--check]` reports ms/step
(from `solve(..., timings=)`, so setup is excluded) and, with `--check`, the
difference to a serial solve; run it under `mpiexec` with increasing `-n` for a
strong-scaling table. `solve` gathers the global state every `save_every` steps
-- leave it large when benchmarking, or you measure pickling the whole field.

Measured single-node numbers (this machine, 16 logical cores, 80k cells):

| Kernel | before | after |
| --- | --- | --- |
| Roe face fluxes, 1 thread | 264 ms | ~10 ms |
| Roe face fluxes, 16 threads | 39 ms | ~2 ms |
| full RK2 step, 10k cells | 79 ms | ~1.5 ms |

Threaded scaling of the numba kernels is limited (about 3x on 16 threads at
180k cells: they are memory-bound), which is exactly where decomposition
helps: each rank's working set is small and cache-resident.

## Design notes

- **Face-based fluxes.** Each unique face is evaluated once from its owner cell
  (lowest cell index); the neighbour receives the negative flux in a
  conflict-free per-cell gather (`face_id`, `face_sign`). Conservation holds by
  construction (a closed wall domain conserves mass/energy to round-off).
- **Gradients.** Weighted least squares is the default: exact for linear fields
  on any mesh. Green-Gauss with linear interpolation is *not* exact on skewed
  meshes and its non-orthogonal correction measurably increased the error, so
  that correction (`over_relaxation`) is off by default. Every boundary face
  adds a *virtual neighbour* (the cell centroid mirrored across the face) whose
  value is the boundary-condition ghost state of the cell, so wall cells see the
  reflected normal velocity (exact for wall-consistent linear fields), outflow
  cells see zero difference, and corner cells stay well conditioned. The limiter
  includes the same ghost states in its min/max bounds.
- **Limiters.** Both are bound-based, with `phi` in `[0, 1]`. Ratio limiters
  (minmod, superbee) do not fit this formulation (minmod was identical to
  Barth-Jespersen and superbee could exceed the bound), so they were removed;
  `"minmod"` remains a deprecated alias.
- **Boundary ghost states** are built from primitives and never mutate the
  interior state: inlet = prescribed velocity (density, pressure from inside);
  outlet = prescribed pressure (Euler) or water depth (shallow water), the rest
  from inside; wall = reflection; transmissive = zero gradient.

## Known limitations

- Solver kernels are 2D only (`fvm_geometry` raises for other dimensions).
- Inlet/outlet conditions are simple primitive-variable Dirichlet ghost states,
  not characteristic (non-reflecting) conditions; strong waves can reflect.
- `history` stores every `save_every`-th step in memory; with MPI it is gathered
  on rank 0 at each saved step (use a large `save_every` for big runs).
- MPI: every rank holds the *global* mesh during setup (O(N) memory per rank);
  the partitioner is geometric (RCB), not graph-based (METIS); communication is
  not overlapped with computation; the shallow-water fluxes are not yet
  allocation-free.
- The distributed logic is verified two ways: thread-simulated ranks
  (`tests/thread_comm.py`, runs everywhere, also with compiled numba kernels)
  and real MPI (`tests/mpi_parity.py`, run by `tests/test_mpi_real.py` when
  `mpi4py` + `mpiexec` exist, skipped otherwise). On Microsoft MPI 10.1 +
  mpi4py 4.1 the distributed result equals the serial one **exactly** (max
  difference 0.0) on 2, 3, 4 and 6 ranks, for Roe/HLLC, RK2/Euler, adaptive
  CFL (allreduce) and gathered states.

## Testing

`python -m pytest tests`. `tests/test_fvm_core.py` uses small in-memory meshes
and covers: neighbour search and the unique-face table, per-face BC resolution,
free-stream preservation, global conservation, least-squares exactness on a
skewed mesh, near-wall gradient exactness, Roe shear-wave upwinding, inlet/outlet ghost states, area
centroids, `save_every`, partitioning/halo plans and
distributed-vs-serial equality (`tests/test_parallel.py`). `tests/test_euler_equations.py` checks flux
consistency and upwinding.
