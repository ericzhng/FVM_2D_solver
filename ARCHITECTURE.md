# Architecture

A 2D cell-centred finite-volume solver for hyperbolic conservation laws
(compressible Euler, shallow water) on unstructured meshes, using a
second-order MUSCL scheme with an explicit Runge-Kutta time integrator.
Hot loops are numba-compiled and parallel over cells.

## Data flow

```
 .msh file                                  examples/*/main.py
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
| `driver.py` | Command-line driver shared by the examples: `Problem` description, algorithm options, mesh loading (`.msh` or built-in `unit_square:N`), plots. |
| `visualization.py` | Matplotlib snapshots and animation. |

`examples/<problem>/main.py` are the script entry points (one folder per physical
problem; the algorithm is chosen on the command line, see `src/driver.py`);
`utility/` holds standalone tools grouped by purpose; `tests/unit/` and
`tests/integration/` hold the pytest suites. See [Testing and tools](#testing-and-tools).

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

For a step-by-step explanation of the MPI workflow (with diagrams, a worked
example and a checklist for reusing it), see [PARALLEL_GUIDE.md](PARALLEL_GUIDE.md).

Two levels, which can be combined:

- **Shared memory (numba threads).** Kernels are `prange` loops over blocks of
  cells/faces; scratch buffers are allocated once per block (allocation
  serialises threads). Control with `NUMBA_NUM_THREADS`.
- **Distributed memory (MPI).** `mpiexec -n P python examples/euler_riemann/main.py` (needs `mpi4py`,
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

`python utility/benchmarks/bench_scaling.py --n 400 --steps 40 [--check]` reports ms/step
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

Measured MPI strong scaling (Microsoft MPI 10.1, mpi4py 4.1, one numba thread
per rank, 180,000 cells, Roe + RK2, Intel i7-10700K: 8 cores / 16 threads, idle
machine; `utility/benchmarks/bench_scaling.py --n 300 --steps 40`, two independent runs):

| Ranks | Run 1 ms/step | Run 2 ms/step | Speedup (run 1 / run 2) | Efficiency (run 1 / run 2) |
| --- | --- | --- | --- | --- |
| 1 | 100.0 | 90.7 | 1.00 / 1.00 | 100% / 100% |
| 2 | 70.8 | 69.2 | 1.41 / 1.31 | 71% / 66% |
| 4 | 41.7 | 41.7 | 2.40 / 2.18 | 60% / 54% |
| 8 | 29.2 | 27.2 | 3.42 / 3.34 | 43% / 42% |

Setup (global BCs + domain build) is 0.02-0.2 s and excluded. The limit is the
hardware, not communication: a control where every rank solves the *same* full
180k-cell problem with no communication slows from 44 ms (1-2 ranks) to 53 ms
(4) and 107-112 ms (8) per residual, i.e. concurrent ranks contend for shared
memory bandwidth / cache (the geometry alone is ~60 MB against a 16 MB L3).
Decomposition helps beyond that because each rank's working set shrinks. At
20k cells, where the working set is cache-resident, the residual itself scales
near-ideally (6.4 / 3.3 / 1.75 ms on 1 / 2 / 4 ranks), and halo-exchange
latency is only 8-34 us. Expect better efficiency across several nodes (each
brings its own memory bandwidth) than on one node.

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
  (`ThreadComm` in `tests/integration/test_distributed_threads.py`, runs everywhere, also with compiled numba kernels)
  and real MPI (`utility/checks/mpi_parity_check.py`, run by `tests/integration/test_mpi_real.py` when
  `mpi4py` + `mpiexec` exist, skipped otherwise). On Microsoft MPI 10.1 +
  mpi4py 4.1 the distributed result equals the serial one **exactly** (max
  difference 0.0) on 2, 3, 4 and 6 ranks, for Roe/HLLC, RK2/Euler, adaptive
  CFL (allreduce) and gathered states.

## Testing and tools

Tests are split by what they exercise:

| Folder | Contents | Run |
| --- | --- | --- |
| `tests/unit/` | **Unit tests**: one function or class in isolation, fast. `test_fvm_core.py` (face tables, per-face BCs, gradients incl. near walls, limiters, ghost states, centroids), `test_euler_equations.py` and `test_shallow_water_equations.py` (flux consistency, antisymmetry, upwinding), `test_boundary.py`, `test_limiters.py`, `test_time_step.py` (integrator order, CFL step), `test_mesh_geometry.py` (areas, closed cells, perimeter), `test_case_setup.py`, `test_partition.py` (RCB), `test_parallel.py`, `test_driver.py` (CLI options, mesh selection). | `python -m pytest tests/unit` |
| `tests/integration/` | **Integration and workflow tests**: several modules together or a full run. `test_residual_pipeline.py` (free-stream preservation, global conservation), `test_solver_workflow.py` (serial solve, `save_every`), `test_distributed_threads.py` (distributed == serial with thread-simulated ranks), `test_mpi_real.py` (real MPI via `utility/checks/mpi_parity_check.py`; skipped without mpi4py/mpiexec), `test_examples.py` (every example end to end with several algorithm choices). | `python -m pytest tests/integration` |
| `examples/` | One folder per problem (`euler_riemann/`, `shallow_water_riemann/`), each with a `main.py` that picks flux, limiter, gradient, integrator, CFL and mesh from the command line (`--help`). | `python examples/<problem>/main.py [options]` |
| `utility/mesh_generation/` | gmsh scripts that write `data/*.msh` (run from the repository root). | `python utility/mesh_generation/<tool>.py` |
| `utility/benchmarks/` | `bench_scaling.py` (strong scaling) and micro-benchmarks `bench_centroids.py`, `bench_node_tags_map.py`, `bench_norm_timing.py` (the first two read `data/mesh_arrays/`). | `python utility/benchmarks/<tool>.py` |
| `utility/checks/` | `mpi_parity_check.py`: distributed == serial under real MPI. | `mpiexec -n 4 python utility/checks/mpi_parity_check.py` |
| `utility/figures/` | `make_parallel_figures.py`: regenerates `doc/figures/`. | `python utility/figures/make_parallel_figures.py` |

`python -m pytest` runs everything (JIT is disabled for the whole session in `tests/conftest.py`);
`-m integration` / `-m "not integration"` also selects a layer. `tests/unit/` needs no gmsh, MPI or
display. 
CI runs both test folders as separate steps.
