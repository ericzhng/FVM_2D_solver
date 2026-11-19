r"""Module: src.driver

Description: Shared command-line driver for the scripts in ``examples/``. An example describes
    its physical problem as a ``Problem`` (equation, initial/boundary setup, default mesh and
    end time). ``run_problem`` adds the algorithm options (flux, limiter, gradient, time
    integrator, CFL, ...) as command-line arguments, runs ``solve`` serially or under MPI, and
    writes the plots. The mesh can be a gmsh file or the gmsh-free ``unit_square[:N]`` mesh.

Example:
    >>> from src.driver import Problem, parse_args
    >>> problem = Problem("p", "demo", None, None, "unit_square:4", 1.0)
    >>> args = parse_args(problem, ["--flux", "hllc"])
    >>> args.flux, args.mesh
    ('hllc', 'unit_square:4')

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from src.limiters import LIMITER_CODES
from src.mesh import Mesh
from src.parallel import get_comm
from src.residual import FLUX_CODES
from src.solver import solve
from src.time_step import INTEGRATORS

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIT_SQUARE = "unit_square"


@dataclass(frozen=True)
class Problem:
    """A physical problem: what is solved, independent of the numerical algorithm.

    Attributes:
        name: Short identifier (used in the CLI description and output file names).
        description: One-line summary shown in ``--help``.
        make_equation: Zero-argument factory returning the equation object.
        setup: ``setup(mesh) -> (U_init, bc_dict)``.
        default_mesh: Mesh used when ``--mesh`` is not given.
        t_end: Default end time.
    """

    name: str
    description: str
    make_equation: Callable[[], Any]
    setup: Callable[[Mesh], tuple]
    default_mesh: str
    t_end: float


def load_mesh(spec: str) -> Mesh:
    """Build an analysed mesh from a file path or a ``unit_square[:N]`` spec.

    Relative paths are tried against the working directory, then the repository root.
    """
    if spec.split(":")[0] == UNIT_SQUARE:
        n = int(spec.split(":")[1]) if ":" in spec else 20
        return Mesh.unit_square(n)
    path = spec
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(REPO_ROOT, spec)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mesh '{spec}' not found. Generate it with utility/mesh_generation/ or use "
            f"'--mesh {UNIT_SQUARE}:N' for the built-in gmsh-free mesh."
        )
    mesh = Mesh()
    mesh.read_mesh(path)
    mesh.analyze_mesh()
    return mesh


def parse_args(problem: Problem, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the algorithm and run options for ``problem``."""
    p = argparse.ArgumentParser(description=f"{problem.name}: {problem.description}")
    p.add_argument("--mesh", default=problem.default_mesh, help="gmsh .msh path or unit_square[:N]")
    p.add_argument("--t-end", type=float, default=problem.t_end, help="final time")
    p.add_argument("--flux", choices=sorted(FLUX_CODES), default="roe")
    p.add_argument("--limiter", choices=sorted(LIMITER_CODES), default="barth_jespersen")
    p.add_argument("--gradient", choices=["least_squares", "green_gauss"], default="least_squares")
    p.add_argument("--integrator", choices=sorted(INTEGRATORS), default="rk2")
    p.add_argument("--cfl", type=float, default=0.5, help="CFL number (adaptive step)")
    p.add_argument("--dt", type=float, default=None, help="fixed time step (disables CFL)")
    p.add_argument("--save-every", type=int, default=1, help="store every N-th step")
    p.add_argument("--output-dir", default=".", help="where plots are written")
    p.add_argument("--no-plot", action="store_true", help="skip plots (solver only)")
    p.add_argument("--show", action="store_true", help="open the interactive animation window")
    return p.parse_args(argv)


def run_problem(problem: Problem, argv: Optional[Sequence[str]] = None):
    """Run ``problem`` with the algorithm chosen on the command line.

    Serial with ``python main.py``; distributed with ``mpiexec -n P python main.py``.

    Returns:
        ``(mesh, history, dt_history)`` on rank 0, ``None`` on the other ranks.
    """
    args = parse_args(problem, argv)
    comm = get_comm()
    root = comm.rank == 0

    mesh = load_mesh(args.mesh)
    if root:
        print(f"[{problem.name}] mesh: {args.mesh}")
        mesh.summary()
    U_init, bc_dict = problem.setup(mesh)

    history, dt_history = solve(
        mesh,
        U_init,
        bc_dict,
        problem.make_equation(),
        t_end=args.t_end,
        limiter_type=args.limiter,
        flux_type=args.flux,
        gradient=args.gradient,
        time_integration=args.integrator,
        use_adaptive_dt=args.dt is None,
        cfl=args.cfl,
        dt_initial=args.dt if args.dt is not None else 1e-2,
        save_every=args.save_every,
        comm=comm,
    )
    if not root:
        return None

    if not args.no_plot:
        _write_plots(problem, mesh, history, dt_history, args)
    return mesh, history, dt_history


def _write_plots(problem, mesh, history, dt_history, args):
    import matplotlib.pyplot as plt

    from src.visualization import create_animation, plot_simulation_step

    os.makedirs(args.output_dir, exist_ok=True)
    gif = os.path.join(args.output_dir, f"{problem.name}.gif")
    create_animation(mesh, history, dt_history, filename=gif, show=args.show)
    for k in range(history[-1].shape[1]):
        png = os.path.join(args.output_dir, f"{problem.name}_final_{k}.png")
        plot_simulation_step(mesh, history[-1], "Final State", variable_to_plot=k, filename=png)
    plt.close("all")
