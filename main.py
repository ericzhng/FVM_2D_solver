from src.mesh import Mesh, plot_mesh  # noqa: F401
from src.parallel import get_comm
from src.solver import solve
from src.case_setup import setup_case_shallow_water, setup_case_euler
from src.visualization import plot_simulation_step, create_animation
from src.euler_equations import EulerEquations
from src.shallow_water_equations import ShallowWaterEquations


def main():
    """
    Main function to run the 2D Finite Volume Method (FVM) solver.

    This script orchestrates the entire simulation process:
    1.  Initializes and reads the computational mesh.
    2.  Sets up the initial and boundary conditions for the chosen physical model.
    3.  Runs the FVM solver to advance the solution in time.
    4.  Visualizes the results.
    """

    # Run serially with ``python main.py`` or in parallel with
    # ``mpiexec -n 4 python main.py`` (needs mpi4py; each rank holds a subdomain).
    comm = get_comm()
    root = comm.rank == 0

    # --- 1. Initialize and Read Mesh ---
    if root:
        print("Initializing and reading mesh...")
    mesh = Mesh()
    mesh.read_mesh("data/euler_mesh.msh")
    mesh.analyze_mesh()
    if root:
        mesh.summary()
    # plot_mesh(mesh)  # Optional: Uncomment to visualize the mesh and check normals

    # --- 2. Set Up Case ---
    if root:
        print("Setting up the simulation case...")
    equation_type = "euler"  # Choose 'shallow_water' or 'euler'

    if equation_type == "shallow_water":
        U_init, bc_dict = setup_case_shallow_water(mesh)
        equation = ShallowWaterEquations(g=9.81)
        t_end = 20
    elif equation_type == "euler":
        U_init, bc_dict = setup_case_euler(mesh)
        equation = EulerEquations(gamma=1.4)
        t_end = 0.25
    else:
        raise ValueError("Invalid equation type specified.")

    # --- 3. Solve ---
    if root:
        print("Starting the FVM solver...")
    history, dt_history = solve(
        mesh,
        U_init,
        bc_dict,
        equation,
        t_end=t_end,
        limiter_type="barth_jespersen",  # Options: 'barth_jespersen', 'venkatakrishnan'
        flux_type="roe",
        gradient="least_squares",  # or 'green_gauss'
        use_adaptive_dt=True,
        cfl=0.5,
        dt_initial=1e-2,
        comm=comm,
    )
    if not root:  # only rank 0 receives the assembled states
        return
    print("Solver finished.")

    # --- 4. Visualize ---
    print("Creating animation of the results...")
    create_animation(mesh, history, dt_history, variable_to_plot=0)

    for k in range(U_init.shape[1]):
        plot_simulation_step(mesh, history[-1], "Final State", variable_to_plot=k)


if __name__ == "__main__":
    main()
