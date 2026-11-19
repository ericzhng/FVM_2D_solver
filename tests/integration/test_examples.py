r"""Module: tests.integration.test_examples

Description: Runs every example in ``examples/`` end to end on the gmsh-free unit-square mesh with
    several algorithm choices from the command line: the solution must stay finite and
    physical, plots must be written, and the scripts must start from any working directory.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.driver import run_problem

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NAMES = ["euler_riemann", "shallow_water_riemann"]


def load_problem(name):
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / name / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROBLEM


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "options",
    [
        ["--flux", "roe", "--limiter", "barth_jespersen", "--integrator", "rk2"],
        ["--flux", "hllc", "--limiter", "venkatakrishnan", "--integrator", "euler"],
        ["--gradient", "green_gauss", "--dt", "1e-3"],
    ],
    ids=["roe-bj-rk2", "hllc-venkat-euler", "greengauss-fixed-dt"],
)
def test_example_runs_with_chosen_algorithm(name, options):
    problem = load_problem(name)
    argv = ["--mesh", "unit_square:6", "--t-end", "0.01", "--no-plot", *options]
    mesh, history, dt_history = run_problem(problem, argv)

    assert len(history) == len(dt_history) + 1
    assert sum(dt_history) == pytest.approx(0.01)
    final = history[-1]
    assert final.shape == history[0].shape and np.isfinite(final).all()
    assert (final[:, 0] > 0).all()  # density / water depth stays positive


def test_example_writes_plots(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    problem = load_problem("euler_riemann")
    argv = ["--mesh", "unit_square:4", "--t-end", "0.01", "--output-dir", str(tmp_path)]
    run_problem(problem, argv)
    names = {p.name for p in tmp_path.iterdir()}
    assert "euler_riemann.gif" in names
    assert {f"euler_riemann_final_{k}.png" for k in range(4)} <= names


@pytest.mark.parametrize("name", NAMES)
def test_example_script_starts_from_another_working_directory(name, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / name / "main.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--flux" in proc.stdout and "--integrator" in proc.stdout
