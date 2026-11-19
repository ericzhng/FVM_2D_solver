r"""Module: tests.unit.test_driver

Description: Unit tests for the example driver: command-line parsing and mesh selection.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import pytest

from src.driver import Problem, load_mesh, parse_args

PROBLEM = Problem("demo", "demo problem", lambda: None, lambda m: None, "unit_square:4", 2.0)


def test_defaults_come_from_the_problem():
    args = parse_args(PROBLEM, [])
    assert (args.mesh, args.t_end) == ("unit_square:4", 2.0)
    assert (args.flux, args.limiter, args.gradient, args.integrator) == (
        "roe",
        "barth_jespersen",
        "least_squares",
        "rk2",
    )
    assert args.dt is None and not args.no_plot


def test_algorithm_options_are_overridable():
    argv = "--flux hllc --limiter venkatakrishnan --gradient green_gauss --integrator euler"
    argv += " --cfl 0.3 --dt 1e-3 --t-end 0.1 --no-plot"
    args = parse_args(PROBLEM, argv.split())
    assert (args.flux, args.limiter, args.gradient, args.integrator) == (
        "hllc",
        "venkatakrishnan",
        "green_gauss",
        "euler",
    )
    assert (args.cfl, args.dt, args.t_end, args.no_plot) == (0.3, 1e-3, 0.1, True)


@pytest.mark.parametrize("bad", [["--flux", "ausm"], ["--integrator", "rk4"], ["--limiter", "x"]])
def test_unsupported_algorithm_is_rejected(bad):
    with pytest.raises(SystemExit):
        parse_args(PROBLEM, bad)


def test_load_mesh_builtin_square():
    assert load_mesh("unit_square:3").nelem == 18
    assert load_mesh("unit_square").nelem == 800  # default 20 x 20 x 2


def test_load_mesh_missing_file_explains_the_alternatives():
    with pytest.raises(FileNotFoundError, match="unit_square"):
        load_mesh("data/does_not_exist.msh")
