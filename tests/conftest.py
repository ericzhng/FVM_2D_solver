r"""Module: tests.conftest

Description: Shared pytest configuration. It disables the numba JIT for the whole session (the
    fake-rank threads and the pure-Python reference comparisons need it), marks every test under
    ``tests/integration`` with ``integration`` so the layers can also be selected with
    ``-m integration`` / ``-m "not integration"``, and provides a small shared mesh fixture.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import os
from pathlib import Path

import pytest

# Numba reads this once, at its first import: set it before any test module imports ``src`` so
# the session behaves the same whatever the test order or selection.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

INTEGRATION_DIR = Path(__file__).parent / "integration"


def pytest_collection_modifyitems(items):
    for item in items:
        if INTEGRATION_DIR in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def square_mesh():
    """Analysed 6x6 triangle mesh of the unit square (no gmsh needed)."""
    from src.mesh import Mesh

    return Mesh.unit_square(6)
