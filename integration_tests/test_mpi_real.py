r"""Module: integration_tests.test_mpi_real

Description: pytest wrapper that runs ``utility/mpi_parity_check.py`` under mpiexec with 2 and 3
    ranks. The test is skipped when mpi4py or mpiexec is not available.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("mpi4py")

MPIEXEC = shutil.which("mpiexec") or shutil.which("mpirun")
if MPIEXEC is None and os.name == "nt":
    candidate = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
    MPIEXEC = candidate if os.path.exists(candidate) else None


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not found")
@pytest.mark.parametrize("ranks", [2, 3])
def test_real_mpi_matches_serial(ranks):
    assert MPIEXEC is not None  # guaranteed by skipif
    script = os.path.join(os.path.dirname(__file__), "..", "utility", "mpi_parity_check.py")
    env = dict(os.environ, NUMBA_NUM_THREADS="1")
    proc = subprocess.run(
        [MPIEXEC, "-n", str(ranks), sys.executable, script],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PARITY PASSED" in proc.stdout
