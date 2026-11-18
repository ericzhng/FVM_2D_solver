"""Run the real-MPI parity check when mpi4py and mpiexec are available."""

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
    script = os.path.join(os.path.dirname(__file__), "mpi_parity.py")
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
