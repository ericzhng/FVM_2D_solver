"""Communicator abstraction for distributed-memory runs.

The solver only needs four operations, so any object providing them works:

* ``rank`` / ``size``
* ``allreduce_min(x)``: global minimum of a float (CFL time step)
* ``alltoall(list)``: one picklable object to / from every rank (setup only)
* ``exchange(U, send, recv)``: refresh the ghost rows of ``U`` in place
* ``gather(obj)``: collect one object per rank on rank 0 (``None`` elsewhere)

:class:`SerialComm` is the default; :class:`MPIComm` wraps mpi4py (imported
lazily, so the package works without MPI installed).
"""

import numpy as np


class SerialComm:
    """Single-process communicator (no communication)."""

    rank = 0
    size = 1

    def allreduce_min(self, x):
        return x

    def alltoall(self, objs):
        return list(objs)

    def exchange(self, U, send, recv):
        pass

    def gather(self, obj):
        return [obj]


class MPIComm:
    """Thin mpi4py wrapper implementing the communicator interface."""

    def __init__(self, comm=None):
        from mpi4py import MPI

        self._MPI = MPI
        self._comm = MPI.COMM_WORLD if comm is None else comm
        self.rank = self._comm.Get_rank()
        self.size = self._comm.Get_size()
        self._buffers = {}  # per-neighbour receive buffers, reused across calls

    def allreduce_min(self, x):
        return self._comm.allreduce(x, op=self._MPI.MIN)

    def alltoall(self, objs):
        return self._comm.alltoall(list(objs))

    def gather(self, obj):
        return self._comm.gather(obj, root=0)

    def exchange(self, U, send, recv):
        """Non-blocking point-to-point halo exchange.

        Args:
            U: State array ``(nlocal, nvars)``; ghost rows are overwritten.
            send: ``{rank: local row indices}`` of owned rows to send.
            recv: ``{rank: local row indices}`` of ghost rows to receive.
        """
        nvars = U.shape[1]
        requests = []
        for q, idx in recv.items():
            buf = self._buffers.get((q, nvars))
            if buf is None or buf.shape[0] != len(idx):
                buf = self._buffers[(q, nvars)] = np.empty((len(idx), nvars))
            requests.append(self._comm.Irecv(buf, source=q, tag=17))
        outgoing = []
        for q, idx in send.items():
            sbuf = np.ascontiguousarray(U[idx])
            outgoing.append(sbuf)  # keep alive until Waitall
            requests.append(self._comm.Isend(sbuf, dest=q, tag=17))
        self._MPI.Request.Waitall(requests)
        for q, idx in recv.items():
            U[idx] = self._buffers[(q, nvars)]


def get_comm():
    """Return an :class:`MPIComm` when mpi4py is importable, else :class:`SerialComm`."""
    try:
        return MPIComm()
    except ImportError:
        return SerialComm()
