r"""Module: src.parallel

Description: Communicator interface for distributed runs: ``rank``, ``size``, ``allreduce_min``,
    ``alltoall``, ``exchange`` (halo refresh) and ``gather``. ``SerialComm`` is the no-op
    default. ``MPIComm`` wraps mpi4py with non-blocking halo exchange. ``get_comm`` returns
    ``MPIComm`` when mpi4py is importable. ``Communicator`` is the typing protocol the solver
    depends on.

Example:
    >>> from src.parallel import SerialComm
    >>> comm = SerialComm()
    >>> comm.rank, comm.size
    (0, 1)
    >>> comm.allreduce_min(3.5)
    3.5

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

from typing import Any, Protocol

import numpy as np


class Communicator(Protocol):
    """Interface the solver needs from a communicator (see module docstring)."""

    rank: int
    size: int

    def allreduce_min(self, x: float) -> float: ...

    def alltoall(self, objs: Any) -> list[Any]: ...

    def exchange(self, U: np.ndarray, send: dict, recv: dict) -> None: ...

    def gather(self, obj: Any) -> list[Any] | None: ...


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
