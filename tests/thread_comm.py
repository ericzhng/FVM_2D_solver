"""In-process stand-in for MPI: ranks are threads sharing a barrier.

Implements the communicator interface of :mod:`src.parallel` so decomposition,
halo exchange and the distributed solver can be tested without an MPI runtime.
Use with ``NUMBA_DISABLE_JIT=1`` (numba's default threading layer is not safe
for concurrent parallel regions from several Python threads).
"""

import threading


class _Shared:
    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.mail = {}


class ThreadComm:
    def __init__(self, shared, rank):
        self._s = shared
        self.rank = rank
        self.size = shared.size

    def allreduce_min(self, x):
        s = self._s
        s.slots[self.rank] = x
        s.barrier.wait()
        out = min(s.slots)
        s.barrier.wait()
        return out

    def alltoall(self, objs):
        s = self._s
        for q, o in enumerate(objs):
            s.mail[(self.rank, q)] = o
        s.barrier.wait()
        out = [s.mail[(q, self.rank)] for q in range(self.size)]
        s.barrier.wait()
        return out

    def gather(self, obj):
        s = self._s
        s.slots[self.rank] = obj
        s.barrier.wait()
        out = list(s.slots) if self.rank == 0 else None
        s.barrier.wait()
        return out

    def exchange(self, U, send, recv):
        s = self._s
        for q, idx in send.items():
            s.mail[("halo", self.rank, q)] = U[idx].copy()
        s.barrier.wait()
        for q, idx in recv.items():
            U[idx] = s.mail[("halo", q, self.rank)]
        s.barrier.wait()


def run_ranks(size, fn):
    """Run ``fn(comm)`` on ``size`` thread-ranks; return the list of results."""
    shared = _Shared(size)
    results = [None] * size
    errors = []

    def target(rank):
        try:
            results[rank] = fn(ThreadComm(shared, rank))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)
            shared.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results
