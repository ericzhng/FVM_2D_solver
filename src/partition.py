"""Domain decomposition: cell partitioning, halo layers and exchange plans.

Every rank reads and analyses the *global* mesh (O(N) memory per rank, fine up
to a few million cells) and then keeps only its subdomain:

    owned cells  +  halo ring 1  +  halo ring 2

Two rings make one exchange of ``U`` per residual evaluation sufficient:
gradients and limiters are computed on owned + ring-1 cells (they read ring 2),
and fluxes of faces touching an owned cell read ring-1 gradients/limiters.
Nothing else is computed (ring-2 gradients and ring-only faces are skipped). Local cells are kept
in ascending *global* id order, so the owner of every shared face (lowest local
index) is the same cell as in a serial run, and results are reproducible
independent of the rank count (up to round-off in the time-step reduction).

Ring-2 cells are only read, never trusted: their own residual is discarded and
their faces towards missing cells are treated as transmissive boundaries.
"""

from dataclasses import dataclass, field

import numpy as np

from src.mesh import FVMGeometry

_TRANSMISSIVE = 4


def partition_rcb(centroids: np.ndarray, nparts: int) -> np.ndarray:
    """Recursive coordinate bisection into ``nparts`` balanced parts.

    Args:
        centroids: Cell centroids ``(nelem, >=2)``.
        nparts: Number of parts (any positive integer <= ``nelem``).

    Returns:
        Part index of every cell, shape ``(nelem,)`` (deterministic).
    """
    nelem = len(centroids)
    if nparts > nelem:
        raise ValueError(f"Cannot split {nelem} cells into {nparts} parts.")
    part = np.zeros(nelem, dtype=np.int32)

    def split(ids, n, offset):
        if n == 1:
            part[ids] = offset
            return
        n_left = n // 2
        pts = centroids[ids, :2]
        axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
        order = ids[np.argsort(pts[:, axis], kind="stable")]
        k = int(round(len(ids) * n_left / n))
        split(order[:k], n_left, offset)
        split(order[k:], n - n_left, offset + n_left)

    split(np.arange(nelem), nparts, 0)
    return part


@dataclass
class Domain:
    """Everything one rank needs to advance its subdomain.

    Attributes:
        geo: Local :class:`FVMGeometry` (owned + halo cells).
        bc_type, bc_value: Local per-face boundary data.
        global_ids: Global cell id of every local cell (ascending).
        owned_idx: Local indices of owned cells.
        ghost_idx: Local indices of halo cells.
        send, recv: ``{rank: local indices}`` halo exchange plan.
        comm: Communicator.
    """

    geo: FVMGeometry
    bc_type: np.ndarray
    bc_value: np.ndarray
    global_ids: np.ndarray
    owned_idx: np.ndarray
    ghost_idx: np.ndarray
    send: dict = field(default_factory=dict)
    recv: dict = field(default_factory=dict)
    comm: object = None

    @property
    def all_owned(self) -> bool:
        return self.ghost_idx.size == 0

    def exchange(self, U: np.ndarray) -> None:
        """Refresh ghost rows of ``U`` from their owners (in place)."""
        if self.send or self.recv:
            self.comm.exchange(U, self.send, self.recv)

    def gather(self, U: np.ndarray, nelem: int):
        """Assemble the global state on rank 0 (``None`` on other ranks)."""
        if self.all_owned:
            return U.copy()
        parts = self.comm.gather((self.global_ids[self.owned_idx], U[self.owned_idx]))
        if parts is None:
            return None
        full = np.empty((nelem, U.shape[1]))
        for ids, rows in parts:
            full[ids] = rows
        return full


def _ring(neighbors: np.ndarray, members: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    """Cells adjacent to ``members`` that are not in ``exclude`` (boolean mask in)."""
    nb = neighbors[members]
    nb = np.unique(nb[nb >= 0])
    return nb[~exclude[nb]]


def build_domain(geo, bc_type, bc_value, comm, part=None) -> Domain:
    """Cut the global geometry down to this rank's subdomain.

    Args:
        geo: Global :class:`FVMGeometry`.
        bc_type, bc_value: Global per-face boundary arrays.
        comm: Communicator (``rank``, ``size``, ``alltoall``).
        part: Optional precomputed partition; defaults to RCB.

    Returns:
        The :class:`Domain` of ``comm.rank``. With one rank it is the whole mesh.
    """
    nelem, nfaces = geo.neighbors.shape
    if comm.size == 1:
        everything = np.arange(nelem)
        return Domain(geo, bc_type, bc_value, everything, everything, np.empty(0, int), comm=comm)

    if part is None:
        part = partition_rcb(geo.centroids, comm.size)
    mine = part == comm.rank
    owned = np.flatnonzero(mine)
    in_set = mine.copy()
    ring1 = _ring(geo.neighbors, owned, in_set)
    in_set[ring1] = True
    ring2 = _ring(geo.neighbors, ring1, in_set)
    in_set[ring2] = True
    local = np.flatnonzero(in_set)  # ascending global ids
    g2l = np.full(nelem, -1, dtype=np.int64)
    g2l[local] = np.arange(local.size)

    nb_global = geo.neighbors[local]
    neighbors = np.where(nb_global >= 0, g2l[np.maximum(nb_global, 0)], -1)

    is_owned = mine[local]
    owned_idx = np.flatnonzero(is_owned)
    ghost_idx = np.flatnonzero(~is_owned)
    ring1_local = g2l[ring1]
    grad_cells = np.sort(np.concatenate([owned_idx, ring1_local]))

    # Unique-face table (owner = lowest local flat index, as in the serial run),
    # keeping only faces that touch an owned cell: the others are never needed.
    _, first, inverse = np.unique(
        geo.face_id[local].reshape(-1), return_index=True, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    flat = np.arange(local.size * nfaces)
    touches_owned = np.zeros(first.size, dtype=bool)
    touches_owned[inverse[np.repeat(is_owned, nfaces)]] = True
    new_id = np.cumsum(touches_owned) - 1
    keep = np.flatnonzero(touches_owned)
    face_id = np.where(touches_owned[inverse], new_id[inverse], 0)
    face_sign = np.where(first[inverse] == flat, 1, -1)

    local_geo = FVMGeometry(
        neighbors=np.ascontiguousarray(neighbors),
        normals=np.ascontiguousarray(geo.normals[local]),
        areas=np.ascontiguousarray(geo.areas[local]),
        midpoints=np.ascontiguousarray(geo.midpoints[local]),
        face_r=np.ascontiguousarray(geo.face_r[local]),
        centroids=np.ascontiguousarray(geo.centroids[local]),
        volumes=np.ascontiguousarray(geo.volumes[local]),
        weights=np.ascontiguousarray(geo.weights[local]),
        nonorth=np.ascontiguousarray(geo.nonorth[local]),
        lsq=np.ascontiguousarray(geo.lsq[local]),
        face_id=face_id.reshape(local.size, nfaces),
        face_sign=face_sign.reshape(local.size, nfaces),
        owner_cell=first[keep] // nfaces,
        owner_local=first[keep] % nfaces,
        grad_cells=grad_cells,
        owned_cells=owned_idx,
    )

    local_bc_type = bc_type[local].copy()
    local_bc_type[(neighbors == -1) & (local_bc_type == 0)] = _TRANSMISSIVE
    local_bc_value = bc_value[local].copy()

    # Exchange plan: tell each owner which of its cells we need (ascending ids).
    ghost_global = local[ghost_idx]
    ghost_owner = part[ghost_global]
    wanted = [ghost_global[ghost_owner == q] for q in range(comm.size)]
    requested = comm.alltoall(wanted)  # requested[q]: global ids rank q wants from us
    recv = {q: ghost_idx[ghost_owner == q] for q in range(comm.size) if (ghost_owner == q).any()}
    send = {}
    for q, ids in enumerate(requested):
        if len(ids):
            rows = g2l[np.asarray(ids)]
            if (rows < 0).any() or not is_owned[rows].all():
                raise RuntimeError("Halo request for a cell this rank does not own.")
            send[q] = rows

    return Domain(
        local_geo,
        local_bc_type,
        local_bc_value,
        local,
        owned_idx,
        ghost_idx,
        send,
        recv,
        comm,
    )
