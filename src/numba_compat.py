"""Typing shim for numba.

numba's ``prange`` has no ``__iter__`` in its type stubs, so type checkers flag
every ``for i in prange(n)``. At runtime this is exactly ``numba.prange``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    prange = range
else:
    from numba import prange

__all__ = ["prange"]
