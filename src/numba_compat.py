r"""Module: src.numba_compat

Description: Import shim for ``prange``. Type checkers see ``range``, which they accept as an
    iterable. At runtime the name is ``numba.prange``, so parallel loops are unchanged.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    prange = range
else:
    from numba import prange

__all__ = ["prange"]
