"""bandqr -- fast, deterministic banded and almost-banded QR.

A small C kernel (no LAPACK, no pivoting, no densifying) with a Python front
end, plus a batched Metal backend for the many-independent-systems regime.

    >>> from bandqr import solve
    >>> x = solve(A, b)          # structure detected, factorisation chosen

Why QR rather than LU: the fill is *deterministic* -- ``ku + kl`` upper, always,
with no pivoting and no value-dependent growth -- the factorisation is
unconditionally backward stable, and for narrow bands it factors as fast as or
faster than LAPACK's band LU.  A predictable, fast *factorisation* is the point:
it makes re-factorising often (adaptive timestepping, parameter sweeps) cheap.

Main entry points
-----------------
``solve(A, b)``, ``factorize(A)``
    Single systems, dense or ``scipy.sparse``, real or complex.
``solve_batched(ab, b, kl, ku)``
    Many same-shape systems at once, on the GPU when one is available.
``BandQR``, ``RideQR``, ``CollapseQR``, ...
    The factorisations themselves, for factor-once/solve-many.
``detect_structure``, ``detect_recurrence``
    The structure inference the dispatch is built on.
"""

from ._band import (band_matvec, band_to_dense, band_transpose, dense_to_band,
                    from_scipy_band, to_band, to_scipy_band)
from ._native import LibraryNotFound, has_threads
from .almost import (BorderedQR, CollapseBothQR, CollapseColsQR, CollapseQR,
                     RideQR, RideQRRect, RideTauQR, solve_bordered, solve_ride,
                     solve_precond_rows, solve_with_tau_columns)
from .api import METHODS, factorize, solve
from .banded import LQ, BandQR
from .batched import available_backends, solve_batched
from .recurrence import (apply_preconditioner, build_preconditioner,
                         detect_recurrence, recurrence_residual)
from .refine import (cond1_estimate, compensated_residual, condest,
                     iterative_refinement)
from .structure import Structure, StructureWarning, detect_structure

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # front end
    "solve", "factorize", "METHODS", "solve_batched", "available_backends",
    # factorisations
    "BandQR", "LQ", "RideQR", "RideQRRect", "RideTauQR",
    "CollapseQR", "CollapseColsQR", "CollapseBothQR", "BorderedQR",
    # one-shot helpers
    "solve_ride", "solve_precond_rows", "solve_bordered",
    "solve_with_tau_columns",
    # structure and recurrence
    "detect_structure", "Structure", "StructureWarning",
    "detect_recurrence", "recurrence_residual",
    "build_preconditioner", "apply_preconditioner",
    # accuracy
    "iterative_refinement", "compensated_residual", "cond1_estimate", "condest",
    # band storage
    "dense_to_band", "to_band", "band_to_dense", "band_transpose",
    "band_matvec", "to_scipy_band", "from_scipy_band",
    # environment
    "has_threads", "LibraryNotFound",
]


def gpu_available():
    """True if the Metal batched backend can run in this process."""
    from . import gpu
    return gpu.is_available()
