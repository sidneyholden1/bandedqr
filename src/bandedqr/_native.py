"""ctypes binding to the C kernel.

The library is loaded from the package directory, accepting either the name the
developer Makefile produces (``libbandedqr.dylib`` / ``libbandedqr.so``) or the one
setuptools produces for an installed wheel (``_libbandedqr.<abi>.so``).

Every C routine exists in a ``_d`` (float64) and a ``_z`` (complex128) flavour.
Real arrays are passed as ``POINTER(c_double)``; complex128 arrays are passed by
address as ``c_void_p``, since numpy's interleaved re/im layout matches C's
``double _Complex`` exactly.
"""
import ctypes as ct
import glob
import os

import numpy as np

__all__ = [
    "lib", "has_threads", "suffix_for", "ptr_for", "as_ptr", "as_cptr",
    "is_complex", "dtype_for", "LibraryNotFound",
]

_dbl = ct.POINTER(ct.c_double)
_vp = ct.c_void_p
_int = ct.c_int
_intp = ct.POINTER(ct.c_int)


class LibraryNotFound(RuntimeError):
    """The compiled kernel could not be located."""


def _candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("BANDEDQR_LIBRARY")
    if env:
        yield env
    for pattern in ("libbandedqr.dylib", "libbandedqr.so", "_libbandedqr*.so",
                    "_libbandedqr*.dylib", "_libbandedqr*.pyd"):
        yield from sorted(glob.glob(os.path.join(here, pattern)))


def _load():
    tried = []
    for path in _candidates():
        try:
            return ct.CDLL(path)
        except OSError as exc:  # wrong architecture, missing deps, ...
            tried.append(f"  {path}: {exc}")
    raise LibraryNotFound(
        "could not load the bandedqr C kernel.\n"
        "Build it with `make` (in a checkout) or `pip install -e .`, or point\n"
        "BANDEDQR_LIBRARY at an existing shared library.\n"
        + ("Tried:\n" + "\n".join(tried) if tried else "No candidate files found.")
    )


lib = _load()


def _bind():
    for suf, arr in (("_d", _dbl), ("_z", _vp)):
        def fn(name):
            return getattr(lib, "bandedqr_" + name + suf)

        f = fn("factor");           f.restype = _vp; f.argtypes = [arr, _int, _int, _int, _int]
        f = fn("factor_mt");        f.restype = _vp; f.argtypes = [arr, _int, _int, _int, _int, _int]
        f = fn("factor_hh");        f.restype = _vp; f.argtypes = [arr, _int, _int, _int, _int]
        f = fn("factor_ab");        f.restype = _vp; f.argtypes = [arr, _int, _int, _int, _int, _intp, _int, arr]
        f = fn("solve_ab");         f.restype = ct.c_double; f.argtypes = [_vp, arr, arr]
        f = fn("solve_many_ab");    f.restype = None; f.argtypes = [_vp, arr, _int, arr, _dbl]
        f = fn("solve");            f.restype = ct.c_double; f.argtypes = [_vp, arr, arr]
        f = fn("solve_many");       f.restype = None; f.argtypes = [_vp, arr, _int, arr, _dbl]
        f = fn("apply_Qt");         f.restype = None; f.argtypes = [_vp, arr]
        f = fn("apply_Q");          f.restype = None; f.argtypes = [_vp, arr]
        f = fn("solve_R");          f.restype = None; f.argtypes = [_vp, arr, arr]
        f = fn("solve_Rt");         f.restype = None; f.argtypes = [_vp, arr, arr]
        f = fn("apply_P");          f.restype = None; f.argtypes = [arr, _int, arr, _int, arr]
        f = fn("band_matvec");      f.restype = None; f.argtypes = [arr, _int, _int, _int, _int, arr, arr]
        f = fn("band_residual_batched"); f.restype = None
        f.argtypes = [arr, _int, _int, _int, _int, _int, arr, arr, arr]
        f = fn("get_R");            f.restype = None; f.argtypes = [_vp, arr]
        f = fn("free");             f.restype = None; f.argtypes = [_vp]
    lib.bandedqr_has_threads.restype = _int
    lib.bandedqr_has_threads.argtypes = []


_bind()


def has_threads():
    """True if the kernel was built with pthread support (``factor_mt`` is real)."""
    return bool(lib.bandedqr_has_threads())


def as_ptr(a):
    """Pointer to a float64 array."""
    return a.ctypes.data_as(_dbl)


def as_cptr(a):
    """Address of a complex128 array (interleaved re/im == C double _Complex)."""
    return a.ctypes.data_as(_vp)


def is_complex(a):
    dt = getattr(a, "dtype", None)
    if dt is None:
        dt = np.asarray(a).dtype
    return np.issubdtype(dt, np.complexfloating)


def dtype_for(a):
    """complex128 if `a` is complex, else float64 -- the dtype follows the input."""
    return np.complex128 if is_complex(a) else np.float64


def suffix_for(complex_):
    return "_z" if complex_ else "_d"


def ptr_for(complex_):
    return as_cptr if complex_ else as_ptr
