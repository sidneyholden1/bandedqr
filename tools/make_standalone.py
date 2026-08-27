"""Fold bandqr.h + bandqr.c + bandqr.py into one self-compiling file,
bandqr_standalone.py, that can be copy-pasted/emailed to a collaborator.

Run:  python make_standalone.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# C source = header (opaque typedefs + prototypes), then the driver with both
# local includes inlined: bandqr.h once at the top, bandqr_impl.h at each of the
# two instantiation points (it is included once per scalar type).
h    = open(os.path.join(HERE, "bandqr.h")).read()
impl = open(os.path.join(HERE, "bandqr_impl.h")).read()
c = open(os.path.join(HERE, "bandqr.c")).read()
c = c.replace('#include "bandqr.h"', "/* header inlined above */")
c = c.replace('#include "bandqr_impl.h"', impl)        # both occurrences
csrc = h + "\n" + c

# Python API = everything in bandqr.py AFTER the CDLL-load line.
py = open(os.path.join(HERE, "bandqr.py")).read()
marker = '_lib = ct.CDLL(os.path.join(os.path.dirname(__file__), "libbandqr.dylib"))'
body = py.split(marker, 1)[1]

PREAMBLE = '''"""Self-contained banded / almost-banded Givens QR solver (single file).

The C kernel is embedded below and compiled on first import -- needs a C
compiler (cc/clang/gcc) and numpy; scipy only for the sparse helpers. Copy this
one file anywhere and:

    from bandqr_standalone import BandQR, dense_to_band, solve_bordered
    qr = BandQR(dense_to_band(A, kl, ku), kl, ku)
    x  = qr.solve(b)

Set CC=... to choose a compiler, BANDQR_CACHE=... to choose the build dir.
"""
import ctypes as ct
import hashlib, os, platform, subprocess, sys, tempfile
import numpy as np

_C_SOURCE = r"""__CSRC__"""

def _build():
    tag = hashlib.sha1(_C_SOURCE.encode()).hexdigest()[:12]
    cache = os.environ.get("BANDQR_CACHE") or os.path.join(tempfile.gettempdir(), "bandqr_build")
    os.makedirs(cache, exist_ok=True)
    extra = []
    if sys.platform == "darwin":
        ext, shared, arch = ".dylib", ["-dynamiclib"], ["-arch", platform.machine()]
        # The Command Line Tools 'cc' often cannot find <stdlib.h> etc. without
        # an explicit SDK path (unlike /usr/bin/cc, the xcrun shim).
        try:
            sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
            if sdk:
                extra = ["-isysroot", sdk]
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        ext, shared, arch = ".so", ["-shared"], []
    else:
        ext, shared, arch = ".dll", ["-shared"], []   # Windows/MSYS: untested
    lib = os.path.join(cache, "libbandqr_%s%s" % (tag, ext))
    if not os.path.exists(lib):
        src = os.path.join(cache, "bandqr_%s.c" % tag)
        open(src, "w").write(_C_SOURCE)
        cc = (os.environ.get("CC") or "cc").split()
        cmd = cc + ["-O3", "-funroll-loops", "-ffp-contract=fast", "-fPIC",
                    "-pthread", *shared, *arch, *extra, "-o", lib, src]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("bandqr: C compilation failed.\\nCommand: %s\\n%s"
                               % (" ".join(cmd), proc.stderr))
    return lib

_lib = ct.CDLL(_build())
'''

out = PREAMBLE.replace("__CSRC__", csrc) + body
dest = os.path.join(HERE, "bandqr_standalone.py")
open(dest, "w").write(out)
print("wrote %s  (%d lines, %.0f KB)" % (dest, out.count("\n") + 1, len(out) / 1024))
