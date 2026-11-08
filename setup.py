"""Build the C kernel as a shared library shipped inside the package.

It is loaded with ctypes rather than imported, so it deliberately has no
``PyInit`` -- setuptools is used purely as a portable C build system that knows
each platform's shared-library flags.
"""
import platform
import sys

from setuptools import Extension, setup

extra_compile_args = ["-O3", "-funroll-loops", "-ffp-contract=fast"]
extra_link_args = []
define_macros = []

if sys.platform == "darwin":
    # universal2 when building on Apple silicon, so the wheel also loads under
    # an x86_64 (Rosetta) Python.  AVX2 is *slower* under Rosetta -- it
    # translates wide vectors poorly -- so the x86_64 slice stays at baseline.
    if platform.machine() == "arm64":
        arch = ["-arch", "arm64", "-Xarch_arm64", "-mcpu=apple-m1", "-arch", "x86_64"]
        extra_compile_args += arch
        extra_link_args += [a for a in arch if a != "-Xarch_arm64"
                            and a != "-mcpu=apple-m1"]
elif sys.platform.startswith("win"):
    # no pthreads / C11 atomics on MSVC: build the serial kernel
    define_macros.append(("BANDEDQR_NO_THREADS", "1"))
    extra_compile_args = ["/O2"]

setup(
    ext_modules=[
        Extension(
            "bandedqr._libbandedqr",
            sources=["src/bandedqr/csrc/bandedqr.c"],
            include_dirs=["src/bandedqr/csrc"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=define_macros,
            py_limited_api=False,
        )
    ],
)
