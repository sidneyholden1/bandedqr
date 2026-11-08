# Developer build.  Compiles the C kernel in place inside the package so that
# `import bandedqr` works straight from a checkout (no install step).  For a real
# install, `pip install -e .` runs the same compile through setuptools.
CC      ?= cc
SRC      = src/bandedqr/csrc
PKG      = src/bandedqr
PY      ?= python3

UNAME   := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  LIB    = $(PKG)/libbandedqr.dylib
  SHARED = -dynamiclib
  # universal2 so the lib loads under both native arm64 and x86_64 (Rosetta)
  # Python.  NOTE: AVX2 is *slower* under Rosetta (it translates wide vectors
  # poorly), so the x86_64 slice stays at SSE2 baseline -- measured fastest
  # under Rosetta.  The arm64 slice gets apple-m1 tuning; run a native arm64
  # Python to use it (2-3x faster than the Rosetta path on wide bands).
  ARCH   = -arch arm64 -Xarch_arm64 -mcpu=apple-m1 -arch x86_64
else
  LIB    = $(PKG)/libbandedqr.so
  SHARED = -shared
  ARCH   =
endif

OPT     = -O3 -funroll-loops -ffp-contract=fast -fPIC -Wall -Wextra
CFLAGS  = $(OPT) $(ARCH)

all: $(LIB)

$(LIB): $(SRC)/bandedqr.c $(SRC)/bandedqr.h $(SRC)/bandedqr_impl.h
	$(CC) $(CFLAGS) $(SHARED) -o $@ $(SRC)/bandedqr.c -lpthread -lm

# Single-threaded build -- no pthread/C11-atomics dependency (portability check).
nothreads: $(SRC)/bandedqr.c
	$(CC) $(CFLAGS) -DBANDEDQR_NO_THREADS $(SHARED) -o $(LIB) $(SRC)/bandedqr.c -lm

test: $(LIB)
	$(PY) -m pytest tests -q

bench: $(LIB)
	$(PY) bench/bench_cpu.py

bench-gpu: $(LIB)
	$(PY) bench/bench_gpu.py

# Regenerate the single-file distribution from the package sources.
standalone: $(LIB)
	$(PY) tools/make_standalone.py

# Diagnostic: report which loops the vectoriser took.
vecreport: $(SRC)/bandedqr.c
	$(CC) $(OPT) -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
	      -c $(SRC)/bandedqr.c -o /dev/null

clean:
	rm -f $(PKG)/libbandedqr.dylib $(PKG)/libbandedqr.so $(PKG)/_libbandedqr*.so
	rm -rf build dist src/*.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: all clean vecreport test bench bench-gpu standalone nothreads
