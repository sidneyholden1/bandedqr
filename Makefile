CC  = clang
LIB = libbandqr.dylib

# universal2 so the lib loads under both native arm64 and x86_64 (Rosetta) Python.
# NOTE: AVX2 is *slower* under Rosetta (it translates wide vectors poorly), so the
# x86_64 slice stays at SSE2 baseline (2 doubles/vector) -- measured fastest under
# Rosetta.  The arm64 slice gets apple-m1 tuning; run a native arm64 Python to use
# it (2-3x faster than the Rosetta path on wide bands).
OPT    = -O3 -funroll-loops -ffp-contract=fast -fPIC -Wall -Wextra
ARCH   = -arch arm64 -Xarch_arm64 -mcpu=apple-m1 -arch x86_64
CFLAGS = $(OPT) $(ARCH)

$(LIB): bandqr.c bandqr.h bandqr_impl.h
	$(CC) $(CFLAGS) -dynamiclib -o $(LIB) bandqr.c

# Diagnostic: report which loops the vectoriser took (x86_64 slice).
vecreport: bandqr.c
	$(CC) $(OPT) -arch x86_64 -mavx2 -mfma -Rpass=loop-vectorize \
	      -Rpass-missed=loop-vectorize -c bandqr.c -o /dev/null

clean:
	rm -f $(LIB)

.PHONY: clean vecreport
