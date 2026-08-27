#include "bandqr.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>

/* Band buffer layout, per row i:
 *   offset 0      .. kl-1        subdiagonals  (columns i-kl .. i-1)
 *   offset kl                    the diagonal  (column i)
 *   offset kl+1   .. kl+kuR      superdiagonals of R (columns i+1 .. i+kuR)
 *   offset kl+kuR+1              one "guard" slot
 *
 * An adjacent-row rotation zeroing (q,jcol) writes row q-1 out to column q+kuR,
 * exactly ONE past that row's nominal kuR extent; the single guard slot absorbs
 * it.  Minimal stride (W = 2kl+ku+2) matters: wide-band factorisation is
 * bandwidth bound.  Column of (row i, offset o) is  j = i - kl + o.
 */
#define IDX(qr, i, j)  ((size_t)(i) * (qr)->W + ((j) - (i) + (qr)->kl))

static inline void cpu_relax(void)
{
#if defined(__x86_64__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

static inline int ilast_of(int jcol, int kl, int m)
{
    int v = jcol + kl;
    return v < m - 1 ? v : m - 1;
}

/* ---- float64 instantiation ---- */
#define SCALAR double
#define REAL   double
#define SUF(x) x##_d
#define CONJ(x) (x)
#define ABS2(x) ((x) * (x))
#define RE(x)  (x)
#include "bandqr_impl.h"
#undef SCALAR
#undef REAL
#undef SUF
#undef CONJ
#undef ABS2
#undef RE

/* ---- complex128 instantiation ---- */
#define SCALAR double _Complex
#define REAL   double
#define SUF(x) x##_z
#define CONJ(x) conj(x)
#define ABS2(x) (creal(x) * creal(x) + cimag(x) * cimag(x))
#define RE(x)  creal(x)
#include "bandqr_impl.h"
#undef SCALAR
#undef REAL
#undef SUF
#undef CONJ
#undef ABS2
#undef RE
