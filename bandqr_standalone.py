"""Self-contained banded / almost-banded Givens QR solver (single file).

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

_C_SOURCE = r"""/* bandqr -- banded QR factorisation by Givens rotations.
 *
 * For an m x n matrix A with lower bandwidth kl and upper bandwidth ku
 * (m >= n), A = Q R with R upper triangular of upper bandwidth ku+kl and Q held
 * implicitly as plane rotations.  Work is O(n*kl*(kl+ku)) with no dense fill.
 *
 * Two scalar types, generated from one template (bandqr_impl.h): a _d (float64)
 * and a _z (complex128) variant of every routine.  The handle is opaque; pick
 * the variant by your data's dtype.
 */
#ifndef BANDQR_H
#define BANDQR_H

typedef struct band_qr_d band_qr_d;   /* opaque, float64    */
typedef struct band_qr_z band_qr_z;   /* opaque, complex128 */

/* ---- float64 ---- */
band_qr_d *bandqr_factor_d(const double *ab, int m, int n, int kl, int ku);
band_qr_d *bandqr_factor_mt_d(const double *ab, int m, int n, int kl, int ku, int nthreads);
band_qr_d *bandqr_factor_hh_d(const double *ab, int m, int n, int kl, int ku);
band_qr_d *bandqr_factor_ab_d(const double *ab, int m, int n, int kl, int ku,
                              const int *dr, int r, const double *L);
double bandqr_solve_ab_d(const band_qr_d *qr, const double *b, double *x);
double bandqr_solve_d(const band_qr_d *qr, const double *b, double *x);
void bandqr_solve_many_d(const band_qr_d *qr, const double *B, int nrhs, double *X, double *res);
void bandqr_apply_Qt_d(const band_qr_d *qr, double *v);
void bandqr_solve_R_d(const band_qr_d *qr, const double *y, double *x);
void bandqr_apply_P_d(const double *a, int p, const double *y, int n, double *x);
void bandqr_get_R_d(const band_qr_d *qr, double *out);
void bandqr_free_d(band_qr_d *qr);

/* ---- complex128 ---- */
band_qr_z *bandqr_factor_z(const double _Complex *ab, int m, int n, int kl, int ku);
band_qr_z *bandqr_factor_mt_z(const double _Complex *ab, int m, int n, int kl, int ku, int nthreads);
band_qr_z *bandqr_factor_hh_z(const double _Complex *ab, int m, int n, int kl, int ku);
band_qr_z *bandqr_factor_ab_z(const double _Complex *ab, int m, int n, int kl, int ku,
                              const int *dr, int r, const double _Complex *L);
double bandqr_solve_ab_z(const band_qr_z *qr, const double _Complex *b, double _Complex *x);
double bandqr_solve_z(const band_qr_z *qr, const double _Complex *b, double _Complex *x);
void bandqr_solve_many_z(const band_qr_z *qr, const double _Complex *B, int nrhs, double _Complex *X, double *res);
void bandqr_apply_Qt_z(const band_qr_z *qr, double _Complex *v);
void bandqr_solve_R_z(const band_qr_z *qr, const double _Complex *y, double _Complex *x);
void bandqr_apply_P_z(const double _Complex *a, int p, const double _Complex *y, int n, double _Complex *x);
void bandqr_get_R_z(const band_qr_z *qr, double _Complex *out);
void bandqr_free_z(band_qr_z *qr);

#endif /* BANDQR_H */

/* header inlined above */
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
/* bandqr_impl.h -- templated band-QR kernel.  Included once per scalar type by
 * bandqr.c, with macros set:
 *     SCALAR   element type (double | double _Complex)
 *     REAL     real magnitude type (double)
 *     SUF(x)   name mangling  (x##_d | x##_z)
 *     CONJ(x)  conjugate      (identity | conj)
 *     ABS2(x)  squared magnitude ((x)*(x) | re*re+im*im)
 * The driver also provides type-independent IDX(qr,i,j), cpu_relax(), ilast_of().
 * No include guard -- this header is meant to be included multiple times.
 *
 * Givens (unitary) rotation zeroing b:  [[c, s], [-conj(s), c]] [a;b] = [r;0],
 * c real >= 0.  For real data conj is identity and this reduces to a plain
 * Givens rotation (the R diagonal may differ in sign from a c=a/t convention,
 * which is immaterial -- Q absorbs it).
 */

struct SUF(band_qr) {
    int  m, n, kl, ku, kuR, W;
    SCALAR *band;
    /* Givens factorisation: Q as a list of plane rotations. */
    REAL   *c;     /* cosines: real for both real and complex rotations */
    SCALAR *s;     /* sines */
    int    *p;     /* upper row index of each rotation */
    long    nrot;
    /* Householder factorisation (reflectors != 0): Q as reflectors.
     * V is n x (kl+1) row-major (V[j][0]=1 implicit), tau length n. */
    SCALAR *V;
    SCALAR *tau;
    int     reflectors;
    /* Almost-banded factorisation (rfill > 0): M = B + U L, Givens rotations
     * stored in c/s/p as usual; U is m x rfill (evolved), L is rfill x n (fixed
     * dense-row out-of-band content).  R = R_band(band) + U L. */
    SCALAR *U;
    SCALAR *L;
    int     rfill;
};

void SUF(bandqr_free)(struct SUF(band_qr) *qr)
{
    if (!qr) return;
    free(qr->band); free(qr->c); free(qr->s); free(qr->p);
    free(qr->V); free(qr->tau);
    free(qr->U); free(qr->L);
    free(qr);
}

static inline void SUF(givens)(SCALAR a, SCALAR b, REAL *c, SCALAR *s)
{
    REAL bb2 = ABS2(b);
    if (bb2 == 0) { *c = 1; *s = 0; return; }
    REAL aa2 = ABS2(a);
    REAL t = sqrt(aa2 + bb2);            /* = sqrt(a^2+b^2) for real (fast) */
    if (aa2 == 0) { *c = 0; *s = CONJ(b) / sqrt(bb2); return; }
    REAL aa = sqrt(aa2);
    *c = aa / t;
    *s = (a / aa) * CONJ(b) / t;
}

/* Apply one rotation to two contiguous, non-overlapping row-strips. */
static inline void SUF(apply_rotation)(SCALAR *restrict rp, SCALAR *restrict rq,
                                       int len, REAL c, SCALAR s)
{
    for (int t = 0; t < len; t++) {
        SCALAR ap = rp[t], aq = rq[t];
        rp[t] = c * ap + s * aq;
        rq[t] = c * aq - CONJ(s) * ap;
    }
}

static void SUF(run_column)(struct SUF(band_qr) *qr, int jcol, long off,
                            _Atomic int *front)
{
    int m = qr->m, n = qr->n, kl = qr->kl, kuR = qr->kuR, W = qr->W;
    SCALAR *band = qr->band;
    int ilast = ilast_of(jcol, kl, m);
    long r = 0;
    for (int q = ilast; q >= jcol + 1; q--, r++) {
        if (front && jcol > 0)
            while (atomic_load_explicit(&front[jcol - 1], memory_order_acquire) > q - 1)
                cpu_relax();
        int     p  = q - 1;
        SCALAR *rp = &band[(size_t)p * W + (jcol - p + kl)];
        SCALAR *rq = rp + (W - 1);
        SCALAR  b  = *rq;
        REAL c; SCALAR s;
        SUF(givens)(*rp, b, &c, &s);
        if (b != 0) {
            int jmax = q + kuR < n - 1 ? q + kuR : n - 1;
            SUF(apply_rotation)(rp, rq, jmax - jcol + 1, c, s);
        }
        qr->c[off + r] = c;
        qr->s[off + r] = s;
        qr->p[off + r] = p;
        if (front)
            atomic_store_explicit(&front[jcol], q, memory_order_release);
    }
    if (front)
        atomic_store_explicit(&front[jcol], -1, memory_order_release);
}

static struct SUF(band_qr) *SUF(alloc_and_load)(const SCALAR *ab, int m, int n,
                                                int kl, int ku, long **off_out,
                                                long *total_out)
{
    if (m < n || kl < 0 || ku < 0) return NULL;
    struct SUF(band_qr) *qr = calloc(1, sizeof *qr);
    if (!qr) return NULL;
    qr->m = m; qr->n = n; qr->kl = kl; qr->ku = ku;
    qr->kuR = ku + kl;
    qr->W   = kl + 1 + qr->kuR + 1;

    long *off = malloc((size_t)(n + 1) * sizeof(long));
    qr->band = calloc((size_t)m * qr->W, sizeof(SCALAR));
    if (!off || !qr->band) { free(off); SUF(bandqr_free)(qr); return NULL; }

    long total = 0;
    for (int j = 0; j < n; j++) {
        off[j] = total;
        int nr = ilast_of(j, kl, m) - j;
        if (nr > 0) total += nr;
    }
    off[n] = total;

    qr->c = malloc((size_t)(total + 1) * sizeof(REAL));
    qr->s = malloc((size_t)(total + 1) * sizeof(SCALAR));
    qr->p = malloc((size_t)(total + 1) * sizeof(int));
    if (!qr->c || !qr->s || !qr->p) { free(off); SUF(bandqr_free)(qr); return NULL; }
    qr->nrot = total;

    int Win = kl + ku + 1;
    for (int i = 0; i < m; i++) {
        int jlo = i - kl < 0 ? 0 : i - kl;
        int jhi = i + ku > n - 1 ? n - 1 : i + ku;
        for (int j = jlo; j <= jhi; j++)
            qr->band[IDX(qr, i, j)] = ab[i * Win + (j - i + kl)];
    }
    *off_out = off; *total_out = total;
    return qr;
}

struct SUF(band_qr) *SUF(bandqr_factor)(const SCALAR *ab, int m, int n, int kl, int ku)
{
    long *off, total;
    struct SUF(band_qr) *qr = SUF(alloc_and_load)(ab, m, n, kl, ku, &off, &total);
    if (!qr) return NULL;
    for (int j = 0; j < n; j++)
        SUF(run_column)(qr, j, off[j], NULL);
    free(off);
    return qr;
}

typedef struct {
    struct SUF(band_qr) *qr;
    const long  *off;
    _Atomic int *front;
    _Atomic int *next_col;
    int          n;
} SUF(worker_arg);

static void *SUF(worker)(void *a)
{
    SUF(worker_arg) *w = a;
    for (;;) {
        int j = atomic_fetch_add_explicit(w->next_col, 1, memory_order_relaxed);
        if (j >= w->n) break;
        SUF(run_column)(w->qr, j, w->off[j], w->front);
    }
    return NULL;
}

struct SUF(band_qr) *SUF(bandqr_factor_mt)(const SCALAR *ab, int m, int n,
                                           int kl, int ku, int nthreads)
{
    if (nthreads == 0) nthreads = 1;          /* auto = serial (see notes) */
    if (nthreads <= 1)
        return SUF(bandqr_factor)(ab, m, n, kl, ku);

    long *off, total;
    struct SUF(band_qr) *qr = SUF(alloc_and_load)(ab, m, n, kl, ku, &off, &total);
    if (!qr) return NULL;

    _Atomic int *front = malloc((size_t)n * sizeof *front);
    if (!front) { free(off); SUF(bandqr_free)(qr); return NULL; }
    for (int j = 0; j < n; j++)
        atomic_init(&front[j], ilast_of(j, kl, m) + 1);

    _Atomic int next_col;
    atomic_init(&next_col, 0);
    if (nthreads > n) nthreads = n;

    SUF(worker_arg) wa = { qr, off, front, &next_col, n };
    pthread_t *th = malloc((size_t)nthreads * sizeof *th);
    for (int t = 0; t < nthreads; t++) pthread_create(&th[t], NULL, SUF(worker), &wa);
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);

    free(th); free(front); free(off);
    return qr;
}

/* Householder band QR.  One reflector per column annihilates all kl
 * subdiagonals at once; applied row-contiguously (weighted sum of row strips,
 * then an outer-product update).  R is the same band format; Q is stored as the
 * reflectors (V, tau) for Q^H application. */
struct SUF(band_qr) *SUF(bandqr_factor_hh)(const SCALAR *ab, int m, int n,
                                           int kl, int ku)
{
    if (m < n || kl < 0 || ku < 0) return NULL;
    struct SUF(band_qr) *qr = calloc(1, sizeof *qr);
    if (!qr) return NULL;
    qr->m = m; qr->n = n; qr->kl = kl; qr->ku = ku;
    qr->kuR = ku + kl;
    qr->W   = kl + 1 + qr->kuR + 1;
    qr->reflectors = 1;
    qr->band = calloc((size_t)m * qr->W, sizeof(SCALAR));
    qr->V    = calloc((size_t)n * (kl + 1), sizeof(SCALAR));
    qr->tau  = calloc((size_t)n, sizeof(SCALAR));
    SCALAR *u = malloc((size_t)(qr->kuR + 1) * sizeof(SCALAR));
    if (!qr->band || !qr->V || !qr->tau || !u) { free(u); SUF(bandqr_free)(qr); return NULL; }

    int Win = kl + ku + 1, W = qr->W, kuR = qr->kuR;
    SCALAR *band = qr->band;
    for (int i = 0; i < m; i++) {
        int jlo = i - kl < 0 ? 0 : i - kl;
        int jhi = i + ku > n - 1 ? n - 1 : i + ku;
        for (int j = jlo; j <= jhi; j++)
            band[IDX(qr, i, j)] = ab[i * Win + (j - i + kl)];
    }

    for (int j = 0; j < n; j++) {
        int ilast = j + kl < m - 1 ? j + kl : m - 1;
        int L = ilast - j + 1;                          /* reflector length    */
        SCALAR *vj = &qr->V[(size_t)j * (kl + 1)];
        vj[0] = 1;

        SCALAR alpha = band[(size_t)j * W + kl];        /* x[0] = diagonal      */
        REAL xnorm2 = 0;                                /* ||subdiagonals||^2   */
        for (int i = 1; i < L; i++) {
            SCALAR xi = band[(size_t)(j + i) * W + (kl - i)];
            xnorm2 += ABS2(xi);
        }
        if (xnorm2 == 0) { qr->tau[j] = 0; continue; }  /* already upper-tri    */

        REAL bnorm = sqrt(ABS2(alpha) + xnorm2);
        REAL signr = RE(alpha) >= 0 ? 1.0 : -1.0;
        SCALAR beta = -signr * bnorm;                   /* real-valued          */
        qr->tau[j] = (beta - alpha) / beta;
        SCALAR scal = 1.0 / (alpha - beta);
        for (int i = 1; i < L; i++)
            vj[i] = band[(size_t)(j + i) * W + (kl - i)] * scal;

        /* apply H = I - tau v v^H to rows j..ilast, columns j..jmax */
        int jmax = j + kuR < n - 1 ? j + kuR : n - 1;
        int wlen = jmax - j + 1;
        for (int t = 0; t < wlen; t++) u[t] = 0;
        for (int i = 0; i < L; i++) {                   /* u = sum conj(v_i) row_i */
            SCALAR *rb = &band[(size_t)(j + i) * W + (kl - i)];
            SCALAR vc = CONJ(vj[i]);
            for (int t = 0; t < wlen; t++) u[t] += vc * rb[t];
        }
        /* Apply H^H = I - conj(tau) v v^H (it is H^H, not H, that zeroes the
         * column for complex data; conj is identity for real). */
        SCALAR ctau = CONJ(qr->tau[j]);
        for (int i = 0; i < L; i++) {
            SCALAR *rb = &band[(size_t)(j + i) * W + (kl - i)];
            SCALAR f = ctau * vj[i];
            for (int t = 0; t < wlen; t++) rb[t] -= f * u[t];
        }
    }
    free(u);
    return qr;
}

/* Almost-banded QR ("ride"): M = B + U L (B in-band, L = r x n out-of-band
 * content of the dense rows dr[0..r-1], U = m x r indicator evolving under the
 * rotations).  Adjacent-row Givens zero the true value M[q,j] = B[q,j] +
 * U[q,:].L[:,j], applied to B's band strips and U's rows; L stays fixed.
 * R = R_band + (Q^H U) L stays almost-banded.  Q is the stored rotations. */
struct SUF(band_qr) *SUF(bandqr_factor_ab)(const SCALAR *ab, int m, int n,
                                           int kl, int ku,
                                           const int *dr, int r, const SCALAR *Lin)
{
    if (m < n || kl < 0 || ku < 0 || r < 1) return NULL;
    struct SUF(band_qr) *qr = calloc(1, sizeof *qr);
    if (!qr) return NULL;
    qr->m = m; qr->n = n; qr->kl = kl; qr->ku = ku;
    qr->kuR = ku + kl;
    qr->W   = kl + 1 + qr->kuR + 1;
    qr->rfill = r;

    long total = 0;
    for (int j = 0; j < n; j++) { int nr = ilast_of(j, kl, m) - j; if (nr > 0) total += nr; }
    qr->band = calloc((size_t)m * qr->W, sizeof(SCALAR));
    qr->c = malloc((size_t)(total + 1) * sizeof(REAL));
    qr->s = malloc((size_t)(total + 1) * sizeof(SCALAR));
    qr->p = malloc((size_t)(total + 1) * sizeof(int));
    qr->U = calloc((size_t)m * r, sizeof(SCALAR));
    qr->L = malloc((size_t)r * n * sizeof(SCALAR));
    if (!qr->band || !qr->c || !qr->s || !qr->p || !qr->U || !qr->L) {
        SUF(bandqr_free)(qr); return NULL;
    }
    qr->nrot = total;

    int Win = kl + ku + 1, W = qr->W, kuR = qr->kuR;
    SCALAR *band = qr->band, *U = qr->U, *L = qr->L;
    for (int i = 0; i < m; i++) {
        int jlo = i - kl < 0 ? 0 : i - kl, jhi = i + ku > n - 1 ? n - 1 : i + ku;
        for (int j = jlo; j <= jhi; j++) band[IDX(qr, i, j)] = ab[i * Win + (j - i + kl)];
    }
    memcpy(L, Lin, (size_t)r * n * sizeof(SCALAR));
    for (int t = 0; t < r; t++) U[(size_t)dr[t] * r + t] = 1;   /* indicator */

    long nrot = 0;
    for (int jcol = 0; jcol < n; jcol++) {
        int ilast = jcol + kl < m - 1 ? jcol + kl : m - 1;
        for (int q = ilast; q >= jcol + 1; q--) {
            int pp = q - 1;
            SCALAR *rp = &band[(size_t)pp * W + (jcol - pp + kl)];
            SCALAR *rq = rp + (W - 1);
            SCALAR ap = *rp, aq = *rq;                  /* M[pp,j], M[q,j] = B + U.L */
            SCALAR *up = &U[(size_t)pp * r], *uq = &U[(size_t)q * r];
            for (int t = 0; t < r; t++) {
                SCALAR ljc = L[(size_t)t * n + jcol];
                ap += up[t] * ljc; aq += uq[t] * ljc;
            }
            REAL c; SCALAR s;
            SUF(givens)(ap, aq, &c, &s);
            int jmax = q + kuR < n - 1 ? q + kuR : n - 1;
            SUF(apply_rotation)(rp, rq, jmax - jcol + 1, c, s);   /* B rows */
            for (int t = 0; t < r; t++) {                          /* U rows */
                SCALAR u1 = up[t], u2 = uq[t];
                up[t] = c * u1 + s * u2;
                uq[t] = c * u2 - CONJ(s) * u1;
            }
            qr->c[nrot] = c; qr->s[nrot] = s; qr->p[nrot] = pp; nrot++;
        }
    }
    qr->nrot = nrot;
    return qr;
}

/* Solve A x = b for an almost-banded factorisation.  Q^H b, then back-solve
 * R x = y with R = R_band + U L: band sweep plus a running r-vector
 * g = sum_{j>i} L[:,j] x[j] carrying the low-rank fill. */
double SUF(bandqr_solve_ab)(const struct SUF(band_qr) *qr, const SCALAR *b, SCALAR *x)
{
    int m = qr->m, n = qr->n, W = qr->W, kuR = qr->kuR, kl = qr->kl, r = qr->rfill;
    const SCALAR *band = qr->band, *U = qr->U, *L = qr->L;
    SCALAR *y = malloc((size_t)m * sizeof(SCALAR));
    memcpy(y, b, (size_t)m * sizeof(SCALAR));
    SUF(bandqr_apply_Qt)(qr, y);                  /* Q^H b (stored rotations) */

    SCALAR *g = calloc(r, sizeof(SCALAR));
    for (int i = n - 1; i >= 0; i--) {
        int jhi = i + kuR < n - 1 ? i + kuR : n - 1;
        const SCALAR *Rrow = &band[(size_t)i * W + (kl + 1)];
        SCALAR bandsum = 0;
        for (int d = 1; d <= jhi - i; d++) bandsum += Rrow[d - 1] * x[i + d];
        SCALAR fillsum = 0, diag = band[(size_t)i * W + kl];
        const SCALAR *Ui = &U[(size_t)i * r];
        for (int t = 0; t < r; t++) {
            fillsum += Ui[t] * g[t];
            diag    += Ui[t] * L[(size_t)t * n + i];
        }
        x[i] = (y[i] - bandsum - fillsum) / diag;
        for (int t = 0; t < r; t++) g[t] += L[(size_t)t * n + i] * x[i];
    }
    free(g);
    REAL res = 0;
    for (int i = n; i < m; i++) res += ABS2(y[i]);
    free(y);
    return sqrt(res);
}

/* v (length m) <- Q^H v (Q^T for real). */
void SUF(bandqr_apply_Qt)(const struct SUF(band_qr) *qr, SCALAR *v)
{
    if (qr->reflectors) {                 /* Householder: apply H_0 .. H_{n-1} */
        int kl = qr->kl, m = qr->m, klp = kl + 1;
        for (int j = 0; j < qr->n; j++) {
            SCALAR tau = qr->tau[j];
            if (tau == 0) continue;
            int L = j + kl < m ? kl + 1 : m - j;
            const SCALAR *vj = &qr->V[(size_t)j * klp];
            SCALAR w = 0;
            for (int i = 0; i < L; i++) w += CONJ(vj[i]) * v[j + i];
            w *= CONJ(tau);                      /* apply H^H (conj for complex) */
            for (int i = 0; i < L; i++) v[j + i] -= vj[i] * w;
        }
        return;
    }
    for (long k = 0; k < qr->nrot; k++) { /* Givens: replay the rotations */
        int p = qr->p[k]; REAL c = qr->c[k]; SCALAR s = qr->s[k];
        SCALAR vp = v[p], vq = v[p + 1];
        v[p]     = c * vp + s * vq;
        v[p + 1] = c * vq - CONJ(s) * vp;
    }
}

void SUF(bandqr_solve_R)(const struct SUF(band_qr) *qr, const SCALAR *y, SCALAR *x)
{
    const SCALAR *restrict band = qr->band;
    int n = qr->n, W = qr->W, kuR = qr->kuR, kl = qr->kl;
    for (int i = n - 1; i >= 0; i--) {
        int jhi = i + kuR < n - 1 ? i + kuR : n - 1;
        const SCALAR *restrict Rrow = &band[(size_t)i * W + (kl + 1)];
        const SCALAR *restrict xj   = &x[i + 1];
        SCALAR sum = 0;
        for (int t = 0; t < jhi - i; t++) sum += Rrow[t] * xj[t];
        x[i] = (y[i] - sum) / band[(size_t)i * W + kl];
    }
}

double SUF(bandqr_solve)(const struct SUF(band_qr) *qr, const SCALAR *b, SCALAR *x)
{
    int m = qr->m, n = qr->n;
    SCALAR *y = malloc((size_t)m * sizeof(SCALAR));
    memcpy(y, b, (size_t)m * sizeof(SCALAR));
    SUF(bandqr_apply_Qt)(qr, y);
    SUF(bandqr_solve_R)(qr, y, x);
    REAL res = 0;
    for (int i = n; i < m; i++) res += ABS2(y[i]);
    free(y);
    return sqrt(res);
}

void SUF(bandqr_solve_many)(const struct SUF(band_qr) *qr, const SCALAR *B,
                            int nrhs, SCALAR *X, double *res)
{
    int m = qr->m, n = qr->n, W = qr->W, kuR = qr->kuR, kl = qr->kl;
    const SCALAR *restrict band = qr->band;
    SCALAR *Y = malloc((size_t)m * nrhs * sizeof(SCALAR));
    memcpy(Y, B, (size_t)m * nrhs * sizeof(SCALAR));

    if (qr->reflectors) {                        /* Y <- Q^H B (Householder) */
        int klp = kl + 1;
        SCALAR *w = malloc((size_t)nrhs * sizeof(SCALAR));
        for (int j = 0; j < n; j++) {
            SCALAR tau = qr->tau[j];
            if (tau == 0) continue;
            int L = j + kl < m ? kl + 1 : m - j;
            const SCALAR *vj = &qr->V[(size_t)j * klp];
            for (int t = 0; t < nrhs; t++) w[t] = 0;
            for (int i = 0; i < L; i++) {
                SCALAR vc = CONJ(vj[i]);
                const SCALAR *yi = &Y[(size_t)(j + i) * nrhs];
                for (int t = 0; t < nrhs; t++) w[t] += vc * yi[t];
            }
            SCALAR ctau = CONJ(tau);             /* apply H^H (conj for complex) */
            for (int t = 0; t < nrhs; t++) w[t] *= ctau;
            for (int i = 0; i < L; i++) {
                SCALAR vi = vj[i];
                SCALAR *yi = &Y[(size_t)(j + i) * nrhs];
                for (int t = 0; t < nrhs; t++) yi[t] -= vi * w[t];
            }
        }
        free(w);
    } else {
        for (long k = 0; k < qr->nrot; k++) {    /* Y <- Q^H B (Givens) */
            int p = qr->p[k]; REAL c = qr->c[k]; SCALAR s = qr->s[k];
            SCALAR *restrict yp = &Y[(size_t)p * nrhs];
            SCALAR *restrict yq = &Y[(size_t)(p + 1) * nrhs];
            for (int t = 0; t < nrhs; t++) {
                SCALAR ap = yp[t], aq = yq[t];
                yp[t] = c * ap + s * aq;
                yq[t] = c * aq - CONJ(s) * ap;
            }
        }
    }
    for (int i = n - 1; i >= 0; i--) {           /* blocked band back-sub */
        int jhi = i + kuR < n - 1 ? i + kuR : n - 1;
        const SCALAR *restrict Rrow = &band[(size_t)i * W + (kl + 1)];
        SCALAR       *restrict xi   = &X[(size_t)i * nrhs];
        const SCALAR *restrict yi   = &Y[(size_t)i * nrhs];
        for (int t = 0; t < nrhs; t++) xi[t] = yi[t];
        for (int d = 1; d <= jhi - i; d++) {
            SCALAR Rid = Rrow[d - 1];
            const SCALAR *restrict xd = &X[(size_t)(i + d) * nrhs];
            for (int t = 0; t < nrhs; t++) xi[t] -= Rid * xd[t];
        }
        SCALAR dinv = 1.0 / band[(size_t)i * W + kl];
        for (int t = 0; t < nrhs; t++) xi[t] *= dinv;
    }
    if (res)
        for (int t = 0; t < nrhs; t++) {
            REAL r = 0;
            for (int i = n; i < m; i++) r += ABS2(Y[(size_t)i * nrhs + t]);
            res[t] = sqrt(r);
        }
    free(Y);
}

/* x = P y, P[k,j] = a[k-j+p]:  x[i] = sum_{m=0}^p a[p-m] y[i+m]. */
void SUF(bandqr_apply_P)(const SCALAR *a, int p, const SCALAR *restrict y,
                         int n, SCALAR *restrict x)
{
    for (int i = 0; i < n; i++) {
        int mmax = p < n - 1 - i ? p : n - 1 - i;
        SCALAR s = 0;
        for (int m = 0; m <= mmax; m++) s += a[p - m] * y[i + m];
        x[i] = s;
    }
}

void SUF(bandqr_get_R)(const struct SUF(band_qr) *qr, SCALAR *out)
{
    int n = qr->n, kuR = qr->kuR, Wout = kuR + 1;
    memset(out, 0, (size_t)n * Wout * sizeof(SCALAR));
    for (int i = 0; i < n; i++) {
        int jhi = i + kuR < n - 1 ? i + kuR : n - 1;
        for (int j = i; j <= jhi; j++)
            out[i * Wout + (j - i)] = qr->band[IDX(qr, i, j)];
    }
}

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
/* bandqr_impl.h -- templated band-QR kernel.  Included once per scalar type by
 * bandqr.c, with macros set:
 *     SCALAR   element type (double | double _Complex)
 *     REAL     real magnitude type (double)
 *     SUF(x)   name mangling  (x##_d | x##_z)
 *     CONJ(x)  conjugate      (identity | conj)
 *     ABS2(x)  squared magnitude ((x)*(x) | re*re+im*im)
 * The driver also provides type-independent IDX(qr,i,j), cpu_relax(), ilast_of().
 * No include guard -- this header is meant to be included multiple times.
 *
 * Givens (unitary) rotation zeroing b:  [[c, s], [-conj(s), c]] [a;b] = [r;0],
 * c real >= 0.  For real data conj is identity and this reduces to a plain
 * Givens rotation (the R diagonal may differ in sign from a c=a/t convention,
 * which is immaterial -- Q absorbs it).
 */

struct SUF(band_qr) {
    int  m, n, kl, ku, kuR, W;
    SCALAR *band;
    /* Givens factorisation: Q as a list of plane rotations. */
    REAL   *c;     /* cosines: real for both real and complex rotations */
    SCALAR *s;     /* sines */
    int    *p;     /* upper row index of each rotation */
    long    nrot;
    /* Householder factorisation (reflectors != 0): Q as reflectors.
     * V is n x (kl+1) row-major (V[j][0]=1 implicit), tau length n. */
    SCALAR *V;
    SCALAR *tau;
    int     reflectors;
    /* Almost-banded factorisation (rfill > 0): M = B + U L, Givens rotations
     * stored in c/s/p as usual; U is m x rfill (evolved), L is rfill x n (fixed
     * dense-row out-of-band content).  R = R_band(band) + U L. */
    SCALAR *U;
    SCALAR *L;
    int     rfill;
};

void SUF(bandqr_free)(struct SUF(band_qr) *qr)
{
    if (!qr) return;
    free(qr->band); free(qr->c); free(qr->s); free(qr->p);
    free(qr->V); free(qr->tau);
    free(qr->U); free(qr->L);
    free(qr);
}

static inline void SUF(givens)(SCALAR a, SCALAR b, REAL *c, SCALAR *s)
{
    REAL bb2 = ABS2(b);
    if (bb2 == 0) { *c = 1; *s = 0; return; }
    REAL aa2 = ABS2(a);
    REAL t = sqrt(aa2 + bb2);            /* = sqrt(a^2+b^2) for real (fast) */
    if (aa2 == 0) { *c = 0; *s = CONJ(b) / sqrt(bb2); return; }
    REAL aa = sqrt(aa2);
    *c = aa / t;
    *s = (a / aa) * CONJ(b) / t;
}

/* Apply one rotation to two contiguous, non-overlapping row-strips. */
static inline void SUF(apply_rotation)(SCALAR *restrict rp, SCALAR *restrict rq,
                                       int len, REAL c, SCALAR s)
{
    for (int t = 0; t < len; t++) {
        SCALAR ap = rp[t], aq = rq[t];
        rp[t] = c * ap + s * aq;
        rq[t] = c * aq - CONJ(s) * ap;
    }
}

static void SUF(run_column)(struct SUF(band_qr) *qr, int jcol, long off,
                            _Atomic int *front)
{
    int m = qr->m, n = qr->n, kl = qr->kl, kuR = qr->kuR, W = qr->W;
    SCALAR *band = qr->band;
    int ilast = ilast_of(jcol, kl, m);
    long r = 0;
    for (int q = ilast; q >= jcol + 1; q--, r++) {
        if (front && jcol > 0)
            while (atomic_load_explicit(&front[jcol - 1], memory_order_acquire) > q - 1)
                cpu_relax();
        int     p  = q - 1;
        SCALAR *rp = &band[(size_t)p * W + (jcol - p + kl)];
        SCALAR *rq = rp + (W - 1);
        SCALAR  b  = *rq;
        REAL c; SCALAR s;
        SUF(givens)(*rp, b, &c, &s);
        if (b != 0) {
            int jmax = q + kuR < n - 1 ? q + kuR : n - 1;
            SUF(apply_rotation)(rp, rq, jmax - jcol + 1, c, s);
        }
        qr->c[off + r] = c;
        qr->s[off + r] = s;
        qr->p[off + r] = p;
        if (front)
            atomic_store_explicit(&front[jcol], q, memory_order_release);
    }
    if (front)
        atomic_store_explicit(&front[jcol], -1, memory_order_release);
}

static struct SUF(band_qr) *SUF(alloc_and_load)(const SCALAR *ab, int m, int n,
                                                int kl, int ku, long **off_out,
                                                long *total_out)
{
    if (m < n || kl < 0 || ku < 0) return NULL;
    struct SUF(band_qr) *qr = calloc(1, sizeof *qr);
    if (!qr) return NULL;
    qr->m = m; qr->n = n; qr->kl = kl; qr->ku = ku;
    qr->kuR = ku + kl;
    qr->W   = kl + 1 + qr->kuR + 1;

    long *off = malloc((size_t)(n + 1) * sizeof(long));
    qr->band = calloc((size_t)m * qr->W, sizeof(SCALAR));
    if (!off || !qr->band) { free(off); SUF(bandqr_free)(qr); return NULL; }

    long total = 0;
    for (int j = 0; j < n; j++) {
        off[j] = total;
        int nr = ilast_of(j, kl, m) - j;
        if (nr > 0) total += nr;
    }
    off[n] = total;

    qr->c = malloc((size_t)(total + 1) * sizeof(REAL));
    qr->s = malloc((size_t)(total + 1) * sizeof(SCALAR));
    qr->p = malloc((size_t)(total + 1) * sizeof(int));
    if (!qr->c || !qr->s || !qr->p) { free(off); SUF(bandqr_free)(qr); return NULL; }
    qr->nrot = total;

    int Win = kl + ku + 1;
    for (int i = 0; i < m; i++) {
        int jlo = i - kl < 0 ? 0 : i - kl;
        int jhi = i + ku > n - 1 ? n - 1 : i + ku;
        for (int j = jlo; j <= jhi; j++)
            qr->band[IDX(qr, i, j)] = ab[i * Win + (j - i + kl)];
    }
    *off_out = off; *total_out = total;
    return qr;
}

struct SUF(band_qr) *SUF(bandqr_factor)(const SCALAR *ab, int m, int n, int kl, int ku)
{
    long *off, total;
    struct SUF(band_qr) *qr = SUF(alloc_and_load)(ab, m, n, kl, ku, &off, &total);
    if (!qr) return NULL;
    for (int j = 0; j < n; j++)
        SUF(run_column)(qr, j, off[j], NULL);
    free(off);
    return qr;
}

typedef struct {
    struct SUF(band_qr) *qr;
    const long  *off;
    _Atomic int *front;
    _Atomic int *next_col;
    int          n;
} SUF(worker_arg);

static void *SUF(worker)(void *a)
{
    SUF(worker_arg) *w = a;
    for (;;) {
        int j = atomic_fetch_add_explicit(w->next_col, 1, memory_order_relaxed);
        if (j >= w->n) break;
        SUF(run_column)(w->qr, j, w->off[j], w->front);
    }
    return NULL;
}

struct SUF(band_qr) *SUF(bandqr_factor_mt)(const SCALAR *ab, int m, int n,
                                           int kl, int ku, int nthreads)
{
    if (nthreads == 0) nthreads = 1;          /* auto = serial (see notes) */
    if (nthreads <= 1)
        return SUF(bandqr_factor)(ab, m, n, kl, ku);

    long *off, total;
    struct SUF(band_qr) *qr = SUF(alloc_and_load)(ab, m, n, kl, ku, &off, &total);
    if (!qr) return NULL;

    _Atomic int *front = malloc((size_t)n * sizeof *front);
    if (!front) { free(off); SUF(bandqr_free)(qr); return NULL; }
    for (int j = 0; j < n; j++)
        atomic_init(&front[j], ilast_of(j, kl, m) + 1);

    _Atomic int next_col;
    atomic_init(&next_col, 0);
    if (nthreads > n) nthreads = n;

    SUF(worker_arg) wa = { qr, off, front, &next_col, n };
    pthread_t *th = malloc((size_t)nthreads * sizeof *th);
    for (int t = 0; t < nthreads; t++) pthread_create(&th[t], NULL, SUF(worker), &wa);
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);

    free(th); free(front); free(off);
    return qr;
}

/* Householder band QR.  One reflector per column annihilates all kl
 * subdiagonals at once; applied row-contiguously (weighted sum of row strips,
 * then an outer-product update).  R is the same band format; Q is stored as the
 * reflectors (V, tau) for Q^H application. */
struct SUF(band_qr) *SUF(bandqr_factor_hh)(const SCALAR *ab, int m, int n,
                                           int kl, int ku)
{
    if (m < n || kl < 0 || ku < 0) return NULL;
    struct SUF(band_qr) *qr = calloc(1, sizeof *qr);
    if (!qr) return NULL;
    qr->m = m; qr->n = n; qr->kl = kl; qr->ku = ku;
    qr->kuR = ku + kl;
    qr->W   = kl + 1 + qr->kuR + 1;
    qr->reflectors = 1;
    qr->band = calloc((size_t)m * qr->W, sizeof(SCALAR));
    qr->V    = calloc((size_t)n * (kl + 1), sizeof(SCALAR));
    qr->tau  = calloc((size_t)n, sizeof(SCALAR));
    SCALAR *u = malloc((size_t)(qr->kuR + 1) * sizeof(SCALAR));
    if (!qr->band || !qr->V || !qr->tau || !u) { free(u); SUF(bandqr_free)(qr); return NULL; }

    int Win = kl + ku + 1, W = qr->W, kuR = qr->kuR;
    SCALAR *band = qr->band;
    for (int i = 0; i < m; i++) {
        int jlo = i - kl < 0 ? 0 : i - kl;
        int jhi = i + ku > n - 1 ? n - 1 : i + ku;
        for (int j = jlo; j <= jhi; j++)
            band[IDX(qr, i, j)] = ab[i * Win + (j - i + kl)];
    }

    for (int j = 0; j < n; j++) {
        int ilast = j + kl < m - 1 ? j + kl : m - 1;
        int L = ilast - j + 1;                          /* reflector length    */
        SCALAR *vj = &qr->V[(size_t)j * (kl + 1)];
        vj[0] = 1;

        SCALAR alpha = band[(size_t)j * W + kl];        /* x[0] = diagonal      */
        REAL xnorm2 = 0;                                /* ||subdiagonals||^2   */
        for (int i = 1; i < L; i++) {
            SCALAR xi = band[(size_t)(j + i) * W + (kl - i)];
            xnorm2 += ABS2(xi);
        }
        if (xnorm2 == 0) { qr->tau[j] = 0; continue; }  /* already upper-tri    */

        REAL bnorm = sqrt(ABS2(alpha) + xnorm2);
        REAL signr = RE(alpha) >= 0 ? 1.0 : -1.0;
        SCALAR beta = -signr * bnorm;                   /* real-valued          */
        qr->tau[j] = (beta - alpha) / beta;
        SCALAR scal = 1.0 / (alpha - beta);
        for (int i = 1; i < L; i++)
            vj[i] = band[(size_t)(j + i) * W + (kl - i)] * scal;

        /* apply H = I - tau v v^H to rows j..ilast, columns j..jmax */
        int jmax = j + kuR < n - 1 ? j + kuR : n - 1;
        int wlen = jmax - j + 1;
        for (int t = 0; t < wlen; t++) u[t] = 0;
        for (int i = 0; i < L; i++) {                   /* u = sum conj(v_i) row_i */
            SCALAR *rb = &band[(size_t)(j + i) * W + (kl - i)];
            SCALAR vc = CONJ(vj[i]);
            for (int t = 0; t < wlen; t++) u[t] += vc * rb[t];
        }
        /* Apply H^H = I - conj(tau) v v^H (it is H^H, not H, that zeroes the
         * column for complex data; conj is identity for real). */
        SCALAR ctau = CONJ(qr->tau[j]);
        for (int i = 0; i < L; i++) {
            SCALAR *rb = &band[(size_t)(j + i) * W + (kl - i)];
            SCALAR f = ctau * vj[i];
            for (int t = 0; t < wlen; t++) rb[t] -= f * u[t];
        }
    }
    free(u);
    return qr;
}

/* Almost-banded QR ("ride"): M = B + U L (B in-band, L = r x n out-of-band
 * content of the dense rows dr[0..r-1], U = m x r indicator evolving under the
 * rotations).  Adjacent-row Givens zero the true value M[q,j] = B[q,j] +
 * U[q,:].L[:,j], applied to B's band strips and U's rows; L stays fixed.
 * R = R_band + (Q^H U) L stays almost-banded.  Q is the stored rotations. */
struct SUF(band_qr) *SUF(bandqr_factor_ab)(const SCALAR *ab, int m, int n,
                                           int kl, int ku,
                                           const int *dr, int r, const SCALAR *Lin)
{
    if (m < n || kl < 0 || ku < 0 || r < 1) return NULL;
    struct SUF(band_qr) *qr = calloc(1, sizeof *qr);
    if (!qr) return NULL;
    qr->m = m; qr->n = n; qr->kl = kl; qr->ku = ku;
    qr->kuR = ku + kl;
    qr->W   = kl + 1 + qr->kuR + 1;
    qr->rfill = r;

    long total = 0;
    for (int j = 0; j < n; j++) { int nr = ilast_of(j, kl, m) - j; if (nr > 0) total += nr; }
    qr->band = calloc((size_t)m * qr->W, sizeof(SCALAR));
    qr->c = malloc((size_t)(total + 1) * sizeof(REAL));
    qr->s = malloc((size_t)(total + 1) * sizeof(SCALAR));
    qr->p = malloc((size_t)(total + 1) * sizeof(int));
    qr->U = calloc((size_t)m * r, sizeof(SCALAR));
    qr->L = malloc((size_t)r * n * sizeof(SCALAR));
    if (!qr->band || !qr->c || !qr->s || !qr->p || !qr->U || !qr->L) {
        SUF(bandqr_free)(qr); return NULL;
    }
    qr->nrot = total;

    int Win = kl + ku + 1, W = qr->W, kuR = qr->kuR;
    SCALAR *band = qr->band, *U = qr->U, *L = qr->L;
    for (int i = 0; i < m; i++) {
        int jlo = i - kl < 0 ? 0 : i - kl, jhi = i + ku > n - 1 ? n - 1 : i + ku;
        for (int j = jlo; j <= jhi; j++) band[IDX(qr, i, j)] = ab[i * Win + (j - i + kl)];
    }
    memcpy(L, Lin, (size_t)r * n * sizeof(SCALAR));
    for (int t = 0; t < r; t++) U[(size_t)dr[t] * r + t] = 1;   /* indicator */

    long nrot = 0;
    for (int jcol = 0; jcol < n; jcol++) {
        int ilast = jcol + kl < m - 1 ? jcol + kl : m - 1;
        for (int q = ilast; q >= jcol + 1; q--) {
            int pp = q - 1;
            SCALAR *rp = &band[(size_t)pp * W + (jcol - pp + kl)];
            SCALAR *rq = rp + (W - 1);
            SCALAR ap = *rp, aq = *rq;                  /* M[pp,j], M[q,j] = B + U.L */
            SCALAR *up = &U[(size_t)pp * r], *uq = &U[(size_t)q * r];
            for (int t = 0; t < r; t++) {
                SCALAR ljc = L[(size_t)t * n + jcol];
                ap += up[t] * ljc; aq += uq[t] * ljc;
            }
            REAL c; SCALAR s;
            SUF(givens)(ap, aq, &c, &s);
            int jmax = q + kuR < n - 1 ? q + kuR : n - 1;
            SUF(apply_rotation)(rp, rq, jmax - jcol + 1, c, s);   /* B rows */
            for (int t = 0; t < r; t++) {                          /* U rows */
                SCALAR u1 = up[t], u2 = uq[t];
                up[t] = c * u1 + s * u2;
                uq[t] = c * u2 - CONJ(s) * u1;
            }
            qr->c[nrot] = c; qr->s[nrot] = s; qr->p[nrot] = pp; nrot++;
        }
    }
    qr->nrot = nrot;
    return qr;
}

/* Solve A x = b for an almost-banded factorisation.  Q^H b, then back-solve
 * R x = y with R = R_band + U L: band sweep plus a running r-vector
 * g = sum_{j>i} L[:,j] x[j] carrying the low-rank fill. */
double SUF(bandqr_solve_ab)(const struct SUF(band_qr) *qr, const SCALAR *b, SCALAR *x)
{
    int m = qr->m, n = qr->n, W = qr->W, kuR = qr->kuR, kl = qr->kl, r = qr->rfill;
    const SCALAR *band = qr->band, *U = qr->U, *L = qr->L;
    SCALAR *y = malloc((size_t)m * sizeof(SCALAR));
    memcpy(y, b, (size_t)m * sizeof(SCALAR));
    SUF(bandqr_apply_Qt)(qr, y);                  /* Q^H b (stored rotations) */

    SCALAR *g = calloc(r, sizeof(SCALAR));
    for (int i = n - 1; i >= 0; i--) {
        int jhi = i + kuR < n - 1 ? i + kuR : n - 1;
        const SCALAR *Rrow = &band[(size_t)i * W + (kl + 1)];
        SCALAR bandsum = 0;
        for (int d = 1; d <= jhi - i; d++) bandsum += Rrow[d - 1] * x[i + d];
        SCALAR fillsum = 0, diag = band[(size_t)i * W + kl];
        const SCALAR *Ui = &U[(size_t)i * r];
        for (int t = 0; t < r; t++) {
            fillsum += Ui[t] * g[t];
            diag    += Ui[t] * L[(size_t)t * n + i];
        }
        x[i] = (y[i] - bandsum - fillsum) / diag;
        for (int t = 0; t < r; t++) g[t] += L[(size_t)t * n + i] * x[i];
    }
    free(g);
    REAL res = 0;
    for (int i = n; i < m; i++) res += ABS2(y[i]);
    free(y);
    return sqrt(res);
}

/* v (length m) <- Q^H v (Q^T for real). */
void SUF(bandqr_apply_Qt)(const struct SUF(band_qr) *qr, SCALAR *v)
{
    if (qr->reflectors) {                 /* Householder: apply H_0 .. H_{n-1} */
        int kl = qr->kl, m = qr->m, klp = kl + 1;
        for (int j = 0; j < qr->n; j++) {
            SCALAR tau = qr->tau[j];
            if (tau == 0) continue;
            int L = j + kl < m ? kl + 1 : m - j;
            const SCALAR *vj = &qr->V[(size_t)j * klp];
            SCALAR w = 0;
            for (int i = 0; i < L; i++) w += CONJ(vj[i]) * v[j + i];
            w *= CONJ(tau);                      /* apply H^H (conj for complex) */
            for (int i = 0; i < L; i++) v[j + i] -= vj[i] * w;
        }
        return;
    }
    for (long k = 0; k < qr->nrot; k++) { /* Givens: replay the rotations */
        int p = qr->p[k]; REAL c = qr->c[k]; SCALAR s = qr->s[k];
        SCALAR vp = v[p], vq = v[p + 1];
        v[p]     = c * vp + s * vq;
        v[p + 1] = c * vq - CONJ(s) * vp;
    }
}

void SUF(bandqr_solve_R)(const struct SUF(band_qr) *qr, const SCALAR *y, SCALAR *x)
{
    const SCALAR *restrict band = qr->band;
    int n = qr->n, W = qr->W, kuR = qr->kuR, kl = qr->kl;
    for (int i = n - 1; i >= 0; i--) {
        int jhi = i + kuR < n - 1 ? i + kuR : n - 1;
        const SCALAR *restrict Rrow = &band[(size_t)i * W + (kl + 1)];
        const SCALAR *restrict xj   = &x[i + 1];
        SCALAR sum = 0;
        for (int t = 0; t < jhi - i; t++) sum += Rrow[t] * xj[t];
        x[i] = (y[i] - sum) / band[(size_t)i * W + kl];
    }
}

double SUF(bandqr_solve)(const struct SUF(band_qr) *qr, const SCALAR *b, SCALAR *x)
{
    int m = qr->m, n = qr->n;
    SCALAR *y = malloc((size_t)m * sizeof(SCALAR));
    memcpy(y, b, (size_t)m * sizeof(SCALAR));
    SUF(bandqr_apply_Qt)(qr, y);
    SUF(bandqr_solve_R)(qr, y, x);
    REAL res = 0;
    for (int i = n; i < m; i++) res += ABS2(y[i]);
    free(y);
    return sqrt(res);
}

void SUF(bandqr_solve_many)(const struct SUF(band_qr) *qr, const SCALAR *B,
                            int nrhs, SCALAR *X, double *res)
{
    int m = qr->m, n = qr->n, W = qr->W, kuR = qr->kuR, kl = qr->kl;
    const SCALAR *restrict band = qr->band;
    SCALAR *Y = malloc((size_t)m * nrhs * sizeof(SCALAR));
    memcpy(Y, B, (size_t)m * nrhs * sizeof(SCALAR));

    if (qr->reflectors) {                        /* Y <- Q^H B (Householder) */
        int klp = kl + 1;
        SCALAR *w = malloc((size_t)nrhs * sizeof(SCALAR));
        for (int j = 0; j < n; j++) {
            SCALAR tau = qr->tau[j];
            if (tau == 0) continue;
            int L = j + kl < m ? kl + 1 : m - j;
            const SCALAR *vj = &qr->V[(size_t)j * klp];
            for (int t = 0; t < nrhs; t++) w[t] = 0;
            for (int i = 0; i < L; i++) {
                SCALAR vc = CONJ(vj[i]);
                const SCALAR *yi = &Y[(size_t)(j + i) * nrhs];
                for (int t = 0; t < nrhs; t++) w[t] += vc * yi[t];
            }
            SCALAR ctau = CONJ(tau);             /* apply H^H (conj for complex) */
            for (int t = 0; t < nrhs; t++) w[t] *= ctau;
            for (int i = 0; i < L; i++) {
                SCALAR vi = vj[i];
                SCALAR *yi = &Y[(size_t)(j + i) * nrhs];
                for (int t = 0; t < nrhs; t++) yi[t] -= vi * w[t];
            }
        }
        free(w);
    } else {
        for (long k = 0; k < qr->nrot; k++) {    /* Y <- Q^H B (Givens) */
            int p = qr->p[k]; REAL c = qr->c[k]; SCALAR s = qr->s[k];
            SCALAR *restrict yp = &Y[(size_t)p * nrhs];
            SCALAR *restrict yq = &Y[(size_t)(p + 1) * nrhs];
            for (int t = 0; t < nrhs; t++) {
                SCALAR ap = yp[t], aq = yq[t];
                yp[t] = c * ap + s * aq;
                yq[t] = c * aq - CONJ(s) * ap;
            }
        }
    }
    for (int i = n - 1; i >= 0; i--) {           /* blocked band back-sub */
        int jhi = i + kuR < n - 1 ? i + kuR : n - 1;
        const SCALAR *restrict Rrow = &band[(size_t)i * W + (kl + 1)];
        SCALAR       *restrict xi   = &X[(size_t)i * nrhs];
        const SCALAR *restrict yi   = &Y[(size_t)i * nrhs];
        for (int t = 0; t < nrhs; t++) xi[t] = yi[t];
        for (int d = 1; d <= jhi - i; d++) {
            SCALAR Rid = Rrow[d - 1];
            const SCALAR *restrict xd = &X[(size_t)(i + d) * nrhs];
            for (int t = 0; t < nrhs; t++) xi[t] -= Rid * xd[t];
        }
        SCALAR dinv = 1.0 / band[(size_t)i * W + kl];
        for (int t = 0; t < nrhs; t++) xi[t] *= dinv;
    }
    if (res)
        for (int t = 0; t < nrhs; t++) {
            REAL r = 0;
            for (int i = n; i < m; i++) r += ABS2(Y[(size_t)i * nrhs + t]);
            res[t] = sqrt(r);
        }
    free(Y);
}

/* x = P y, P[k,j] = a[k-j+p]:  x[i] = sum_{m=0}^p a[p-m] y[i+m]. */
void SUF(bandqr_apply_P)(const SCALAR *a, int p, const SCALAR *restrict y,
                         int n, SCALAR *restrict x)
{
    for (int i = 0; i < n; i++) {
        int mmax = p < n - 1 - i ? p : n - 1 - i;
        SCALAR s = 0;
        for (int m = 0; m <= mmax; m++) s += a[p - m] * y[i + m];
        x[i] = s;
    }
}

void SUF(bandqr_get_R)(const struct SUF(band_qr) *qr, SCALAR *out)
{
    int n = qr->n, kuR = qr->kuR, Wout = kuR + 1;
    memset(out, 0, (size_t)n * Wout * sizeof(SCALAR));
    for (int i = 0; i < n; i++) {
        int jhi = i + kuR < n - 1 ? i + kuR : n - 1;
        for (int j = i; j <= jhi; j++)
            out[i * Wout + (j - i)] = qr->band[IDX(qr, i, j)];
    }
}

#undef SCALAR
#undef REAL
#undef SUF
#undef CONJ
#undef ABS2
#undef RE
"""

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
            raise RuntimeError("bandqr: C compilation failed.\nCommand: %s\n%s"
                               % (" ".join(cmd), proc.stderr))
    return lib

_lib = ct.CDLL(_build())


_dbl = ct.POINTER(ct.c_double)
_vp  = ct.c_void_p
_int = ct.c_int

# Set up argtypes for both the _d (float64) and _z (complex128) instantiations.
# Real arrays use POINTER(c_double); complex arrays are passed by address as
# c_void_p (numpy complex128 is interleaved re/im, matching C double _Complex).
for _suf, _arr in (("_d", _dbl), ("_z", _vp)):
    f = getattr(_lib, "bandqr_factor" + _suf);     f.restype = _vp; f.argtypes = [_arr, _int, _int, _int, _int]
    f = getattr(_lib, "bandqr_factor_mt" + _suf);  f.restype = _vp; f.argtypes = [_arr, _int, _int, _int, _int, _int]
    f = getattr(_lib, "bandqr_factor_hh" + _suf);  f.restype = _vp; f.argtypes = [_arr, _int, _int, _int, _int]
    f = getattr(_lib, "bandqr_factor_ab" + _suf);  f.restype = _vp; f.argtypes = [_arr, _int, _int, _int, _int, ct.POINTER(_int), _int, _arr]
    f = getattr(_lib, "bandqr_solve_ab" + _suf);   f.restype = ct.c_double; f.argtypes = [_vp, _arr, _arr]
    f = getattr(_lib, "bandqr_solve" + _suf);      f.restype = ct.c_double; f.argtypes = [_vp, _arr, _arr]
    f = getattr(_lib, "bandqr_solve_many" + _suf); f.argtypes = [_vp, _arr, _int, _arr, _dbl]   # res always real
    f = getattr(_lib, "bandqr_apply_Qt" + _suf);   f.argtypes = [_vp, _arr]
    f = getattr(_lib, "bandqr_solve_R" + _suf);    f.argtypes = [_vp, _arr, _arr]
    f = getattr(_lib, "bandqr_apply_P" + _suf);    f.argtypes = [_arr, _int, _arr, _int, _arr]
    f = getattr(_lib, "bandqr_get_R" + _suf);      f.argtypes = [_vp, _arr]
    f = getattr(_lib, "bandqr_free" + _suf);       f.argtypes = [_vp]


def _ptr(a):
    return a.ctypes.data_as(_dbl)


def _cptr(a):
    return a.ctypes.data_as(ct.c_void_p)


def _apply_P(a, y):
    """x = P y for the recurrence preconditioner P[k,j]=a[k-j+p]  (in C).
    Real or complex, following the dtype of a / y."""
    cplx = np.iscomplexobj(a) or np.iscomplexobj(y)
    dt = np.complex128 if cplx else np.float64
    a = np.ascontiguousarray(a, dtype=dt)
    y = np.ascontiguousarray(y, dtype=dt)
    p = len(a) - 1
    n = y.shape[0]
    pf = _cptr if cplx else _ptr
    cfn = getattr(_lib, "bandqr_apply_P" + ("_z" if cplx else "_d"))
    ap = pf(a)
    fn = lambda yk, xk: cfn(ap, p, pf(yk), n, pf(xk))
    if y.ndim == 1:
        x = np.zeros(n, dtype=dt)
        fn(y, x)
        return x
    X = np.zeros_like(y)
    for k in range(y.shape[1]):
        yk = np.ascontiguousarray(y[:, k]); xk = np.zeros(n, dtype=dt)
        fn(yk, xk)
        X[:, k] = xk
    return X


def _build_P(a, n):
    """Explicit sparse right-preconditioner P with P[j-p+i, j] = a[i] (an upper-
    triangular band), so that M @ P is banded and x = P @ y.  Mirrors _apply_P.
    Built as diagonals (offset p-i carries a[i]) -- no Python element loop."""
    import scipy.sparse as sp
    a = np.asarray(a); p = len(a) - 1
    offs = [p - i for i in range(p + 1)]
    diags = [a[i] * np.ones(n - (p - i), dtype=a.dtype) for i in range(p + 1)]
    return sp.diags(diags, offs, shape=(n, n), format="csc")


def detect_recurrence(rows, max_order=20, tol=1e-8, trim=0):
    """Find the shortest stencil a = [a_0, ..., a_p] that annihilates the
    interior of every given row:  sum_i a_i * row[k+i] = 0.

    This is purely numerical -- it reads the recurrence off the values, with no
    knowledge of the basis.  Any sequence that is a sum of (polynomial in k) x
    (geometric r^k) terms obeys such a recurrence (Chebyshev boundary evals ->
    order 2; their derivatives -> higher order; evals at +-1 -> order 1), and it
    shows up as a null vector of the Hankel matrix of windows.

    rows : 1D row or 2D stack of rows that should share one annihilator.
    Returns the stencil (length p+1, largest coeff normalised to 1), or None.
    """
    _dt = np.complex128 if np.iscomplexobj(rows) else np.float64
    R = np.atleast_2d(np.asarray(rows, dtype=_dt))
    if trim:
        R = R[:, trim:R.shape[1] - trim]
    nwin = R.shape[1]
    for p in range(1, max_order + 1):
        if nwin - p < p + 1:
            break
        blocks = [np.lib.stride_tricks.sliding_window_view(r, p + 1) for r in R]
        H = np.vstack(blocks)
        s = np.linalg.svd(H, compute_uv=False)
        if s[-1] <= tol * s[0]:
            Vt = np.linalg.svd(H, full_matrices=False)[2]
            a = Vt[-1]
            return a / a[np.argmax(np.abs(a))]
    return None


def solve_ride(M, kl, ku, b, dense_rows):
    """Almost-banded QR with the dense (boundary) rows carried along
    (Olver-Townsend): M = B + U L is factored as a whole, so R = Q^H M exactly
    -- well-conditioned, no preconditioner.  Dense rows only (no tau columns).
    Real or complex (dtype follows M).  One-shot; for factor-once/solve-many use
    RideQR or factorize(M, method='ride', ...)."""
    return RideQR(M, kl, ku, dense_rows).solve(b)


def solve_precond_rows(M, kl, ku, b, dense_rows, max_order=20, tol=1e-8):
    """Solve M x = b where M is a (kl,ku) banded bulk plus dense rows (boundary
    conditions, near the top), by collapsing those rows into the band with a
    recurrence preconditioner -- so the solve is a SINGLE banded QR, no low-rank
    correction (the well-conditioned route).

    Detect the shared annihilator a (length p+1) of the dense rows, build the
    upper-triangular Toeplitz P with P[j-p+i, j] = a_i, and solve M P y = b,
    x = P y.  Right-multiplying by P annihilates each dense row's interior; the
    O(p) surviving entries land in the first p columns.

    No restriction on the number of boundary rows: those survivors only widen
    the lower bandwidth to about r-1, which BandQR takes in stride.  The one
    subtlety is that sparse(M) @ P leaves ~1e-16 *structural* cancellation
    entries across the (originally dense) BC rows; we drop them by value so the
    band stays tight rather than appearing dense.  One-shot; for
    factor-once/solve-many use CollapseQR or factorize(M, method='collapse').
    """
    return CollapseQR(M, kl, ku, dense_rows, max_order, tol).solve(b)


def dense_to_band(A, kl, ku):
    """Pack a dense m x n array into row-major band form (m, kl+ku+1).
    Real input stays float64, complex input stays complex128 (dtype follows A)."""
    dt = np.complex128 if np.issubdtype(np.asarray(A).dtype, np.complexfloating) else np.float64
    A = np.ascontiguousarray(A, dtype=dt)
    m, n = A.shape
    ab = np.zeros((m, kl + ku + 1), dtype=dt)
    for i in range(m):
        for j in range(max(0, i - kl), min(n - 1, i + ku) + 1):
            ab[i, j - i + kl] = A[i, j]
    return ab


def _bulk_band(M, kl, ku):
    """Band storage (n, kl+ku+1) of M's in-band entries, with any zero diagonal
    forced to 1 so the banded M0 is invertible.  Accepts dense or scipy.sparse;
    never materialises an n x n dense array for sparse input."""
    n = M.shape[0]
    ab = np.zeros((n, kl + ku + 1))
    try:
        import scipy.sparse as sp
        is_sparse = sp.issparse(M)
    except ImportError:
        is_sparse = False
    if is_sparse:
        Mc = M.tocoo()
        inb = (Mc.col >= Mc.row - kl) & (Mc.col <= Mc.row + ku)
        ab[Mc.row[inb], Mc.col[inb] - Mc.row[inb] + kl] = Mc.data[inb]
    else:
        M = np.asarray(M, dtype=np.float64)
        for i in range(n):
            lo, hi = max(0, i - kl), min(n - 1, i + ku)
            ab[i, lo - i + kl:hi - i + kl + 1] = M[i, lo:hi + 1]
    d = ab[:, kl]
    d[np.abs(d) < 1e-300] = 1.0
    return ab


def _m0_row(ab, kl, ku, i, n):
    row = np.zeros(n)
    lo, hi = max(0, i - kl), min(n - 1, i + ku)
    row[lo:hi + 1] = ab[i, lo - i + kl:hi - i + kl + 1]
    return row


def _m0_col(ab, kl, ku, j, n):
    col = np.zeros(n)
    for i in range(max(0, j - ku), min(n - 1, j + kl) + 1):
        col[i] = ab[i, j - i + kl]
    return col


def solve_bordered(M, kl, ku, b, dense_rows, dense_cols):
    """Solve M x = b for an almost-banded M: a (kl,ku) banded bulk plus the
    dense rows `dense_rows` (boundary conditions) and dense columns
    `dense_cols` (tau corrections), with len(dense_rows), len(dense_cols) = O(1).
    M may be dense or scipy.sparse (sparse is never densified).

    Robust for ANY bandwidth -- in particular when #dense rows > kl, where the
    delete-the-border-and-Schur approach produces a singular bulk.

    Method (Woodbury): M0 = in-band(M) is banded on the main diagonal and
    invertible; M - M0 is nonzero only in the dense rows/cols, hence rank r+c.
    M0^{-1} comes from the fast band QR; the correction is an (r+c) x (r+c)
    dense solve.  The band factorisation -- the expensive part -- is reused
    across the 1 + (r+c) right-hand sides via solve_many.
    """
    n = M.shape[0]
    dr = np.asarray(dense_rows, dtype=int)
    dc = np.asarray(dense_cols, dtype=int)
    r, c = len(dr), len(dc)

    ab = _bulk_band(M, kl, ku)
    qr = BandQR(ab, kl, ku)

    # Dense row/column contents of M (the only O(n)-per-border slices we form).
    try:
        import scipy.sparse as sp
        rows_dense = np.asarray(M.tocsr()[dr, :].todense()) if sp.issparse(M) \
            else np.asarray(M, dtype=np.float64)[dr, :]
        cols_dense = np.asarray(M.tocsc()[:, dc].todense()) if sp.issparse(M) \
            else np.asarray(M, dtype=np.float64)[:, dc]
    except ImportError:
        Md = np.asarray(M, dtype=np.float64)
        rows_dense, cols_dense = Md[dr, :], Md[:, dc]

    # Low-rank correction M - M0 = U V^T.  Full rows for the dense rows; the
    # dense columns contribute only on non-dense rows (the corner is already in
    # the row terms, so it is not double-counted).
    U = np.zeros((n, r + c))
    V = np.zeros((n, r + c))
    for t, i in enumerate(dr):
        U[i, t] = 1.0
        V[:, t] = rows_dense[t] - _m0_row(ab, kl, ku, i, n)
    keep = np.ones(n, dtype=bool); keep[dr] = False
    for t, j in enumerate(dc):
        col = cols_dense[:, t] - _m0_col(ab, kl, ku, j, n)
        col[~keep] = 0.0
        U[:, r + t] = col
        V[j, r + t] = 1.0

    b = np.asarray(b, dtype=np.float64)
    single = b.ndim == 1
    B = b[:, None] if single else b

    MiU = qr.solve(U)                          # M0^{-1} U   (n x (r+c))
    MiB = qr.solve(B)                          # M0^{-1} B   (n x nrhs)
    if MiB.ndim == 1:
        MiB = MiB[:, None]
    cap = np.eye(r + c) + V.T @ MiU            # capacitance (r+c)x(r+c)
    X = MiB - MiU @ np.linalg.solve(cap, V.T @ MiB)
    return X[:, 0] if single else X


def solve_with_tau_columns(N_ab, kl, ku, F, b):
    """Solve [N | F] x = rhs, where the bulk N is m x (m-c) banded (tall by c)
    and F is m x c dense (the tau columns).  Robust for any bandwidth -- no
    invertible-bulk requirement.

        N_ab : band storage of N, shape (m, kl+ku+1)
        F    : (m, c) dense tau columns
        b    : (m,) or (m, nrhs) right-hand side
    Returns x with the bulk unknowns first, the c tau unknowns last.

    Method: QR the tall banded N.  Q^T[N|F] = [R | Q^T F]; the bottom c rows are
    [0 | (Q^T F)_2], a c x c dense system for the tau unknowns; the top n rows
    then give a banded back-solve for the bulk unknowns.
    """
    m, c = F.shape
    n = m - c
    qr = BandQR(N_ab, kl, ku, ncols=n)
    QtF = np.column_stack([qr.apply_Qt(F[:, j]) for j in range(c)])  # (m, c)

    b = np.asarray(b, dtype=np.float64)
    single = b.ndim == 1
    B = b[:, None] if single else b
    X = np.empty((m, B.shape[1]))
    for r in range(B.shape[1]):
        Qtb = qr.apply_Qt(B[:, r])
        x2 = np.linalg.solve(QtF[n:, :], Qtb[n:])          # tau unknowns (c x c)
        x1 = qr.solve_R(Qtb[:n] - QtF[:n, :] @ x2)         # bulk unknowns (banded)
        X[:n, r] = x1
        X[n:, r] = x2
    return X[:, 0] if single else X


# ---------------------------------------------------------------------------
# Unified flagged front-end -- the "QR extravaganza".
#   method axis: auto | givens | householder | collapse | ride | lu
#   device axis: cpu | gpu
# ---------------------------------------------------------------------------

_METHODS = ("auto", "givens", "householder", "lq", "collapse", "ride", "bordered", "lu")
_DEVICES = ("cpu", "gpu")


def _is_complex(A):
    return np.issubdtype(getattr(A, "dtype", np.asarray(A).dtype), np.complexfloating)


def _to_band(A, kl, ku):
    """Band storage (n, kl+ku+1) of A's in-band entries (dense or scipy.sparse).
    Real input stays float64; complex input stays complex128 (dtype follows A)."""
    import scipy.sparse as sp
    n = A.shape[0]
    dt = np.complex128 if _is_complex(A) else np.float64
    ab = np.zeros((n, kl + ku + 1), dtype=dt)
    if sp.issparse(A):
        C = A.tocoo()
        m = (C.col >= C.row - kl) & (C.col <= C.row + ku)
        ab[C.row[m], C.col[m] - C.row[m] + kl] = C.data[m]
    else:
        A = np.asarray(A, dtype=dt)
        for i in range(n):
            lo, hi = max(0, i - kl), min(n - 1, i + ku)
            ab[i, lo - i + kl:hi - i + kl + 1] = A[i, lo:hi + 1]
    return ab


def detect_structure(A, tol=1e-12):
    """Infer (kl, ku, dense_rows, dense_cols) of an almost-banded matrix: rows /
    columns carrying out-of-band nonzeros are flagged dense, the rest set the band."""
    import scipy.sparse as sp
    M = sp.csr_matrix(A)
    rn = np.diff(M.indptr)
    cn = np.diff(M.tocsc().indptr)
    dense_rows = np.where(rn > 4 * np.median(rn) + 4)[0]
    dense_cols = np.where(cn > 4 * np.median(cn) + 4)[0]
    C = M.tocoo()
    keep = (~np.isin(C.row, dense_rows)) & (~np.isin(C.col, dense_cols)) & (np.abs(C.data) > tol)
    off = C.col[keep] - C.row[keep]
    kl = int(max(0, -off.min())) if off.size else 0
    ku = int(max(0, off.max())) if off.size else 0
    return kl, ku, dense_rows, dense_cols


def solve(A, b, method="auto", device="cpu",
          kl=None, ku=None, dense_rows=None, dense_cols=None):
    """Solve A x = b for a banded / almost-banded A, choosing the factorisation.

    method : "auto" picks by structure, or force one of --
        "givens"      Givens band QR (purely banded)                    [ready]
        "lq"          LQ = QR of the transpose (factored from the row side) [ready]
        "collapse"    auto BC-row recurrence collapse -> one band QR    [ready]
        "householder" Householder band QR                              [roadmap]
        "ride"        almost-banded QR, BCs carried along (Olver-Townsend) [roadmap]
        "lu"          banded LU                                        [roadmap]
    device : "cpu" [ready]; "gpu" batched backend -- the partridge      [roadmap]

    kl, ku, dense_rows, dense_cols override structure detection when given.
    """
    if device not in _DEVICES:
        raise ValueError(f"device must be one of {_DEVICES}, got {device!r}")
    if device == "gpu":
        raise NotImplementedError("device='gpu' (batched backend, the partridge) is on the roadmap")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if getattr(A, "ndim", None) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"solve expects a square 2-D matrix, got shape {getattr(A, 'shape', None)}")
    if np.asarray(b).shape[0] != A.shape[0]:
        raise ValueError(f"b has {np.asarray(b).shape[0]} rows but A is {A.shape[0]}x{A.shape[0]}")

    akl, aku, adr, adc = detect_structure(A)
    kl = akl if kl is None else int(kl)
    ku = aku if ku is None else int(ku)
    dense_rows = adr if dense_rows is None else np.asarray(dense_rows, dtype=int)
    dense_cols = adc if dense_cols is None else np.asarray(dense_cols, dtype=int)
    has_rows, has_cols = len(dense_rows) > 0, len(dense_cols) > 0

    if method == "auto":
        if not has_rows and not has_cols:
            # Householder factors ~25% faster for wide bands; Givens wins narrow.
            method = "householder" if kl >= 6 else "givens"
        elif has_rows != has_cols:
            method = "collapse"      # rows-only or tau-columns-only -> recurrence collapse
        else:
            method = "bordered"      # dense rows AND tau columns -> Woodbury

    dt = np.complex128 if _is_complex(A) else np.float64   # dtype follows A

    if method in ("givens", "householder"):
        if has_rows or has_cols:
            raise ValueError(f"method={method!r} needs a purely banded matrix "
                             "(found dense rows/cols); use 'collapse' or 'ride'")
        qr = BandQR(_to_band(A, kl, ku), kl, ku, householder=(method == "householder"))
        return qr.solve(np.ascontiguousarray(b, dt))
    if method == "lq":
        if has_rows or has_cols:
            raise ValueError("method='lq' needs a purely banded matrix "
                             "(found dense rows/cols); use 'collapse' or 'ride'")
        return LQ(A, kl, ku).solve(np.ascontiguousarray(b, dt))
    if method == "collapse":
        if has_rows and has_cols:
            return CollapseBothQR(A, kl, ku, dense_rows, dense_cols).solve(np.ascontiguousarray(b, dt))
        if has_cols:
            return CollapseColsQR(A, kl, ku, dense_cols).solve(np.ascontiguousarray(b, dt))
        return solve_precond_rows(A, kl, ku, b, dense_rows)
    if method == "ride":
        if has_cols:
            if not has_rows:
                raise NotImplementedError("'ride' needs dense rows to carry the tau columns "
                                          "against; pass dense_rows")
            return RideTauQR(A, kl, ku, dense_rows, dense_cols).solve(np.ascontiguousarray(b, dt))
        if not has_rows:
            return BandQR(_to_band(A, kl, ku), kl, ku).solve(np.ascontiguousarray(b, dt))
        return solve_ride(A, kl, ku, b, dense_rows)
    if method == "bordered":
        # Woodbury against the in-band part: handles dense rows AND tau columns.
        # Robust unless the in-band bulk is singular (differential operators) --
        # for those use 'collapse'/'ride' on the rows.
        return solve_bordered(A, kl, ku, b, dense_rows, dense_cols)
    if method == "lu":
        if has_rows or has_cols:
            raise NotImplementedError("'lu' densifies almost-banded systems; "
                                      "use 'collapse'/'ride' for borders")
        import scipy.sparse as sp
        import scipy.linalg as sla
        n = A.shape[0]
        C = A.tocoo() if sp.issparse(A) else sp.coo_matrix(np.asarray(A, dt))
        m = (C.col >= C.row - kl) & (C.col <= C.row + ku)
        ab = np.zeros((kl + ku + 1, n), dtype=dt)   # scipy banded layout
        ab[ku + C.row[m] - C.col[m], C.col[m]] = C.data[m]
        return sla.solve_banded((kl, ku), ab, np.ascontiguousarray(b, dt))
    raise NotImplementedError(f"method={method!r} is on the roadmap (coming: gpu)")


class BandQR:
    """A = Q R for a banded A, Q stored implicitly as Givens rotations.

    nthreads: 0 = auto (wavefront-thread wide bands, serial otherwise),
              1 = force serial, k = use k threads for the factorisation.
    """

    def __init__(self, ab, kl, ku, ncols=None, nthreads=0, householder=False):
        ab = np.ascontiguousarray(ab)
        self._z = np.iscomplexobj(ab)             # float64 or complex128
        self._dt = np.complex128 if self._z else np.float64
        self._suf = "_z" if self._z else "_d"
        self._p = _cptr if self._z else _ptr      # array-pointer fn for this dtype
        ab = np.ascontiguousarray(ab, dtype=self._dt)
        m, w = ab.shape
        assert w == kl + ku + 1, "band width must be kl+ku+1"
        self.m, self.kl, self.ku = m, kl, ku
        # square by default; ncols < m gives a tall least-squares factorisation
        # (used to absorb tau columns as an augmented border).
        self.n = m if ncols is None else int(ncols)
        assert self.n <= m, "need m >= ncols"
        if householder:                           # one reflector per column
            self._h = self._fn("factor_hh")(self._p(ab), m, self.n, kl, ku)
        else:                                     # Givens rotations (default)
            self._h = self._fn("factor_mt")(self._p(ab), m, self.n, kl, ku, nthreads)
        if not self._h:
            raise RuntimeError("bandqr_factor failed (need m >= n, kl,ku >= 0)")
        self.kuR = ku + kl

    def _fn(self, name):
        return getattr(_lib, "bandqr_" + name + self._suf)

    def apply_Qt(self, v):
        """Return Q^T v (Q^H v for complex) for a length-m vector."""
        v = np.array(v, dtype=self._dt, copy=True)
        self._fn("apply_Qt")(self._h, self._p(v))
        return v

    def solve_R(self, y):
        """Solve R x = y[:n] by banded back-substitution (no Q^T)."""
        y = np.ascontiguousarray(y, dtype=self._dt)
        x = np.zeros(self.n, dtype=self._dt)
        self._fn("solve_R")(self._h, self._p(y), self._p(x))
        return x

    def solve(self, b):
        """Solve for one RHS (b shape (n,)) or many (b shape (n, nrhs))."""
        b = np.ascontiguousarray(b, dtype=self._dt)
        if b.ndim == 1:
            x = np.zeros(self.n, dtype=self._dt)
            self.residual = self._fn("solve")(self._h, self._p(b), self._p(x))
            return x
        nrhs = b.shape[1]
        x = np.zeros((self.n, nrhs), dtype=self._dt)
        res = np.zeros(nrhs, dtype=np.float64)
        self._fn("solve_many")(self._h, self._p(b), nrhs, self._p(x), _ptr(res))
        self.residual = res
        return x

    def R_band(self):
        """R in row-major band form (n, kuR+1): out[i, j-i] = R[i, j]."""
        out = np.zeros((self.n, self.kuR + 1), dtype=self._dt)
        self._fn("get_R")(self._h, self._p(out))
        return out

    def __del__(self):
        if getattr(self, "_h", None):
            self._fn("free")(self._h)
            self._h = None


def band_transpose(ab, kl, ku):
    """Transpose a banded matrix in band storage: a (kl, ku) band becomes a
    (ku, kl) band, with abT[i, (j-i)+ku] = A[j, i].  O(n*(kl+ku)), no fill.
    Returns (abT, ku, kl)."""
    n = ab.shape[0]
    abT = np.zeros((n, kl + ku + 1), dtype=ab.dtype)
    for d in range(-ku, kl + 1):              # diagonal offset of A^T (= -offset of A)
        cT, cA = d + ku, kl - d
        if d >= 0:
            abT[:n - d, cT] = ab[d:, cA]
        else:
            abT[-d:, cT] = ab[:n + d, cA]
    return abT, ku, kl


def _band_matvec(ab, kl, ku, x):
    """y = A x for A in (kl, ku) band storage."""
    n = ab.shape[0]
    x = np.asarray(x)
    y = np.zeros(n, dtype=np.result_type(ab.dtype, x.dtype))
    for off in range(-kl, ku + 1):
        col = off + kl
        if off >= 0:
            y[:n - off] += ab[:n - off, col] * x[off:]
        else:
            y[-off:] += ab[-off:, col] * x[:n + off]
    return y


def _to_scipy_band(ab, kl, ku):
    """My band storage (ab[i, j-i+kl]) -> scipy solve_banded layout (absc[ku+i-j, j])."""
    n = ab.shape[0]
    absc = np.zeros((kl + ku + 1, n), dtype=ab.dtype)
    for off in range(-kl, ku + 1):            # ab[i, off+kl] -> absc[ku-off, i+off]
        col = off + kl
        if off >= 0:
            absc[ku - off, off:] = ab[:n - off, col]
        else:
            absc[ku - off, :n + off] = ab[-off:, col]
    return absc


class LQ:
    """A = L Q for a square banded A (L lower-triangular, Q with orthonormal rows),
    obtained as the QR of the conjugate transpose A^H -- a banded transpose just
    swaps the bandwidths.  `.solve(b)` solves A x = b, factored from the row side;
    the dual of QR, and the natural form for dense-column (tau) structure by
    transpose symmetry.  Real or complex, dtype following A."""

    def __init__(self, A, kl, ku):
        self.kl, self.ku = kl, ku
        ab = np.ascontiguousarray(_to_band(A, kl, ku))
        self._z = np.iscomplexobj(ab)
        abT, klT, kuT = band_transpose(ab, kl, ku)
        abH = np.conjugate(abT) if self._z else abT      # band of A^H
        self._abH, self.klH, self.kuH = abH, klT, kuT
        self.qrT = BandQR(abH, klT, kuT)                 # A^H = Q_t R_t
        self.n = self.qrT.n

    def L_band(self):
        """L = R_t^H in band storage (kl=kuR, ku=0): lower-triangular."""
        kuR = self.qrT.kuR
        RbandT, _, _ = band_transpose(self.qrT.R_band(), 0, kuR)
        return (np.conjugate(RbandT) if self._z else RbandT), kuR

    def solve(self, b):
        import scipy.linalg as sla
        kuR = self.qrT.kuR
        b = np.ascontiguousarray(b, dtype=self.qrT._dt)
        RbandH, _ = self.L_band()                        # R_t^H band (lower-triangular)
        u = sla.solve_banded((kuR, 0), _to_scipy_band(RbandH, kuR, 0), b)  # R_t^{-H} b
        z = self.qrT.solve_R(u)                          # R_t^{-1} u
        return _band_matvec(self._abH, self.klH, self.kuH, z)   # A^H z


# ---------------------------------------------------------------------------
# Almost-banded factor-once / solve-many: reusable factorisation objects.
# ---------------------------------------------------------------------------

class RideQR:
    """Reusable almost-banded QR (Olver-Townsend 'ride'): factor once, solve
    many.  Dense rows carried along; well-conditioned.  Real or complex."""

    def __init__(self, M, kl, ku, dense_rows):
        import scipy.sparse as sp
        n = M.shape[0]
        dr = np.ascontiguousarray(np.asarray(dense_rows), dtype=np.int32)
        r = len(dr)
        self.n = n
        self._z = _is_complex(M)
        self._dt = np.complex128 if self._z else np.float64
        self._suf = "_z" if self._z else "_d"
        self._p = _cptr if self._z else _ptr
        kle = max(kl, int(dr.max()))
        ab = np.ascontiguousarray(_to_band(M, kle, ku), self._dt)
        L = np.asarray(sp.csr_matrix(M)[dr, :].todense()).astype(self._dt)
        for t in range(r):
            i = int(dr[t]); lo, hi = max(0, i - kle), min(n - 1, i + ku)
            L[t, lo:hi + 1] = 0
        L = np.ascontiguousarray(L, self._dt)
        self._h = getattr(_lib, "bandqr_factor_ab" + self._suf)(
            self._p(ab), n, n, kle, ku, dr.ctypes.data_as(ct.POINTER(ct.c_int)), r, self._p(L))
        if not self._h:
            raise RuntimeError("ride factor failed (need kl,ku >= 0, r >= 1)")

    def solve(self, b):
        b = np.ascontiguousarray(b, self._dt)
        single = b.ndim == 1
        B = b[:, None] if single else b
        X = np.zeros((self.n, B.shape[1]), dtype=self._dt)
        sfn = getattr(_lib, "bandqr_solve_ab" + self._suf)
        for k in range(B.shape[1]):
            bk = np.ascontiguousarray(B[:, k]); xk = np.zeros(self.n, dtype=self._dt)
            sfn(self._h, self._p(bk), self._p(xk))
            X[:, k] = xk
        return X[:, 0] if single else X

    def __del__(self):
        if getattr(self, "_h", None):
            getattr(_lib, "bandqr_free" + self._suf)(self._h)
            self._h = None


class RideTauQR:
    """The full Dedalus almost-banded structure: dense BC rows AND sparse / low-rank
    tau columns.  Ride carries the dense rows; the tau columns are removed to unit
    columns (so the band stays nonsingular) and restored by a Woodbury update.
    Factor once, solve many.  Real or complex.

    Writing M = M_R + G W with M_R the ride-able matrix (tau columns dc set to e_c),
    G = (true tau columns) - e_c and W = e_dc^T, the solve is
        x = y - Z (I + Z[dc])^{-1} y[dc],   y = M_R^{-1} b,   Z = M_R^{-1} G,
    with the small dc x dc system and Z formed once at factor time."""

    def __init__(self, M, kl, ku, dense_rows, dense_cols, tol=1e-8):
        import scipy.sparse as sp, scipy.linalg as sla
        n = M.shape[0]
        dr = np.asarray(dense_rows, dtype=int)
        dc = np.asarray(dense_cols, dtype=int)
        self.n = n
        self._z = _is_complex(M)
        self._dt = np.complex128 if self._z else np.float64
        # ride needs the dense BC rows at the top -- permute them there (x unchanged)
        rest = np.setdiff1d(np.arange(n), dr)              # complement, ascending (vectorized)
        self.perm = np.concatenate([dr, rest]).astype(int)
        Mp = sp.csr_matrix(M)[self.perm, :].astype(self._dt, copy=False)   # stay in CSR
        r_rows = len(dr)
        # tau columns -> unit columns (M_R), and G = (true tau columns) - e_c
        G = np.asarray(Mp[:, dc].todense()).astype(self._dt)   # n x r_c
        for k, c in enumerate(int(x) for x in dc):
            G[c, k] -= 1.0
        keep = np.ones(n, dtype=self._dt); keep[dc] = 0.0       # zero the tau columns
        E = sp.csr_matrix((np.ones(len(dc), dtype=self._dt), (dc, dc)), shape=(n, n))
        Mr = (Mp @ sp.diags(keep) + E).tocsr()                  # + unit columns at dc
        akl, aku, _, _ = detect_structure(Mr)
        # ride (not collapse) for M_R: setting tau columns to e_c breaks the dense-row
        # recurrence collapse depends on, but ride carries the rows without one.
        self.inner = RideQR(Mr, akl, aku, np.arange(r_rows))
        self.dc = dc
        self.Z = self.inner.solve(G)                      # M_R^{-1} G  (n, r_c)
        small = np.eye(len(dc), dtype=self._dt) + self.Z[dc, :]   # I + W M_R^{-1} G
        self.small_lu = sla.lu_factor(small)

    def solve(self, b):
        import scipy.linalg as sla
        b = np.ascontiguousarray(np.asarray(b)[self.perm], self._dt)
        y = self.inner.solve(b)
        c = sla.lu_solve(self.small_lu, y[self.dc])
        return y - self.Z @ c


class CollapseQR:
    """Reusable recurrence-preconditioned solve ('collapse'): the band QR of the
    sparsified operator is built once; solve many applies it + the preconditioner.

    The right-preconditioner P (with M @ P banded and x = P @ y) is represented
    compactly by its recurrence coefficients `.a`.  Pass store_P=True to also retain
    the explicit sparse matrix as `.P`; `.preconditioner()` returns it either way
    (rebuilding from `.a` on demand if it was not stored), and `.apply_P(y)` applies
    P to an arbitrary vector or block."""

    def __init__(self, M, kl, ku, dense_rows, max_order=20, tol=1e-8, store_P=False):
        import scipy.sparse as sp
        n = M.shape[0]
        dr = np.asarray(dense_rows, dtype=int)
        Msp = sp.csr_matrix(M)
        a = detect_recurrence(np.asarray(Msp[dr, :].todense()), max_order, tol)
        if a is None:
            raise RuntimeError("no recurrence found for the dense rows (raise max_order?)")
        P = _build_P(a, n)
        MP = (Msp @ P).tocoo()
        keep = np.abs(MP.data) > tol * np.abs(MP.data).max()
        r_, c_, d_ = MP.row[keep], MP.col[keep], MP.data[keep]
        off = c_ - r_
        kl_eff, ku_eff = int(-off.min()), int(off.max())
        ab = np.zeros((n, kl_eff + ku_eff + 1), dtype=d_.dtype)
        ab[r_, c_ - r_ + kl_eff] = d_
        self.qr = BandQR(ab, kl_eff, ku_eff)        # reusable band factorisation
        self.a = a                                   # recurrence coefficients of P
        self.n = n
        self.P = P if store_P else None              # explicit preconditioner (optional)

    def apply_P(self, y):
        """Apply the right-preconditioner: x = P @ y, via the recurrence (in C)."""
        return _apply_P(self.a, y)

    def preconditioner(self):
        """The explicit sparse right-preconditioner P (M @ P banded, x = P @ y);
        returns the stored matrix if store_P was set, else rebuilds it from `.a`."""
        return self.P if self.P is not None else _build_P(self.a, self.n)

    def solve(self, b):
        return _apply_P(self.a, self.qr.solve(b))


class CollapseColsQR:
    """Dense-column collapse -- the transpose-dual of CollapseQR.  A left-preconditioner
    Q, built from the recurrence the dense columns satisfy, sparsifies them into the
    band, so M x = b becomes a single banded QR: (Q M) x = Q b.  Q is algebraic (not
    orthogonal), so a plain transpose suffices for complex.  The recurrence is in `.a`
    and the explicit left-preconditioner in `.Q`.

    NB the columns must be *dense and recurrence-structured*.  Dedalus tau columns are
    instead sparse / low-rank (a single lifted mode) and are not handled here -- carry
    those along with 'ride'/'bordered'."""

    def __init__(self, M, kl, ku, dense_cols, max_order=20, tol=1e-8):
        import scipy.sparse as sp
        n = M.shape[1]
        dc = np.asarray(dense_cols, dtype=int)
        if dc.size == 0:
            raise ValueError("CollapseColsQR needs at least one dense column")
        Msp = sp.csr_matrix(M)
        # recurrence satisfied by the dense columns (i.e. the dense rows of M^T)
        a = detect_recurrence(np.asarray(Msp[:, dc].T.todense()), max_order, tol)
        if a is None:
            raise RuntimeError("no recurrence found for the dense columns (raise max_order?)")
        Q = _build_P(a, n).T.tocsr()                # left preconditioner: Q M is banded
        QM = (Q @ Msp).tocoo()
        keep = np.abs(QM.data) > tol * np.abs(QM.data).max()
        r_, c_, d_ = QM.row[keep], QM.col[keep], QM.data[keep]
        off = c_ - r_
        kl_eff, ku_eff = int(-off.min()), int(off.max())
        ab = np.zeros((n, kl_eff + ku_eff + 1), dtype=d_.dtype)
        ab[r_, c_ - r_ + kl_eff] = d_
        self.qr = BandQR(ab, kl_eff, ku_eff)        # reusable band factorisation
        self.a = a                                   # recurrence coefficients
        self.n = n
        self.Q = Q                                   # left preconditioner (applied each solve)

    def solve(self, b):
        return self.qr.solve(self.Q @ np.asarray(b))


class CollapseBothQR:
    """Doubly-bordered collapse: dense BC rows AND dense columns at once.  A right-
    preconditioner P (row recurrence) and a left-preconditioner Q (column recurrence)
    together sparsify M into a band -- Q M P is banded -- so M x = b becomes a single
    banded QR:  x = P (Q M P)^{-1} Q b.  Real or complex.

    NB both borders must be *dense and recurrence-structured*.  This is NOT the Dedalus
    tau-column structure: there the columns are sparse / low-rank, so collapse drops
    them and the band goes singular -- carry them along with 'ride'/'bordered' instead."""

    def __init__(self, M, kl, ku, dense_rows, dense_cols, max_order=20, tol=1e-8):
        import scipy.sparse as sp
        n = M.shape[0]
        dr = np.asarray(dense_rows, dtype=int)
        dc = np.asarray(dense_cols, dtype=int)
        if dr.size == 0 or dc.size == 0:
            raise ValueError("CollapseBothQR needs at least one dense row and one dense column")
        Msp = sp.csr_matrix(M)
        a_r = detect_recurrence(np.asarray(Msp[dr, :].todense()), max_order, tol)
        a_c = detect_recurrence(np.asarray(Msp[:, dc].T.todense()), max_order, tol)
        if a_r is None or a_c is None:
            raise RuntimeError("no recurrence found for the dense rows/columns (raise max_order?)")
        P = _build_P(a_r, n)                          # right precond: sparsifies dense rows
        Q = _build_P(a_c, n).T.tocsr()               # left precond: sparsifies dense cols
        B = (Q @ Msp @ P).tocoo()
        keep = np.abs(B.data) > tol * np.abs(B.data).max()
        r_, c_, d_ = B.row[keep], B.col[keep], B.data[keep]
        off = c_ - r_
        kl_eff, ku_eff = int(-off.min()), int(off.max())
        ab = np.zeros((n, kl_eff + ku_eff + 1), dtype=d_.dtype)
        ab[r_, c_ - r_ + kl_eff] = d_
        self.qr = BandQR(ab, kl_eff, ku_eff)         # reusable band factorisation
        self.a_r = a_r                               # row recurrence (P, applied after solve)
        self.Q = Q                                   # column preconditioner (applied to b)
        self.n = n

    def solve(self, b):
        return _apply_P(self.a_r, self.qr.solve(self.Q @ np.asarray(b)))


def factorize(A, method="auto", kl=None, ku=None, dense_rows=None, dense_cols=None,
              store_P=False):
    """Factor once, solve many: returns an object with a .solve(b) method.
    Same method/structure dispatch as solve(); see solve() for the flags.
    store_P=True (collapse only) retains the explicit sparse preconditioner on
    the returned object (`.P` / `.preconditioner()`)."""
    if method not in _METHODS or method == "lu":
        raise ValueError(f"factorize supports {_METHODS[:-1]} (not 'lu'); got {method!r}")
    akl, aku, adr, adc = detect_structure(A)
    kl = akl if kl is None else int(kl)
    ku = aku if ku is None else int(ku)
    dense_rows = adr if dense_rows is None else np.asarray(dense_rows, dtype=int)
    dense_cols = adc if dense_cols is None else np.asarray(dense_cols, dtype=int)
    has_rows, has_cols = len(dense_rows) > 0, len(dense_cols) > 0
    if method == "auto":
        if not has_rows and not has_cols:
            method = "householder" if kl >= 6 else "givens"
        else:
            method = "collapse"          # any borders -> recurrence collapse
    if method in ("givens", "householder"):
        return BandQR(_to_band(A, kl, ku), kl, ku, householder=(method == "householder"))
    if method == "lq":
        return LQ(A, kl, ku)
    if method == "collapse":
        if has_rows and has_cols:
            return CollapseBothQR(A, kl, ku, dense_rows, dense_cols)
        if has_cols:
            return CollapseColsQR(A, kl, ku, dense_cols)
        return CollapseQR(A, kl, ku, dense_rows, store_P=store_P)
    if method == "ride":
        if has_cols:
            if not has_rows:
                raise NotImplementedError("'ride' needs dense rows to carry the tau columns against")
            return RideTauQR(A, kl, ku, dense_rows, dense_cols)
        if not has_rows:
            return BandQR(_to_band(A, kl, ku), kl, ku)
        return RideQR(A, kl, ku, dense_rows)
    raise NotImplementedError(f"factorize method={method!r} is on the roadmap")
