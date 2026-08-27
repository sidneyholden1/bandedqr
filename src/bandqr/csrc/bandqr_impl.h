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
                            bandqr_front_t *front)
{
    int m = qr->m, n = qr->n, kl = qr->kl, kuR = qr->kuR, W = qr->W;
    SCALAR *band = qr->band;
    int ilast = ilast_of(jcol, kl, m);
    long r = 0;
#ifdef BANDQR_NO_THREADS
    (void)front;
#else
    int since = 0;
#endif
    for (int q = ilast; q >= jcol + 1; q--, r++) {
#ifndef BANDQR_NO_THREADS
        if (front && jcol > 0)
            while (atomic_load_explicit(&front[jcol - 1].v, memory_order_acquire) > q - 1)
                cpu_relax();
#endif
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
#ifndef BANDQR_NO_THREADS
        if (front && ++since >= BANDQR_FRONT_CHUNK) {
            atomic_store_explicit(&front[jcol].v, q, memory_order_release);
            since = 0;
        }
#endif
    }
#ifndef BANDQR_NO_THREADS
    if (front)
        atomic_store_explicit(&front[jcol].v, -1, memory_order_release);
#endif
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

#ifndef BANDQR_NO_THREADS
typedef struct {
    struct SUF(band_qr) *qr;
    const long      *off;
    bandqr_front_t  *front;
    _Atomic int     *next_col;
    int              n;
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
#endif

struct SUF(band_qr) *SUF(bandqr_factor_mt)(const SCALAR *ab, int m, int n,
                                           int kl, int ku, int nthreads)
{
#ifdef BANDQR_NO_THREADS
    (void)nthreads;
    return SUF(bandqr_factor)(ab, m, n, kl, ku);
#else
    if (nthreads <= 1)                        /* 0 and 1 both mean serial */
        return SUF(bandqr_factor)(ab, m, n, kl, ku);

    long *off, total;
    struct SUF(band_qr) *qr = SUF(alloc_and_load)(ab, m, n, kl, ku, &off, &total);
    if (!qr) return NULL;

    bandqr_front_t *front = calloc((size_t)n, sizeof *front);
    pthread_t      *th    = NULL;
    if (!front) { free(off); SUF(bandqr_free)(qr); return NULL; }
    for (int j = 0; j < n; j++)
        atomic_init(&front[j].v, ilast_of(j, kl, m) + 1);

    _Atomic int next_col;
    atomic_init(&next_col, 0);
    if (nthreads > n) nthreads = n;

    SUF(worker_arg) wa = { qr, off, front, &next_col, n };
    th = malloc((size_t)nthreads * sizeof *th);
    if (!th) { free(front); free(off); SUF(bandqr_free)(qr); return NULL; }
    int spawned = 0;
    for (int t = 0; t < nthreads; t++)
        if (pthread_create(&th[t], NULL, SUF(worker), &wa) == 0) spawned++;
        else break;
    if (spawned == 0)                         /* no threads: run it here */
        SUF(worker)(&wa);
    for (int t = 0; t < spawned; t++) pthread_join(th[t], NULL);

    free(th); free(front); free(off);
    return qr;
#endif
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

/* v (length m) <- Q v -- the inverse of bandqr_apply_Qt.
 * Givens:  Q^H = G_{nrot-1} .. G_0, so Q = G_0^H .. G_{nrot-1}^H: replay the
 *          rotations in reverse, each conjugate-transposed.
 * Householder: Q^H applies H_0^H .. H_{n-1}^H in increasing j, so
 *          Q = H_0 .. H_{n-1}: apply H_j (tau, not conj tau) in decreasing j. */
void SUF(bandqr_apply_Q)(const struct SUF(band_qr) *qr, SCALAR *v)
{
    if (qr->reflectors) {
        int kl = qr->kl, m = qr->m, klp = kl + 1;
        for (int j = qr->n - 1; j >= 0; j--) {
            SCALAR tau = qr->tau[j];
            if (tau == 0) continue;
            int L = j + kl < m ? kl + 1 : m - j;
            const SCALAR *vj = &qr->V[(size_t)j * klp];
            SCALAR w = 0;
            for (int i = 0; i < L; i++) w += CONJ(vj[i]) * v[j + i];
            w *= tau;                          /* H, not H^H */
            for (int i = 0; i < L; i++) v[j + i] -= vj[i] * w;
        }
        return;
    }
    for (long k = qr->nrot - 1; k >= 0; k--) {
        int p = qr->p[k]; REAL c = qr->c[k]; SCALAR s = qr->s[k];
        SCALAR vp = v[p], vq = v[p + 1];
        v[p]     = c * vp - s * vq;
        v[p + 1] = CONJ(s) * vp + c * vq;
    }
}

/* Solve R^H x = y by forward substitution.  R is upper triangular with upper
 * bandwidth kuR, so R^H is lower triangular with lower bandwidth kuR:
 *     x[i] = (y[i] - sum_{j=i-kuR}^{i-1} conj(R[j][i]) x[j]) / conj(R[i][i]).
 * Together with apply_Q this gives A^{-H} v = Q R^{-H} v, the second operator a
 * 1-norm condition estimator needs. */
void SUF(bandqr_solve_Rt)(const struct SUF(band_qr) *qr, const SCALAR *y, SCALAR *x)
{
    const SCALAR *restrict band = qr->band;
    int n = qr->n, W = qr->W, kuR = qr->kuR, kl = qr->kl;
    for (int i = 0; i < n; i++) {
        int jlo = i - kuR > 0 ? i - kuR : 0;
        SCALAR sum = 0;
        for (int j = jlo; j < i; j++)
            sum += CONJ(band[(size_t)j * W + (i - j + kl)]) * x[j];
        x[i] = (y[i] - sum) / CONJ(band[(size_t)i * W + kl]);
    }
}

/* Multi-RHS almost-banded solve: Q^H B once, then a blocked back-substitution
 * that carries the low-rank fill as an r x nrhs running block.  Amortises the
 * rotation replay across right-hand sides (the factor-once/solve-many path, and
 * what iterative refinement leans on). */
void SUF(bandqr_solve_many_ab)(const struct SUF(band_qr) *qr, const SCALAR *B,
                               int nrhs, SCALAR *X, double *res)
{
    int m = qr->m, n = qr->n, W = qr->W, kuR = qr->kuR, kl = qr->kl, r = qr->rfill;
    const SCALAR *restrict band = qr->band, *restrict U = qr->U, *restrict L = qr->L;
    SCALAR *Y = malloc((size_t)m * nrhs * sizeof(SCALAR));
    SCALAR *g = calloc((size_t)r * nrhs, sizeof(SCALAR));
    if (!Y || !g) { free(Y); free(g); return; }
    memcpy(Y, B, (size_t)m * nrhs * sizeof(SCALAR));

    for (long k = 0; k < qr->nrot; k++) {              /* Y <- Q^H B */
        int p = qr->p[k]; REAL c = qr->c[k]; SCALAR s = qr->s[k];
        SCALAR *restrict yp = &Y[(size_t)p * nrhs];
        SCALAR *restrict yq = &Y[(size_t)(p + 1) * nrhs];
        for (int t = 0; t < nrhs; t++) {
            SCALAR ap = yp[t], aq = yq[t];
            yp[t] = c * ap + s * aq;
            yq[t] = c * aq - CONJ(s) * ap;
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        int jhi = i + kuR < n - 1 ? i + kuR : n - 1;
        const SCALAR *restrict Rrow = &band[(size_t)i * W + (kl + 1)];
        const SCALAR *restrict Ui   = &U[(size_t)i * r];
        SCALAR       *restrict xi   = &X[(size_t)i * nrhs];
        const SCALAR *restrict yi   = &Y[(size_t)i * nrhs];
        for (int t = 0; t < nrhs; t++) xi[t] = yi[t];
        for (int d = 1; d <= jhi - i; d++) {           /* banded part */
            SCALAR Rid = Rrow[d - 1];
            const SCALAR *restrict xd = &X[(size_t)(i + d) * nrhs];
            for (int t = 0; t < nrhs; t++) xi[t] -= Rid * xd[t];
        }
        SCALAR diag = band[(size_t)i * W + kl];
        for (int u = 0; u < r; u++) {                  /* low-rank fill */
            SCALAR Uiu = Ui[u];
            const SCALAR *restrict gu = &g[(size_t)u * nrhs];
            for (int t = 0; t < nrhs; t++) xi[t] -= Uiu * gu[t];
            diag += Uiu * L[(size_t)u * n + i];
        }
        SCALAR dinv = 1.0 / diag;
        for (int t = 0; t < nrhs; t++) xi[t] *= dinv;
        for (int u = 0; u < r; u++) {                  /* g += L[:,i] x[i] */
            SCALAR Lui = L[(size_t)u * n + i];
            SCALAR *restrict gu = &g[(size_t)u * nrhs];
            for (int t = 0; t < nrhs; t++) gu[t] += Lui * xi[t];
        }
    }
    if (res)
        for (int t = 0; t < nrhs; t++) {
            REAL rr = 0;
            for (int i = n; i < m; i++) rr += ABS2(Y[(size_t)i * nrhs + t]);
            res[t] = sqrt(rr);
        }
    free(g); free(Y);
}

/* y <- A x for A in the caller's compact (kl,ku) band storage, ab[i*(kl+ku+1) +
 * (j-i+kl)].  Independent of any factorisation -- this is the original operator,
 * which iterative refinement needs to form residuals against. */
void SUF(bandqr_band_matvec)(const SCALAR *ab, int m, int n, int kl, int ku,
                             const SCALAR *restrict x, SCALAR *restrict y)
{
    int Win = kl + ku + 1;
    for (int i = 0; i < m; i++) {
        int jlo = i - kl < 0 ? 0 : i - kl;
        int jhi = i + ku > n - 1 ? n - 1 : i + ku;
        SCALAR s = 0;
        for (int j = jlo; j <= jhi; j++)
            s += ab[(size_t)i * Win + (j - i + kl)] * x[j];
        y[i] = s;
    }
}

/* Batched banded residual: r = b - A x, for `batch` independent systems held in
 * the batch-fastest layout the GPU backend uses -- ab[(i*w+o)*batch + s],
 * x[i*batch + s].
 *
 * This is the accuracy half of the mixed-precision GPU path: the factorisation
 * runs in fp32 on the GPU, and refinement is only worth anything if the
 * residual is formed against the *original* float64 operator.  Doing it here
 * rather than in numpy matters -- the obvious vectorised expression allocates a
 * whole (n, batch) temporary per diagonal and lands well short of memory
 * bandwidth.  Accumulating a row of the output across all its diagonals before
 * moving on keeps that row in cache and touches each array exactly once. */
void SUF(bandqr_band_residual_batched)(const SCALAR *ab, int n, int w, int batch,
                                       int kl, int ku, const SCALAR *restrict x,
                                       const SCALAR *restrict b,
                                       SCALAR *restrict r)
{
    for (int i = 0; i < n; i++) {
        SCALAR *restrict ri = &r[(size_t)i * batch];
        const SCALAR *restrict bi = &b[(size_t)i * batch];
        for (int s = 0; s < batch; s++) ri[s] = bi[s];
        int olo = i - kl < 0 ? kl - i : 0;             /* clip to valid columns */
        int ohi = i + ku > n - 1 ? kl + (n - 1 - i) : kl + ku;
        for (int o = olo; o <= ohi; o++) {
            int j = i - kl + o;
            const SCALAR *restrict a = &ab[((size_t)i * w + o) * batch];
            const SCALAR *restrict xj = &x[(size_t)j * batch];
            for (int s = 0; s < batch; s++) ri[s] -= a[s] * xj[s];
        }
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
