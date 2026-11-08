/* bandedqr -- banded and almost-banded QR factorisation.
 *
 * For an m x n matrix A with lower bandwidth kl and upper bandwidth ku
 * (m >= n), A = Q R with R upper triangular of upper bandwidth ku+kl and Q held
 * implicitly (as plane rotations, or as Householder reflectors).  Work is
 * O(n*kl*(kl+ku)) with no dense fill and no pivoting, so the cost is a function
 * of the sparsity pattern alone -- never of the values.
 *
 * The almost-banded variant (factor_ab) carries a few dense rows through the
 * factorisation as a rank-r update, M = B + U L, keeping R almost-banded too.
 *
 * Two scalar types, generated from one template (bandedqr_impl.h): a _d (float64)
 * and a _z (complex128) variant of every routine.  The handle is opaque; pick
 * the variant by your data's dtype.
 *
 * Build with -DBANDEDQR_NO_THREADS to drop the pthread/C11-atomics dependency
 * (factor_mt then falls back to the serial factorisation).
 */
#ifndef BANDEDQR_H
#define BANDEDQR_H

typedef struct band_qr_d band_qr_d;   /* opaque, float64    */
typedef struct band_qr_z band_qr_z;   /* opaque, complex128 */

/* 1 if built with threading support, 0 if -DBANDEDQR_NO_THREADS. */
int bandedqr_has_threads(void);

/* ---- float64 ---- */
band_qr_d *bandedqr_factor_d(const double *ab, int m, int n, int kl, int ku);
band_qr_d *bandedqr_factor_mt_d(const double *ab, int m, int n, int kl, int ku, int nthreads);
band_qr_d *bandedqr_factor_hh_d(const double *ab, int m, int n, int kl, int ku);
band_qr_d *bandedqr_factor_ab_d(const double *ab, int m, int n, int kl, int ku,
                              const int *dr, int r, const double *L);
double bandedqr_solve_ab_d(const band_qr_d *qr, const double *b, double *x);
void bandedqr_solve_many_ab_d(const band_qr_d *qr, const double *B, int nrhs,
                            double *X, double *res);
double bandedqr_solve_d(const band_qr_d *qr, const double *b, double *x);
void bandedqr_solve_many_d(const band_qr_d *qr, const double *B, int nrhs, double *X, double *res);
void bandedqr_apply_Qt_d(const band_qr_d *qr, double *v);
void bandedqr_apply_Q_d(const band_qr_d *qr, double *v);
void bandedqr_solve_R_d(const band_qr_d *qr, const double *y, double *x);
void bandedqr_solve_Rt_d(const band_qr_d *qr, const double *y, double *x);
void bandedqr_apply_P_d(const double *a, int p, const double *y, int n, double *x);
void bandedqr_band_matvec_d(const double *ab, int m, int n, int kl, int ku,
                          const double *x, double *y);
void bandedqr_band_residual_batched_d(const double *ab, int n, int w, int batch,
                                    int kl, int ku, const double *x,
                                    const double *b, double *r);
void bandedqr_get_R_d(const band_qr_d *qr, double *out);
void bandedqr_free_d(band_qr_d *qr);

/* ---- complex128 ---- */
band_qr_z *bandedqr_factor_z(const double _Complex *ab, int m, int n, int kl, int ku);
band_qr_z *bandedqr_factor_mt_z(const double _Complex *ab, int m, int n, int kl, int ku, int nthreads);
band_qr_z *bandedqr_factor_hh_z(const double _Complex *ab, int m, int n, int kl, int ku);
band_qr_z *bandedqr_factor_ab_z(const double _Complex *ab, int m, int n, int kl, int ku,
                              const int *dr, int r, const double _Complex *L);
double bandedqr_solve_ab_z(const band_qr_z *qr, const double _Complex *b, double _Complex *x);
void bandedqr_solve_many_ab_z(const band_qr_z *qr, const double _Complex *B, int nrhs,
                            double _Complex *X, double *res);
double bandedqr_solve_z(const band_qr_z *qr, const double _Complex *b, double _Complex *x);
void bandedqr_solve_many_z(const band_qr_z *qr, const double _Complex *B, int nrhs, double _Complex *X, double *res);
void bandedqr_apply_Qt_z(const band_qr_z *qr, double _Complex *v);
void bandedqr_apply_Q_z(const band_qr_z *qr, double _Complex *v);
void bandedqr_solve_R_z(const band_qr_z *qr, const double _Complex *y, double _Complex *x);
void bandedqr_solve_Rt_z(const band_qr_z *qr, const double _Complex *y, double _Complex *x);
void bandedqr_apply_P_z(const double _Complex *a, int p, const double _Complex *y, int n, double _Complex *x);
void bandedqr_band_matvec_z(const double _Complex *ab, int m, int n, int kl, int ku,
                          const double _Complex *x, double _Complex *y);
void bandedqr_band_residual_batched_z(const double _Complex *ab, int n, int w, int batch,
                                    int kl, int ku, const double _Complex *x,
                                    const double _Complex *b, double _Complex *r);
void bandedqr_get_R_z(const band_qr_z *qr, double _Complex *out);
void bandedqr_free_z(band_qr_z *qr);

#endif /* BANDEDQR_H */
