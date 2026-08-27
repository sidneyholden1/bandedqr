/* bandqr -- banded QR factorisation by Givens rotations.
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
