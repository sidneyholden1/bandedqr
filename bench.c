/* Standalone timing harness, no Python.  Build native arm64 and x86_64 to see
 * the algorithm's true speed vs the Rosetta-translation tax. */
#include "bandqr.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double wall(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static unsigned long rng = 88172645463325252UL;
static double rnd(void) {                 /* xorshift, no libc dependence */
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (double)(rng >> 11) / 9007199254740992.0 - 0.5;
}

static void run(int n, int kl, int ku, int reps) {
    int W = kl + ku + 1;
    double *ab = calloc((size_t)n * W, sizeof(double));
    double *b  = malloc((size_t)n * sizeof(double));
    double *x  = malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int o = 0; o < W; o++) {
            int j = i - kl + o;
            if (j >= 0 && j < n) ab[i * W + o] = rnd();
        }
        ab[i * W + kl] += kl + ku + 2;     /* diagonally dominant */
        b[i] = rnd();
    }
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = wall();
        band_qr_d *qr = bandqr_factor_d(ab, n, n, kl, ku);
        bandqr_solve_d(qr, b, x);
        bandqr_free_d(qr);
        double dt = wall() - t0;
        if (dt < best) best = dt;
    }
    double flops = 6.0 * n * kl * (kl + ku);     /* ~ rotations * work */
    printf("n=%7d kl=%3d ku=%3d | %9.3f ms | %6.2f Gflop/s\n",
           n, kl, ku, best * 1e3, flops / best / 1e9);
    free(ab); free(b); free(x);
}

int main(void) {
    int cases[][3] = {{1000,2,2},{5000,4,4},{20000,8,8},{100000,8,8},{50000,32,32},{20000,64,64}};
    for (unsigned k = 0; k < sizeof(cases)/sizeof(cases[0]); k++)
        run(cases[k][0], cases[k][1], cases[k][2], 10);
    return 0;
}
