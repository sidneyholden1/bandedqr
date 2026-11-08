### CPU size scaling

Fixed `kl = ku = 4`, one thread; median factor + solve time.

| n | bandedqr Givens | bandedqr Householder | LAPACK band LU | max residual |
|---:|---:|---:|---:|---:|
| 1,024 | 0.11 ms | 0.084 ms | 0.074 ms | 3.4e-16 |
| 4,096 | 0.42 ms | 0.31 ms | 0.26 ms | 3.4e-16 |
| 16,384 | 1.64 ms | 1.20 ms | 0.99 ms | 3.4e-16 |
| 65,536 | 6.54 ms | 4.75 ms | 3.94 ms | 3.5e-16 |
| 262,144 | 26.9 ms | 20.0 ms | 16.9 ms | 3.5e-16 |

### CPU thread scaling

Fixed `n = 32,768`, `kl = ku = 64`; median Givens factor + solve time. Speedup is relative to one thread.

| threads | time | speedup | residual |
|---:|---:|---:|---:|
| 1 | 153.0 ms | 1.00× | 1.2e-15 |
| 2 | 281.7 ms | 0.54× | 1.2e-15 |
| 4 | 170.5 ms | 0.90× | 1.2e-15 |
| 8 | 128.3 ms | 1.19× | 1.2e-15 |
| 16 | 124.1 ms | 1.23× | 1.2e-15 |
| 32 | 124.2 ms | 1.23× | 1.2e-15 |

### CPU factor reuse

Fixed `n = 65,536`, `kl = ku = 8`; median solve time with an already-factorized matrix.

| right-hand sides | total solve | time / RHS | max residual |
|---:|---:|---:|---:|
| 1 | 4.57 ms | 4.57 ms | 4.7e-16 |
| 8 | 6.01 ms | 0.75 ms | 4.7e-16 |
| 32 | 12.0 ms | 0.38 ms | 4.7e-16 |

### CUDA batch scaling

Fixed `n = 256`, `kl = ku = 4`. CUDA refined timings include host/device transfers, factorization, solve, and mixed-precision refinement; the resident column is the raw solve with factors and RHS already on device.

| batch | CPU NumPy factor + solve | CUDA refined end-to-end | speedup | CUDA resident re-solve | refined residual |
|---:|---:|---:|---:|---:|---:|
| 64 | 31.1 ms | 3.10 ms | 10.0× | 0.46 ms | 3.2e-14 |
| 256 | 40.9 ms | 3.70 ms | 11.0× | 0.48 ms | 4.1e-14 |
| 1,024 | 79.6 ms | 7.20 ms | 11.1× | 0.48 ms | 4.2e-14 |
| 4,096 | 268.9 ms | 22.8 ms | 11.8× | 1.00 ms | 4.5e-14 |
| 16,384 | 836.9 ms | 109.6 ms | 7.6× | 1.06 ms | 4.9e-14 |
