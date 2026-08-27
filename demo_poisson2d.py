"""2D Poisson, Fourier(x) x Chebyshev(z), TAU-FREE, solved with bandqr QR.

    laplacian(u) = f  on  [0, Lx) x [-1, 1],  periodic in x,  u = 0 at z = +-1.

FFT in x decouples the modes; each wavenumber k gives a 1D Helmholtz in z,
    u'' - k^2 u = f_hat_k ,  u(+-1) = 0 ,
posed in the ultraspherical (Olver-Townsend) basis: a banded operator plus two
dense boundary rows -- NO tau variables/columns.  Each mode is a banded almost-
banded solve, done with bandqr's QR ('collapse' or 'ride').  Verified against a
manufactured exact solution.
"""
import numpy as np
import scipy.sparse as sp
from bandqr import factorize

# ------------------------------------------------------------------ z: Chebyshev
def cheb(Nz):
    """Gauss-Lobatto nodes and the T-coeff <-> values transform (dense, exact)."""
    z = np.cos(np.pi * np.arange(Nz) / (Nz - 1))              # z_j = cos(pi j/(Nz-1))
    k = np.arange(Nz)
    V = np.cos(np.outer(np.arccos(z), k))                     # V[j,k] = T_k(z_j)
    return z, V, np.linalg.inv(V)                             # values = V@coeffs; coeffs = Vinv@values

def ultra_ops(n):
    """Ultraspherical operators (T-coeff space): 2nd derivative D2 and conversions."""
    j = np.arange(n)
    D2 = sp.diags(2.0 * (np.arange(n - 2) + 2), 2, shape=(n, n))          # T -> C^(2)
    S0 = sp.diags(np.r_[1.0, 0.5 * np.ones(n - 1)], 0) + sp.diags(-0.5 * np.ones(n - 2), 2, shape=(n, n))
    S1 = sp.diags(1.0 / (1.0 + j), 0) + sp.diags(-1.0 / (np.arange(n - 2) + 3.0), 2, shape=(n, n))
    return D2, S0, S1

def helmholtz_M(c, D2, S0, S1, n):
    """Almost-banded system for u'' - c u = f, u(+-1)=0: 2 dense BC rows + banded bulk."""
    L = (D2 - c * (S1 @ S0)).tocoo()                          # banded operator T -> C^(2)
    m = L.row < n - 2                                         # keep C^(2) rows 0..n-3
    rows = np.r_[np.zeros(n, int), np.ones(n, int), L.row[m] + 2]   # BC rows on top
    cols = np.r_[np.arange(n), np.arange(n), L.col[m]]
    data = np.r_[np.ones(n), (-1.0) ** np.arange(n), L.data[m]]     # T_k(+1)=1, T_k(-1)=(-1)^k
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

# ------------------------------------------------------------------ 2D solve
def poisson2d(fgrid, Lx, z, V, Vinv, method="collapse"):
    """Solve laplacian(u)=f (u=0 on walls) for f sampled on the (z, x) grid."""
    Nz, Nx = fgrid.shape
    D2, S0, S1 = ultra_ops(Nz)
    SS = (S1 @ S0)                                            # T -> C^(2) conversion for the RHS
    fk = np.fft.rfft(fgrid, axis=1)                           # Fourier in x
    fT = Vinv @ fk                                            # Chebyshev(T) in z, per mode
    ks = 2 * np.pi * np.arange(fk.shape[1]) / Lx              # physical wavenumbers
    uT = np.zeros_like(fT)
    for mi, k in enumerate(ks):
        M = helmholtz_M(k * k, D2, S0, S1, Nz)               # banded + dense BC rows
        rhs = SS @ fT[:, mi]                                  # forcing in C^(2) basis
        b = np.zeros(Nz, dtype=complex); b[2:] = rhs[:Nz - 2]  # b[0],b[1] = wall values = 0
        qr = factorize(M, method=method, dense_rows=[0, 1])  # tau-free QR (factor once)
        uT[:, mi] = qr.solve(np.ascontiguousarray(b.real)) + 1j * qr.solve(np.ascontiguousarray(b.imag))
    return np.fft.irfft(V @ uT, n=Nx, axis=1)                # back to the (z, x) grid

# ------------------------------------------------------------------ verification
def main(Nx=32, Nz=48, method="collapse"):
    Lx = 2 * np.pi
    z, V, Vinv = cheb(Nz)
    x = np.arange(Nx) * Lx / Nx
    X, Z = np.meshgrid(x, z)                                  # (Nz, Nx)
    # manufactured exact solution (u = 0 at z = +-1) and its analytic Laplacian
    u_exact = (1 - Z**2) * (np.exp(Z) * np.cos(X) + (1 + Z) * np.sin(2 * X))
    Fa = np.exp(z) * (-2.0 - 4.0 * z)                        # (P'' - 1*P), P=(1-z^2)e^z
    Fb = -6.0 - 10.0 * z + 4.0 * z**2 + 4.0 * z**3           # (Q'' - 4*Q), Q=(1-z^2)(1+z)
    f = np.cos(x)[None, :] * Fa[:, None] + np.sin(2 * x)[None, :] * Fb[:, None]

    u = poisson2d(f, Lx, z, V, Vinv, method=method)
    err = np.max(np.abs(u - u_exact)) / np.max(np.abs(u_exact))
    print(f"2D Poisson, tau-free, bandqr '{method}'  (Nx={Nx}, Nz={Nz})")
    print(f"  max relative error vs exact solution: {err:.2e}")
    return err

if __name__ == "__main__":
    for method in ("collapse", "ride"):
        main(method=method)
