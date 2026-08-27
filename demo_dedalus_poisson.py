"""2D Poisson in Dedalus, solved with the bandqr matsolver.

    laplacian(u) = f  on  [0, 2pi) x [-1, 1],  periodic in x,  u = 0 at z = +-1.

Dedalus assembles the (tau) problem as usual; the only change is
`matsolver=BandQR`, which factorises each Fourier mode's almost-banded L_min with
bandqr's QR.  Verified against a manufactured exact solution and against Dedalus's
default sparse-LU.
"""
import numpy as np
import dedalus.public as d3
from dedalus_bandqr import BandQR                      # registers the 'bandqr' matsolver

Nx, Nz, Lx = 64, 48, 2 * np.pi
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=np.float64)
xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
zb = d3.Chebyshev(coords['z'], size=Nz, bounds=(-1, 1))
x = dist.local_grid(xb)
z = dist.local_grid(zb)

u = dist.Field(name='u', bases=(xb, zb))
tau1 = dist.Field(name='tau1', bases=xb)
tau2 = dist.Field(name='tau2', bases=xb)
f = dist.Field(name='f', bases=(xb, zb))

# manufactured solution (u = 0 at z = +-1) and its analytic Laplacian
u_exact = (1 - z**2) * (np.exp(z) * np.cos(x) + (1 + z) * np.sin(2 * x))
Fa = np.exp(z) * (-2.0 - 4.0 * z)                      # (P'' - 1*P),  P = (1-z^2) e^z
Fb = -6.0 - 10.0 * z + 4.0 * z**2 + 4.0 * z**3         # (Q'' - 4*Q),  Q = (1-z^2)(1+z)
f['g'] = np.cos(x) * Fa + np.sin(2 * x) * Fb

lap = lambda A: d3.Laplacian(A, coords)
lift = lambda A, n: d3.Lift(A, zb.derivative_basis(2), n)
ns = dict(u=u, tau1=tau1, tau2=tau2, f=f, lap=lap, lift=lift)

def solve_with(matsolver):
    problem = d3.LBVP([u, tau1, tau2], namespace=ns)
    problem.add_equation("lap(u) + lift(tau1, -1) + lift(tau2, -2) = f")
    problem.add_equation("u(z=-1) = 0")
    problem.add_equation("u(z=+1) = 0")
    solver = problem.build_solver(matsolver=matsolver)
    solver.solve()
    return u['g'].copy()

u_bandqr = solve_with(BandQR)
err_exact = np.max(np.abs(u_bandqr - u_exact)) / np.max(np.abs(u_exact))
u_default = solve_with('SuperluNaturalSpsolve')
err_default = np.max(np.abs(u_bandqr - u_default)) / np.max(np.abs(u_default))

print(f"2D Poisson in Dedalus (Nx={Nx}, Nz={Nz}), matsolver=bandqr")
print(f"  max rel error vs exact solution : {err_exact:.2e}")
print(f"  max rel diff vs Dedalus SuperLU : {err_default:.2e}")
