"""2D Poisson in Dedalus, TAU-FREE (boundary bordering) + bandqr QR.

    laplacian(u) = f  on  [0, 2pi) x [-1, 1],  periodic in x,  u = 0 at z = +-1.

No tau fields and no `Lift` terms are declared.  `boundary_border()` drops the top
Chebyshev modes of the volume equation, so Dedalus squares the system by dropping
rows (boundary bordering) instead of adding tau variables -- the assembled L_min is
banded + dense BC rows, in the volume field only.  The bandqr matsolver factorises
it with QR.  Compared against the exact solution and the standard tau + SuperLU run.
"""
import numpy as np
import dedalus.public as d3
from dedalus_bandqr import BandQR, boundary_border

Nx, Nz, Lx = 64, 48, 2 * np.pi
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=np.float64)
xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
zb = d3.Chebyshev(coords['z'], size=Nz, bounds=(-1, 1))
x = dist.local_grid(xb)
z = dist.local_grid(zb)
lap = lambda A: d3.Laplacian(A, coords)

# manufactured exact solution (u = 0 at z = +-1) and its analytic Laplacian
def forcing(F):
    Fa = np.exp(z) * (-2.0 - 4.0 * z)                 # (P'' - 1*P),  P = (1-z^2) e^z
    Fb = -6.0 - 10.0 * z + 4.0 * z**2 + 4.0 * z**3    # (Q'' - 4*Q),  Q = (1-z^2)(1+z)
    F['g'] = np.cos(x) * Fa + np.sin(2 * x) * Fb
u_exact = (1 - z**2) * (np.exp(z) * np.cos(x) + (1 + z) * np.sin(2 * x))

# ---- tau-free: no tau fields, no Lift, boundary-bordered, bandqr QR ----
u = dist.Field(name='u', bases=(xb, zb))
f = dist.Field(name='f', bases=(xb, zb)); forcing(f)
prob = d3.LBVP([u], namespace=dict(u=u, f=f, lap=lap))
prob.add_equation("lap(u) = f")
prob.add_equation("u(z=-1) = 0")
prob.add_equation("u(z=+1) = 0")
boundary_border(prob)                                   # <-- removes tau entirely
solver = prob.build_solver(store_expanded_matrices=True, matsolver=BandQR)
solver.solve()
u_taufree = u['g'].copy()

# ---- standard Dedalus for reference: tau variables + SuperLU ----
u2 = dist.Field(name='u2', bases=(xb, zb))
t1 = dist.Field(name='t1', bases=xb); t2 = dist.Field(name='t2', bases=xb)
f2 = dist.Field(name='f2', bases=(xb, zb)); forcing(f2)
lift = lambda A, n: d3.Lift(A, zb.derivative_basis(2), n)
prob2 = d3.LBVP([u2, t1, t2], namespace=dict(u2=u2, t1=t1, t2=t2, f2=f2, lap=lap, lift=lift))
prob2.add_equation("lap(u2) + lift(t1, -1) + lift(t2, -2) = f2")
prob2.add_equation("u2(z=-1) = 0")
prob2.add_equation("u2(z=+1) = 0")
solver2 = prob2.build_solver(store_expanded_matrices=True)
solver2.solve()

err_exact = np.max(np.abs(u_taufree - u_exact)) / np.max(np.abs(u_exact))
err_tau = np.max(np.abs(u_taufree - u2['g'])) / np.max(np.abs(u2['g']))
print(f"2D Poisson in Dedalus (Nx={Nx}, Nz={Nz})")
print(f"  tau-free L_min : {solver.subproblems[0].L_min.shape}   (u only -- no tau columns)")
print(f"  standard L_min : {solver2.subproblems[0].L_min.shape}   (u + 2 tau variables)")
print(f"  tau-free bandqr vs exact solution : {err_exact:.2e}")
print(f"  tau-free bandqr vs standard tau+LU: {err_tau:.2e}")
