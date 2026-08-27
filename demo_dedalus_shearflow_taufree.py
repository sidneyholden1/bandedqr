"""2D incompressible shear flow in Dedalus, TAU-FREE + bandqr QR.

Vorticity-streamfunction formulation of 2D incompressible flow:

    dt(w) + u.grad(w) = nu*lap(w) ,   lap(psi) = -w ,   u = (psi_y, -psi_x)

on [0, Lx) x [-1, 1], periodic in x, no-slip walls (u = v = 0 at y = +-1, i.e.
psi = 0 and psi_y = 0).  The streamfunction bakes in incompressibility, so there is
NO pressure and NO gauge -- and the implicit step is a coupled (psi, w) wall-bounded
system with four BCs, exactly the case boundary bordering + bandqr handles.

No tau fields, no `Lift`: `boundary_border` drops the top two modes of each of the
two volume equations, and the bandqr matsolver factorises the block-banded +
dense-BC system with QR.  The nonlinear advection is explicit (IMEX RK222).  A run
is compared, step for step, against the standard tau + SuperLU formulation.
"""
import numpy as np
import dedalus.public as d3
from dedalus_bandqr import BandQR, boundary_border

Nx, Ny, Lx, nu = 48, 48, 2 * np.pi, 0.02
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)
xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
yb = d3.Chebyshev(coords['y'], size=Ny, bounds=(-1, 1))
x = dist.local_grid(xb); y = dist.local_grid(yb)
lap = lambda A: d3.Laplacian(A, coords)
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

def run(taufree, nsteps=20, dt=0.002):
    psi = dist.Field(name='psi', bases=(xb, yb))
    w = dist.Field(name='w', bases=(xb, yb))
    u, v = dy(psi), -dx(psi)                                  # velocity from streamfunction
    if taufree:
        ns = dict(psi=psi, w=w, u=u, v=v, nu=nu, lap=lap, dx=dx, dy=dy)
        prob = d3.IVP([psi, w], namespace=ns)                # NO tau fields
        prob.add_equation("dt(w) - nu*lap(w) = -(u*dx(w) + v*dy(w))")
        prob.add_equation("lap(psi) + w = 0")
    else:
        lift = lambda A, n: d3.Lift(A, yb.derivative_basis(2), n)
        t = [dist.Field(bases=xb) for _ in range(4)]
        ns = dict(psi=psi, w=w, u=u, v=v, nu=nu, lap=lap, dx=dx, dy=dy, lift=lift,
                  t1=t[0], t2=t[1], t3=t[2], t4=t[3])
        prob = d3.IVP([psi, w] + t, namespace=ns)
        prob.add_equation("dt(w) - nu*lap(w) + lift(t1,-1) + lift(t2,-2) = -(u*dx(w) + v*dy(w))")
        prob.add_equation("lap(psi) + w + lift(t3,-1) + lift(t4,-2) = 0")
    prob.add_equation("psi(y=-1) = 0"); prob.add_equation("psi(y=+1) = 0")
    prob.add_equation("dy(psi)(y=-1) = 0"); prob.add_equation("dy(psi)(y=+1) = 0")
    if taufree:
        boundary_border(prob, orders=[2, 2])                 # drop top 2 of each volume eqn
    w['g'] = np.exp(-8 * y**2) * (1 + 0.1 * np.cos(x))        # shear layer + perturbation
    solver = prob.build_solver(d3.RK222,
                               matsolver=(BandQR if taufree else 'SuperluNaturalSpsolve'))
    for _ in range(nsteps):
        solver.step(dt)
    return w['g'].copy(), prob

w_tf, prob_tf = run(taufree=True)
w_ref, prob_ref = run(taufree=False)
err = np.max(np.abs(w_tf - w_ref)) / np.max(np.abs(w_ref))
print("2D incompressible shear flow (vorticity-streamfunction), 20 IMEX steps")
print(f"  tau-free (bandqr)  : {len(prob_tf.variables)} fields  (psi, w -- no pressure, no tau)")
print(f"  standard (tau+LU)  : {len(prob_ref.variables)} fields  (psi, w + 4 tau)")
print(f"  max rel diff between the two runs: {err:.2e}")
