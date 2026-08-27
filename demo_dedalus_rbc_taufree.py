"""2D Rayleigh-Benard convection in Dedalus, TAU-FREE + bandqr QR.

Vorticity-streamfunction Boussinesq convection -- the shear flow plus a temperature
field.  Streamfunction bakes in incompressibility (no pressure, no gauge); the
buoyancy enters the vorticity equation as the baroclinic torque Ra*dx(b):

    dt(w) - nu*lap(w) - Ra*dx(b) = -u.grad(w)
    lap(psi) + w = 0                                 u = (psi_y, -psi_x)
    dt(b) - kappa*lap(b)         = -u.grad(b)

on [0, Lx) x [-1, 1], periodic in x, no-slip walls (psi = psi_y = 0), fixed wall
temperatures.  Three coupled wall-bounded fields, six BCs, no pressure -- exactly
the coupled tau-free case.  No tau fields, no `Lift`: `boundary_border` drops the
top two modes of each of the three volume equations, and the bandqr matsolver
factorises the block-banded + dense-BC system with QR.  Advection is explicit
(IMEX RK222).  Compared step for step against the standard tau + SuperLU run.
"""
import numpy as np
import dedalus.public as d3
from dedalus_bandqr import BandQR, boundary_border

Nx, Ny, Lx = 48, 48, 4.0
nu, kappa, Ra = 1e-2, 1e-2, 2e4
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)
xb = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
yb = d3.Chebyshev(coords['y'], size=Ny, bounds=(-1, 1))
x = dist.local_grid(xb); y = dist.local_grid(yb)
lap = lambda A: d3.Laplacian(A, coords)
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

def run(taufree, nsteps=15, dt=1e-3):
    psi = dist.Field(name='psi', bases=(xb, yb))
    w = dist.Field(name='w', bases=(xb, yb))
    b = dist.Field(name='b', bases=(xb, yb))
    u, v = dy(psi), -dx(psi)
    if taufree:
        ns = dict(psi=psi, w=w, b=b, u=u, v=v, nu=nu, kappa=kappa, Ra=Ra, lap=lap, dx=dx, dy=dy)
        prob = d3.IVP([w, psi, b], namespace=ns)                 # NO tau fields
        prob.add_equation("dt(w) - nu*lap(w) - Ra*dx(b) = -(u*dx(w) + v*dy(w))")
        prob.add_equation("lap(psi) + w = 0")
        prob.add_equation("dt(b) - kappa*lap(b) = -(u*dx(b) + v*dy(b))")
    else:
        lift = lambda A, n: d3.Lift(A, yb.derivative_basis(2), n)
        t = [dist.Field(bases=xb) for _ in range(6)]
        ns = dict(psi=psi, w=w, b=b, u=u, v=v, nu=nu, kappa=kappa, Ra=Ra, lap=lap, dx=dx, dy=dy,
                  lift=lift, t1=t[0], t2=t[1], t3=t[2], t4=t[3], t5=t[4], t6=t[5])
        prob = d3.IVP([w, psi, b] + t, namespace=ns)
        prob.add_equation("dt(w) - nu*lap(w) - Ra*dx(b) + lift(t1,-1) + lift(t2,-2) = -(u*dx(w) + v*dy(w))")
        prob.add_equation("lap(psi) + w + lift(t3,-1) + lift(t4,-2) = 0")
        prob.add_equation("dt(b) - kappa*lap(b) + lift(t5,-1) + lift(t6,-2) = -(u*dx(b) + v*dy(b))")
    prob.add_equation("psi(y=-1)=0"); prob.add_equation("psi(y=+1)=0")
    prob.add_equation("dy(psi)(y=-1)=0"); prob.add_equation("dy(psi)(y=+1)=0")
    prob.add_equation("b(y=-1)=0.5"); prob.add_equation("b(y=+1)=-0.5")
    if taufree:
        boundary_border(prob, orders=[2, 2, 2])                  # drop top 2 of each volume eqn
    b['g'] = -0.5 * y + 0.05 * np.sin(2 * np.pi * x / Lx) * np.cos(np.pi * y / 2)  # conductive + seed
    solver = prob.build_solver(d3.RK222,
                               matsolver=(BandQR if taufree else 'SuperluNaturalSpsolve'))
    for _ in range(nsteps):
        solver.step(dt)
    return b['g'].copy(), len(prob.variables)

b_tf, n_tf = run(taufree=True)
b_ref, n_ref = run(taufree=False)
err = np.max(np.abs(b_tf - b_ref)) / np.max(np.abs(b_ref))
print("2D Rayleigh-Benard (vorticity-streamfunction), 15 IMEX steps")
print(f"  tau-free (bandqr) : {n_tf} fields  (psi, w, b -- no pressure, no tau)")
print(f"  standard (tau+LU) : {n_ref} fields  (psi, w, b + 6 tau)")
print(f"  max rel diff between the two runs: {err:.2e}")
