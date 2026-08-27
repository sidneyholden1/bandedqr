"""Tau-free (boundary-bordering) in Dedalus's spherical BALL basis.

The Chebyshev case is easy: `Lift(tau, -1)` is essentially a unit column, so dropping
the top output mode reproduces the tau solve bit-for-bit.  The ball is genuinely harder
and `taufree_ball` handles the two ball-specific complications:

  1. The radial resolution depends on the spherical-harmonic degree l
     (n_size(l) = Nr - floor(l/2)), and for vector/tensor fields also on the component's
     spin weight -- the unused radial slots are structural PADDING that must be dropped
     from both equation rows and field columns, or the pencil is singular.

  2. A differential operator's OUTPUT basis can carry fewer top radial modes than the
     field (the ball vector Laplacian self-borders by one; the scalar Laplacian does
     not).  Bordering must measure this and drop  output_count - field_count + order
     top modes so that (rows kept) + (BC rows) = (field columns).

Both are detected structurally (probe random data, read the coefficient support), so no
spin/regularity bookkeeping is hard-coded.  Because the ball's tau column is spread over
several modes (not a unit column), tau-free bordering and the tau method are DIFFERENT
(both valid) discretisations at finite Nr -- they need not match mode-for-mode, but both
converge to the exact solution at the spectral rate.  We verify that here against
manufactured solutions (scalar and vector Poisson), with the standard tau solve alongside.
"""
import numpy as np
import dedalus.public as d3
from dedalus_bandqr import taufree_ball


def scalar_poisson(Nr, taufree):
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=np.float64)
    ball = d3.BallBasis(coords, shape=(16, 8, Nr), radius=1, dtype=np.float64)
    sphere = ball.surface
    phi, theta, r = dist.local_grids(ball)
    x, y, z = (r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta))
    lap = lambda A: d3.Laplacian(A, coords)
    uex = dist.Field(bases=ball); uex['g'] = (1 - r**2) * (1 + x + x * y + z * z * y)   # multi-l, =0 at r=1
    f = dist.Field(bases=ball); f['g'] = lap(uex).evaluate()['g']
    u = dist.Field(bases=ball)
    if taufree:
        prob = d3.LBVP([u], namespace=dict(u=u, f=f, lap=lap))
        prob.add_equation("lap(u) = f"); prob.add_equation("u(r=1) = 0")
        taufree_ball(prob, ball, dist, coords)
    else:
        tau = dist.Field(bases=sphere); lift = lambda A: d3.Lift(A, ball, -1)
        prob = d3.LBVP([u, tau], namespace=dict(u=u, tau=tau, f=f, lap=lap, lift=lift))
        prob.add_equation("lap(u) + lift(tau) = f"); prob.add_equation("u(r=1) = 0")
    prob.build_solver().solve()
    return np.max(np.abs(u['g'] - uex['g'])) / np.max(np.abs(uex['g']))


def vector_poisson(Nr, taufree):
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=np.float64)
    ball = d3.BallBasis(coords, shape=(16, 8, Nr), radius=1, dtype=np.float64)
    sphere = ball.surface
    phi, theta, r = dist.local_grids(ball)
    lap = lambda A: d3.Laplacian(A, coords)
    s = (1 - r**2)                                            # smooth (1-r^2) * e_z, =0 at r=1
    uex = dist.VectorField(coords, bases=ball)
    uex['g'][2] = s * np.cos(theta); uex['g'][1] = -s * np.sin(theta)
    f = dist.VectorField(coords, bases=ball); f['g'] = lap(uex).evaluate()['g']
    u = dist.VectorField(coords, bases=ball)
    if taufree:
        prob = d3.LBVP([u], namespace=dict(u=u, f=f, lap=lap))
        prob.add_equation("lap(u) = f"); prob.add_equation("u(r=1) = 0")
        taufree_ball(prob, ball, dist, coords)
    else:
        tau = dist.VectorField(coords, bases=sphere); lift = lambda A: d3.Lift(A, ball, -1)
        prob = d3.LBVP([u, tau], namespace=dict(u=u, tau=tau, f=f, lap=lap, lift=lift))
        prob.add_equation("lap(u) + lift(tau) = f"); prob.add_equation("u(r=1) = 0")
    prob.build_solver().solve()
    return np.max(np.abs(u['g'] - uex['g'])) / np.max(np.abs(uex['g']))


print("Tau-free boundary bordering in the Dedalus ball basis (vs standard tau + SuperLU):")
print(f"\n  scalar Poisson  lap(u)=f, u(r=1)=0   (manufactured, multi-l)")
print(f"  {'Nr':>4} {'tau (Lift)':>14} {'tau-free':>14}")
for Nr in (8, 16, 24, 32):
    print(f"  {Nr:>4} {scalar_poisson(Nr, False):>14.2e} {scalar_poisson(Nr, True):>14.2e}")

print(f"\n  vector Poisson  lap(u)=f, u(r=1)=0   (manufactured, smooth (1-r^2) e_z)")
print(f"  {'Nr':>4} {'tau (Lift)':>14} {'tau-free':>14}")
for Nr in (8, 16, 24, 32):
    print(f"  {Nr:>4} {vector_poisson(Nr, False):>14.2e} {vector_poisson(Nr, True):>14.2e}")
