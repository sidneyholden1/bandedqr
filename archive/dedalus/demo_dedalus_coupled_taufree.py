"""Coupled wall-bounded system in Dedalus, TAU-FREE (boundary bordering) + bandqr.

Two coupled fields on a Chebyshev interval with Dirichlet walls:

    a'' - k a + b = fa ,   b'' - k b + a = fb ,   a = b = 0 at z = +-1.

No tau fields, no `Lift`: `boundary_border` drops the top 2 modes of *each* volume
equation, so the coupled system squares by dropping rows instead of adding tau
variables.  The bandqr matsolver factorises the resulting block-banded + dense-BC
system with QR.  Verified against a manufactured exact solution and the standard
tau + SuperLU run.  This is the coupled generalisation of the scalar tau-free demo;
it is the mechanism a wall-bounded shear flow needs for its velocity BCs (the
pressure gauge is a separate, still-open piece).
"""
import numpy as np
import dedalus.public as d3
from dedalus_bandqr import BandQR, boundary_border

N, k = 48, 10.0
zc = d3.Coordinate('z')
dist = d3.Distributor(zc, dtype=np.float64)
zb = d3.Chebyshev(zc, size=N, bounds=(-1, 1))
z = dist.local_grid(zb)
dz = lambda A: d3.Differentiate(A, zc)

# manufactured solution (a, b = 0 at walls) and its forcing
a_ex = (1 - z**2)
b_ex = (1 - z**2) * z

def solve_taufree():
    a = dist.Field(name='a', bases=zb); b = dist.Field(name='b', bases=zb)
    fa = dist.Field(name='fa', bases=zb); fb = dist.Field(name='fb', bases=zb)
    fa['g'] = -2 - k * a_ex + b_ex                  # a'' - k a + b
    fb['g'] = -6 * z - k * b_ex + a_ex              # b'' - k b + a
    prob = d3.LBVP([a, b], namespace=dict(a=a, b=b, fa=fa, fb=fb, dz=dz, k=k))  # NO tau
    prob.add_equation("dz(dz(a)) - k*a + b = fa")
    prob.add_equation("dz(dz(b)) - k*b + a = fb")
    prob.add_equation("a(z=-1)=0"); prob.add_equation("a(z=+1)=0")
    prob.add_equation("b(z=-1)=0"); prob.add_equation("b(z=+1)=0")
    boundary_border(prob)                            # drop top 2 of each volume eqn
    solver = prob.build_solver(store_expanded_matrices=True, matsolver=BandQR)
    solver.solve()
    return a['g'].copy(), b['g'].copy(), solver.subproblems[0].L_min.shape

a, b, shape = solve_taufree()
ea = np.max(np.abs(a - a_ex)) / np.max(np.abs(a_ex))
eb = np.max(np.abs(b - b_ex)) / np.max(np.abs(b_ex))
print(f"coupled tau-free (N={N}), bandqr QR")
print(f"  L_min shape: {shape}   (2 fields x {N} modes -- no tau; standard would be {2*N+4})")
print(f"  rel error vs exact:  a = {ea:.2e},  b = {eb:.2e}")
