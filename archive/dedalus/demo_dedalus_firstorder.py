"""Auto first-order reduction + tau-free in Dedalus (the general mechanism).

A high-order ODE is posed by giving its operator coefficients; `first_order_reduce`
introduces the derivative-chain auxiliaries automatically, so every equation is
first-order and tau-free drops exactly ONE mode per equation (uniform -- no
per-equation order bookkeeping).  This is the key to a *general* tau-free: reduce to
first order, then border uniformly.  It is also far better conditioned than bordering
the single high-order equation.

Shown here: manufactured solutions at orders 2/4/6 (machine precision, default LU),
and the conditioning of first-order vs naive high-order bordering as N grows.
"""
import numpy as np
import dedalus.public as d3
from dedalus_bandqr import taufree, first_order_reduce

xc = d3.Coordinate('z')

def naive_highorder(dist, zb, m, rhs, bcs):
    """Border the single m-th-order equation directly (drop m from one equation)."""
    u = dist.Field(name='u', bases=zb); dz = lambda A: d3.Differentiate(A, xc)
    prob = d3.LBVP([u], namespace=dict(u=u, rhs=rhs, dz=dz))
    op = "u"
    for _ in range(m): op = f"dz({op})"
    prob.add_equation(f"{op} = rhs")
    for loc, d, val in bcs: prob.add_equation(f"{'dz('*d}u{')'*d}(z={loc}) = {val}")
    taufree(prob, orders=[m])
    return prob

# manufactured: u = (1-z^2)^(m/2)  ->  d^m u/dz^m = const, clamped-style BCs
SPECS = {
    2: ([0, 0, 1],             lambda z: 1 - z**2,      -2.0,   [(-1,0,0),(1,0,0)]),
    4: ([0, 0, 0, 0, 1],       lambda z: (1 - z**2)**2,  24.0,  [(-1,0,0),(1,0,0),(-1,1,0),(1,1,0)]),
    6: ([0]*6 + [1],           lambda z: (1 - z**2)**3, -720.0, [(-1,0,0),(1,0,0),(-1,1,0),(1,1,0),(-1,2,0),(1,2,0)]),
}

print("auto first-order reduction + tau-free (default LU), manufactured solutions:")
N = 48
for m, (coeffs, uex, fval, bcs) in SPECS.items():
    dist = d3.Distributor(xc, dtype=np.float64)
    zb = d3.Chebyshev(xc, size=N, bounds=(-1, 1)); z = dist.local_grid(zb)
    rhs = dist.Field(bases=zb); rhs['g'] = fval
    prob, U = first_order_reduce(dist, zb, xc, coeffs, rhs, bcs)
    prob.build_solver().solve()
    ue = uex(z); err = np.max(np.abs(U[0]['g'] - ue)) / np.max(np.abs(ue))
    print(f"  order {m}:  {m} fields, uniform drop-1 per equation   rel err = {err:.1e}")

print("\nconditioning of L_min, 4th-order clamped (first-order reduced vs naive):")
print(f"  {'N':>4} {'first-order (drop 1)':>22} {'naive (drop 4)':>16} {'improvement':>12}")
coeffs, uex, fval, bcs = SPECS[4]
for N in (16, 32, 64, 128):
    dist = d3.Distributor(xc, dtype=np.float64)
    zb = d3.Chebyshev(xc, size=N, bounds=(-1, 1))
    rhs = dist.Field(bases=zb); rhs['g'] = fval
    pf, _ = first_order_reduce(dist, zb, xc, coeffs, rhs, bcs)
    sf = pf.build_solver(store_expanded_matrices=True); sf.solve()
    pn = naive_highorder(dist, zb, 4, rhs, bcs)
    sn = pn.build_solver(store_expanded_matrices=True); sn.solve()
    cf = np.linalg.cond(sf.subproblems[0].L_min.toarray())
    cn = np.linalg.cond(sn.subproblems[0].L_min.toarray())
    print(f"  {N:>4} {cf:>22.1e} {cn:>16.1e} {cn/cf:>11.0f}x")
