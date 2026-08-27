"""A Dedalus matsolver backed by bandqr (almost-banded QR).

Usage in a Dedalus script:

    from dedalus_bandqr import BandQR                 # registers as 'bandqr'
    solver = problem.build_solver(matsolver=BandQR)   # or matsolver='bandqr'

It detects the almost-banded structure of each subproblem's L_min -- a banded
bulk plus dense boundary-condition rows (and, if present, off-band tau columns)
-- permutes the BC rows to the top, and factorises with bandqr's QR (ride, or
ride+Woodbury when off-band tau columns appear).  If a subproblem is not
almost-banded (e.g. strong coupling), it falls back to SuperLU so a run never
breaks.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dedalus.libraries.matsolvers import SolverBase, add_solver
from bandqr import detect_structure, RideQR, RideTauQR, BandQR as _BandQR, _to_band


def taufree(problem, orders=None, axis=-1):
    """Reformulate a wall-bounded problem WITHOUT tau variables / `Lift` terms
    (boundary bordering) -- the `tau_BC=False` path.

    Build the problem with only the physical field(s) (no tau fields, no `Lift`, no
    `derivative_basis`), add the PDE(s) and BC equations, then call this before
    `build_solver()`.  It drops the top `order` output modes of each volume equation
    along `axis` (the bounded/Chebyshev direction), so the system squares by dropping
    equation rows instead of adding tau columns.  Solve with ANY matsolver -- the
    default sparse-LU is fine; bandqr is optional (and only pays off for its own
    reasons: GPU batching, conditioning, determinism).

    orders : top modes to drop per volume equation (its differential order in the
        bounded direction), as a list over the volume equations.  If None, the number
        of BC equations is inferred and split evenly -- correct when the volume
        equations share a differential order (the common case: all 2nd-order).  For a
        mixed-order system pass the list explicitly.

    Scope: scalar and coupled wall-bounded systems with no internal nullspace.  A
    pressure gauge / incompressibility constraint is NOT a boundary tau -- keep it as
    a scalar multiplier (a normal field + an `integ(...)=0` equation); this routine
    leaves such constraint scalars alone.  Sets Dedalus's internal `valid_modes`; it
    is a prototype, not (yet) a public Dedalus API.
    """
    eqns = problem.equations
    sizes = [e['valid_modes'].size for e in eqns]
    mx = max(sizes)
    vol = [i for i, s in enumerate(sizes) if s == mx]            # volume eqns (full size)
    n_bc = len(eqns) - len(vol)                                  # the rest are BC / constraint
    if orders is None:
        if len(vol) == 0 or n_bc % len(vol):
            raise ValueError(
                f"taufree: {n_bc} boundary/constraint equations do not split evenly "
                f"across {len(vol)} volume equations -- pass orders=[...] giving each "
                f"volume equation's differential order in the bounded direction.")
        orders = [n_bc // len(vol)] * len(vol)
    if len(orders) != len(vol):
        raise ValueError(f"taufree: got {len(orders)} orders for {len(vol)} volume equations.")
    for i, order in zip(vol, orders):
        if order:
            vm = problem.equations[i]['valid_modes']
            sl = [slice(None)] * vm.ndim
            sl[axis] = slice(-order, None)
            vm[tuple(sl)] = False                                # drop the top modes
    return problem

boundary_border = taufree                                       # back-compat alias


def _radial_slice(vm, phi_sl, theta_sl, lo, hi):
    """Index the last three axes (phi_c, theta_c, radial) of a valid_modes array,
    leaving any leading tensor-component axes untouched."""
    idx = [slice(None)] * vm.ndim
    idx[-3] = phi_sl
    idx[-2] = theta_sl
    idx[-1] = slice(lo, hi)
    return tuple(idx)


def _ball_radial_mask(tensor_rank, ball, dist, coords, ndraws=4):
    """Boolean mask (shape = a rank-`tensor_rank` field's coeff array) that is True on the
    structurally REAL radial modes of a *field* and False on the padding.

    In the ball the radial resolution depends on the spherical-harmonic degree l AND, for
    vector/tensor fields, on the component's spin weight -- some components lose modes at
    the low (regularity at r=0) end, some at the top.  Rather than hard-code the spin
    rules, detect the structure directly: transform random grid data and see which
    coefficients are structurally zero (a padding slot is zero for EVERY draw).
    """
    if tensor_rank == 0:
        fld = dist.Field(bases=ball)
    else:
        fld = dist.VectorField(coords, bases=ball) if tensor_rank == 1 \
            else dist.TensorField((coords,) * tensor_rank, bases=ball)
    mask = None
    for s in range(ndraws):
        fld['g'] = np.random.default_rng(s).standard_normal(fld['g'].shape)
        m = np.abs(fld['c']) > 1e-11
        mask = m if mask is None else (mask | m)
    return mask


def _ball_output_mask(lhs, variables, ndraws=4):
    """Structurally-real radial modes of an equation's LHS OUTPUT (its row space).

    A differential operator's output basis can carry fewer top radial modes than the input
    field (an order-p radial operator "spends" modes) -- e.g. the ball vector Laplacian
    drops one top mode per component while the scalar Laplacian keeps all.  This natural
    reduction is exactly the room the boundary condition fills, so it must be measured (not
    assumed) to border correctly.  Probe by evaluating the LHS on random field data;
    saves/restores the variables so it is side-effect free.
    """
    saved = [v['c'].copy() for v in variables]
    mask = None
    for s in range(ndraws):
        for v in variables:
            v['g'] = np.random.default_rng(1000 + s).standard_normal(v['g'].shape)
        out = lhs.evaluate()
        m = np.abs(out['c']) > 1e-11
        mask = m if mask is None else (mask | m)
    for v, c in zip(variables, saved):
        v['c'] = c
    return mask


def taufree_ball(problem, ball, dist, coords, orders=None):
    """Boundary bordering for Dedalus's spherical **ball** basis (tau-free), the
    spherical analogue of `taufree`.

    The Chebyshev case is easy because `Lift(tau, -1)` is (essentially) a unit column, so
    dropping the top output mode reproduces the tau solve bit-for-bit.  The ball is harder
    in two ways, both handled here:

    1.  The radial resolution depends on the harmonic degree l (`n_size(l)=Nr-floor(l/2)`)
        and, for vector/tensor fields, on the component's spin weight.  The unused radial
        slots are structural PADDING; they are detected (see `_ball_radial_mask`) and
        dropped from both the equations (rows) and the field variables (columns), or the
        restricted pencil is singular.
    2.  Bordering then drops the top `order` *structurally-valid* radial modes of each
        volume equation, per angular mode and per component.

    Because the ball's tau column is spread over several modes (not a unit column),
    tau-free bordering and the tau method are DIFFERENT (both valid) discretisations at
    finite Nr -- they do not match mode-for-mode, but both converge to the exact solution
    at the spectral rate (verified to machine precision on manufactured solutions).

    coords : the SphericalCoordinates (needed to build probe fields for structure detection).
    orders : per-volume-equation count of radial boundary conditions (top valid modes to
        drop); a list over the volume equations (those spanning the full radial axis).  If
        None, inferred as (#BC eqns)//(#volume eqns), split evenly.  Pass explicitly for
        mixed order -- e.g. an incompressible momentum/continuity/energy set is [1, 0, 1]
        (continuity carries only the pressure-gauge scalar, no radial BC).  Mutates
        Dedalus's cached `valid_modes` (internals, not public API).
    """
    Nr = ball.shape[-1]
    fmcache = {}
    def field_mask(rank):
        if rank not in fmcache:
            fmcache[rank] = _ball_radial_mask(rank, ball, dist, coords)
        return fmcache[rank]

    def keep_only(vm, pre, idx):
        keep = np.zeros(vm.shape[-1], dtype=bool)
        keep[idx] = True
        vm[pre] &= keep

    # 1. Field variables: drop structural padding COLUMNS (down to the field's real modes).
    for var in problem.LHS_variables:
        vm = var.valid_modes                                    # cached -> mutate in place
        if vm.shape[-1] != Nr:
            continue
        fm = field_mask(vm.ndim - 3)
        for pre in np.ndindex(vm.shape[:-1]):
            vidx = np.nonzero(vm[pre])[0]
            if not len(vidx):
                continue
            fidx = np.nonzero(fm[pre])[0]
            keep_only(vm, pre, fidx if len(fidx) else vidx)     # zero-slots keep field validity

    # 2. Classify equations; volume = spans the full radial axis.
    eqns = problem.equations
    vol = [i for i, e in enumerate(eqns) if e['valid_modes'].shape[-1] == Nr]
    n_bc = len(eqns) - len(vol)
    if orders is None:
        if len(vol) == 0 or n_bc % len(vol):
            raise ValueError(
                f"taufree_ball: {n_bc} boundary/constraint equations do not split evenly "
                f"across {len(vol)} volume equations -- pass orders=[...].")
        orders = [n_bc // len(vol)] * len(vol)
    if len(orders) != len(vol):
        raise ValueError(f"taufree_ball: got {len(orders)} orders for {len(vol)} volume equations.")

    # 3. Volume equations: restrict ROWS to the LHS output structure, then border so that
    #    (rows kept) + (order BC rows) = (field columns).  The number of top modes to drop
    #    is output_count - field_count + order  --  1 when the operator keeps the top mode
    #    (scalar Laplacian), 0 when the operator already self-borders (vector Laplacian).
    for i, order in zip(vol, orders):
        e = eqns[i]
        vm = e['valid_modes']
        rank = vm.ndim - 3
        fm = field_mask(rank)
        om = _ball_output_mask(e['LHS'], problem.LHS_variables)
        for pre in np.ndindex(vm.shape[:-1]):
            vidx = np.nonzero(vm[pre])[0]
            if not len(vidx):
                continue
            fcount = int(fm[pre].sum()) or len(vidx)            # field columns at this mode
            oidx = np.nonzero(om[pre])[0]
            ovalid = oidx if len(oidx) else vidx                # output rows (zero-slot fallback)
            keep_only(vm, pre, ovalid)
            ndrop = len(ovalid) - fcount + order                # border to field_count - order
            if ndrop > 0:
                vm[pre][ovalid[-ndrop:]] = False
    return problem


def first_order_reduce(dist, basis, coord, coeffs, rhs, bcs):
    """Auto first-order reduction + tau-free, in the bounded (`coord`) direction.

    Pose a constant-coefficient linear ODE  sum_k coeffs[k] * d^k u / dz^k = rhs
    (order m = len(coeffs) - 1) as a first-order chain by introducing the successive
    derivatives u1..u_{m-1} as auxiliary fields.  Every equation is then first-order,
    so `taufree` drops EXACTLY ONE mode per equation -- uniform, with no per-equation
    order bookkeeping -- and the system is far better conditioned than bordering the
    single high-order equation (naive ~N^order vs reduced ~N^1.5).

    coeffs : [c0, c1, ..., cm]  (ck multiplies the k-th derivative).
    rhs    : a Field (the right-hand side).
    bcs    : list of (location, derivative_index, value) -- a BC on the d-th
             derivative is just a condition on the auxiliary u_d.

    Returns (problem, U) with U[0] the solution field; call build_solver() with any
    matsolver (the default LU is fine).  Scalar constant-coefficient case (the
    prototype of the general mechanism); coupled / variable-coefficient / PDE
    reduction is the same idea applied per field.
    """
    import dedalus.public as d3
    m = len(coeffs) - 1
    dz = lambda A: d3.Differentiate(A, coord)
    U = [dist.Field(name=f'u{k}', bases=basis) for k in range(m)]
    ns = {f'u{k}': U[k] for k in range(m)}
    ns.update(rhs=rhs, dz=dz)
    prob = d3.LBVP(U, namespace=ns)
    for k in range(m - 1):
        prob.add_equation(f"u{k+1} - dz(u{k}) = 0")                # u_{k+1} = u_k'
    top = f"({coeffs[m]})*dz(u{m-1})" + "".join(
        f" + ({coeffs[k]})*u{k}" for k in range(m) if coeffs[k])
    prob.add_equation(top + " = rhs")                             # highest term + lower terms
    for loc, d, val in bcs:
        prob.add_equation(f"u{d}(z={loc}) = {val}")               # BC on the d-th derivative
    taufree(prob)                                                 # uniform drop-1 per equation
    return prob, U


@add_solver
class BandQR(SolverBase):
    """bandqr almost-banded QR solver for Dedalus subproblems."""

    sparse = True

    def __init__(self, matrix, solver=None):
        M = sp.csr_matrix(matrix)
        n = M.shape[0]
        self.perm = None
        self.mode = "bandqr"
        try:
            kl, ku, dr, dc = detect_structure(M)
            if len(dr) == 0 and len(dc) == 0:                     # purely banded
                self.qr = _BandQR(_to_band(M, kl, ku), kl, ku)
            else:                                                 # dense BC rows -> top
                self.perm = np.concatenate([dr, np.setdiff1d(np.arange(n), dr)]).astype(int)
                Mp = M[self.perm, :]
                akl, aku, adr, adc = detect_structure(Mp)
                # bail out if it isn't really banded (avoid a huge/degenerate band)
                if max(akl, aku) > n // 4:
                    raise ValueError("not almost-banded")
                self.qr = (RideTauQR(Mp, akl, aku, adr, adc) if len(adc) > 0
                           else RideQR(Mp, akl, aku, adr))
            self._z = bool(getattr(self.qr, "_z", False))
        except Exception:                                         # robust fallback
            self.mode = "splu"
            self.perm = None
            self.lu = spla.splu(sp.csc_matrix(matrix))

    def solve(self, vector):
        b = np.asarray(vector)
        if self.mode == "splu":
            out = self.lu.solve(b)
            return out[:, None] if (b.ndim == 2 and out.ndim == 1) else out
        bp = b[self.perm] if self.perm is not None else b
        if np.iscomplexobj(bp) and not self._z:                   # real factor, complex RHS
            x = (self.qr.solve(np.ascontiguousarray(bp.real))
                 + 1j * self.qr.solve(np.ascontiguousarray(bp.imag)))
        else:
            x = self.qr.solve(np.ascontiguousarray(bp))
        return x[:, None] if (vector.ndim == 2 and x.ndim == 1) else x
