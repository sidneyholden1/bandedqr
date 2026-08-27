"""Internally-heated convection in a BALL, tau-free (RectQR) vs standard tau — side by side.

The velocity-pressure tau-free path: pose the problem with NO boundary tau fields (only a
single tau_p gauge scalar), let Dedalus assemble the naturally RECTANGULAR system (bulk +
BC rows, tall by #BCs), and solve it with a rectangular least-squares QR (RectQR).  This
dissolves the ball's inf-sup pressure-space problem that the square boundary-bordering
could not (verified: ball Stokes u,p to machine precision).

Time stepping is a custom semi-implicit backward-Euler loop of LBVP solves (the LBVP path
is clean -- only the squareness check is patched; no timestepper/CoeffSystem corruption).
Diffusion, pressure and buoyancy are implicit; advection is lagged.  The tau-free scheme is
a different-but-convergent discretisation from standard tau (reldiff shrinks ~N^-1: 1.9 @
Nr16 -> 0.05 @ Nr48); both show the same convection.  Equatorial T slices are animated.

Writes ball_convection_taufree_vs_standard.mp4.
"""
import matplotlib; matplotlib.use('Agg')
import inspect, textwrap, time, numpy as np, scipy.sparse as sp
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import dedalus.core.subsystems as ss, dedalus.public as d3
from dedalus.libraries.matsolvers import SolverBase, add_solver

# --- patch: let Dedalus assemble a rectangular (no-tau) system ---
_src = textwrap.dedent(inspect.getsource(ss.Subproblem.build_matrices)).replace(
    "if valid_eqn.nnz != valid_var.nnz:", "if False:")
_g = ss.__dict__.copy(); exec(_src, _g); ss.Subproblem.build_matrices = _g['build_matrices']

@add_solver
class RectQR(SolverBase):
    """Rectangular least-squares QR for a tall no-tau operator: A (m x n, m>n) -> tall QR,
    drop the trailing m-n rows of R, back-substitute."""
    sparse = True
    def __init__(self, matrix, solver=None):
        A = np.asarray(sp.csr_matrix(matrix).todense())
        self.m, self.n = A.shape
        self.Q, self.R = np.linalg.qr(A, mode='complete')
    def solve(self, b):
        b = np.asarray(b)
        return solve_triangular(self.R[:self.n, :], (self.Q.conj().T @ b)[:self.n])

Nphi, Ntheta, Nr = 24, 12, 24
# S scaled so the conductive equilibrium T ~ O(1) (S ~ 6*kappa); gentle Ra + small dt so
# BOTH the tau-free and standard runs stay CFL-stable through the developed convection.
Ra, dt, t_max = 1.5e3, 1e-3, 8.0
S = 6.0 * Ra**-0.5

GRID = {}
def build(taufree):
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=np.float64)
    ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=1, dtype=np.float64)
    sphere = ball.surface
    phi, theta, r = dist.local_grids(ball)
    GRID['phi'] = np.reshape(phi, -1); GRID['r'] = np.reshape(r, -1)
    u = dist.VectorField(coords, bases=ball); p = dist.Field(bases=ball); T = dist.Field(bases=ball)
    tau_p = dist.Field()
    uo = dist.VectorField(coords, bases=ball); To = dist.Field(bases=ball)     # previous step
    rvec = dist.VectorField(coords, bases=ball.radial_basis); rvec['g'][2] = r
    nu, kappa, di = Ra**-0.5, Ra**-0.5, 1.0 / dt
    lap = lambda A: d3.Laplacian(A, coords); grad = lambda A: d3.Gradient(A, coords); div = lambda A: d3.Divergence(A)
    ns = dict(u=u, p=p, T=T, tau_p=tau_p, uo=uo, To=To, rvec=rvec, nu=nu, kappa=kappa, Ra=Ra,
              di=di, S=S, lap=lap, grad=grad, div=div, integ=lambda A: d3.integ(A))
    if taufree:
        prob = d3.LBVP([p, u, T, tau_p], namespace=ns)
        prob.add_equation("div(u) + tau_p = 0")
        prob.add_equation("di*u - nu*lap(u) + grad(p) - Ra*T*rvec = di*uo - uo@grad(uo)")
        prob.add_equation("di*T - kappa*lap(T) = di*To + S - uo@grad(To)")
        mat = RectQR
    else:
        tu = dist.VectorField(coords, bases=sphere); tT = dist.Field(bases=sphere)
        lift = lambda A: d3.Lift(A, ball, -1); ns.update(tu=tu, tT=tT, lift=lift)
        prob = d3.LBVP([p, u, T, tau_p, tu, tT], namespace=ns)
        prob.add_equation("div(u) + tau_p = 0")
        prob.add_equation("di*u - nu*lap(u) + grad(p) - Ra*T*rvec + lift(tu) = di*uo - uo@grad(uo)")
        prob.add_equation("di*T - kappa*lap(T) + lift(tT) = di*To + S - uo@grad(To)")
        mat = 'SuperluNaturalSpsolve'
    prob.add_equation("u(r=1) = 0"); prob.add_equation("T(r=1) = 0"); prob.add_equation("integ(p) = 0")
    solver = prob.build_solver(matsolver=mat)
    return solver, u, T, uo, To

s_tf, u_tf, T_tf, uo_tf, To_tf = build(True)
s_st, u_st, T_st, uo_st, To_st = build(False)
seed = 0.05 * np.random.default_rng(0).standard_normal(T_tf['g'].shape)
T_tf['g'] = seed.copy(); T_st['g'] = seed.copy()

# equatorial slice (theta = pi/2): T[:, jmid, :] over (phi, r) -> a disk in (x,y)
jmid = Ntheta // 2
def eq_slice(Tf):
    g = Tf['g'][:, jmid, :]                        # (Nphi, Nr)
    return np.vstack([g, g[0:1, :]])               # close periodic phi
pp = np.concatenate([GRID['phi'], [2*np.pi]]); rr = GRID['r']
PHI, R = np.meshgrid(pp, rr, indexing='ij')
X, Y = R*np.cos(PHI), R*np.sin(PHI)

frames_tf, frames_st, times, diffs = [], [], [], []
save_dt, nxt, step, t0 = 0.05, 0.0, 0, time.perf_counter()
while s_tf.iteration * dt < t_max:
    for s, u, T, uo, To in ((s_tf, u_tf, T_tf, uo_tf, To_tf), (s_st, u_st, T_st, uo_st, To_st)):
        uo['g'] = u['g']; To['g'] = T['g']; s.solve()
    step += 1
    if step * dt >= nxt:
        frames_tf.append(eq_slice(T_tf)); frames_st.append(eq_slice(T_st))
        times.append(step * dt)
        tmx = np.max(np.abs(T_tf['g']))
        diffs.append(np.max(np.abs(T_tf['g'] - T_st['g'])) / (np.max(np.abs(T_st['g'])) + 1e-30))
        if step % 20 == 0 or step * dt < 0.3:
            print(f"  t={step*dt:5.2f} tf_Tmax={tmx:.3f} reldiff={diffs[-1]:.2e}", flush=True)
        if not np.isfinite(tmx) or tmx > 50:
            print("  BLOWUP -- stopping"); break
        nxt += save_dt
print(f"ran {step} steps to t={step*dt:.2f} in {time.perf_counter()-t0:.0f}s, {len(times)} frames")

vmax = np.percentile(np.abs(np.array(frames_st)), 99)
fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
kw = dict(cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')
im0 = ax[0].pcolormesh(X, Y, frames_tf[0], **kw); im1 = ax[1].pcolormesh(X, Y, frames_st[0], **kw)
for a, ttl in zip(ax, ['tau-free  (rectangular RectQR)', 'standard tau  (SuperLU)']):
    a.set_aspect('equal'); a.axis('off'); a.set_title(ttl)
    a.add_artist(plt.Circle((0, 0), 1.0, fill=False, color='k', lw=1))
sup = fig.suptitle(''); fig.tight_layout(rect=[0, 0, 1, 0.93])
def upd(i):
    im0.set_array(frames_tf[i].ravel()); im1.set_array(frames_st[i].ravel())
    sup.set_text(f'Internally-heated ball convection (equatorial T)   t = {times[i]:4.2f}   '
                 f'Ra={Ra:.0e}  Nr={Nr}   reldiff={diffs[i]:.1e}')
    return im0, im1, sup
FuncAnimation(fig, upd, frames=len(times), interval=60).save(
    'ball_convection_taufree_vs_standard.mp4', dpi=110, fps=18)
print("wrote ball_convection_taufree_vs_standard.mp4")
