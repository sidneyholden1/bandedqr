"""Side-by-side 2D Rayleigh-Benard buoyancy animation:
bandqr (tau-free, vorticity-streamfunction) vs standard Dedalus (tau + SuperLU).

Both formulations are stepped in lockstep with the same adaptive timestep; the movie
shows the buoyancy field for each, with the running max difference in the title.
Writes rbc_taufree_vs_standard.mp4.
"""
import matplotlib; matplotlib.use('Agg')
import numpy as np, time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import dedalus.public as d3
from dedalus_bandqr import BandQR, boundary_border

Nx, Ny, Lx = 128, 64, 4.0
Ra, Pr = 1e6, 1.0
nu, kappa = (Ra / Pr)**-0.5, (Ra * Pr)**-0.5
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)
xb = d3.RealFourier(coords['x'], Nx, (0, Lx)); yb = d3.Chebyshev(coords['y'], Ny, (-1, 1))
x = dist.local_grid(xb); y = dist.local_grid(yb)
lap = lambda A: d3.Laplacian(A, coords)
dx = lambda A: d3.Differentiate(A, coords['x']); dy = lambda A: d3.Differentiate(A, coords['y'])

def build(taufree):
    psi = dist.Field(bases=(xb, yb)); w = dist.Field(bases=(xb, yb)); b = dist.Field(bases=(xb, yb))
    u, v = dy(psi), -dx(psi)
    if taufree:
        ns = dict(psi=psi, w=w, b=b, u=u, v=v, nu=nu, kappa=kappa, lap=lap, dx=dx, dy=dy)
        prob = d3.IVP([w, psi, b], namespace=ns)
        prob.add_equation("dt(w) - nu*lap(w) - dx(b) = -(u*dx(w) + v*dy(w))")
        prob.add_equation("lap(psi) + w = 0")
        prob.add_equation("dt(b) - kappa*lap(b) = -(u*dx(b) + v*dy(b))")
    else:
        lift = lambda A, n: d3.Lift(A, yb.derivative_basis(2), n)
        t = [dist.Field(bases=xb) for _ in range(6)]
        ns = dict(psi=psi, w=w, b=b, u=u, v=v, nu=nu, kappa=kappa, lap=lap, dx=dx, dy=dy, lift=lift,
                  t1=t[0], t2=t[1], t3=t[2], t4=t[3], t5=t[4], t6=t[5])
        prob = d3.IVP([w, psi, b] + t, namespace=ns)
        prob.add_equation("dt(w) - nu*lap(w) - dx(b) + lift(t1,-1) + lift(t2,-2) = -(u*dx(w) + v*dy(w))")
        prob.add_equation("lap(psi) + w + lift(t3,-1) + lift(t4,-2) = 0")
        prob.add_equation("dt(b) - kappa*lap(b) + lift(t5,-1) + lift(t6,-2) = -(u*dx(b) + v*dy(b))")
    prob.add_equation("psi(y=-1)=0"); prob.add_equation("psi(y=+1)=0")
    prob.add_equation("dy(psi)(y=-1)=0"); prob.add_equation("dy(psi)(y=+1)=0")
    prob.add_equation("b(y=-1)=0.5"); prob.add_equation("b(y=+1)=-0.5")
    if taufree:
        boundary_border(prob, orders=[2, 2, 2])
    solver = prob.build_solver(d3.RK222, matsolver=(BandQR if taufree else 'SuperluNaturalSpsolve'))
    return solver, b, w, u

s_tf, b_tf, w_tf, u_tf = build(True)
s_st, b_st, w_st, u_st = build(False)
rng = np.random.default_rng(0); seed = 0.05 * rng.standard_normal(b_tf['g'].shape)
b_tf['g'] = -0.5 * y + seed; b_st['g'] = -0.5 * y + seed

Btf, Bst, Wtf, Wst, T, D = [], [], [], [], [], []
t_max, save_dt, dt, step, nxt, t0 = 30.0, 0.2, 0.01, 0, 0.0, time.perf_counter()
while s_tf.sim_time < t_max:
    if step % 25 == 0:
        um = np.max(np.abs(u_tf.evaluate()['g'])); dt = min(0.015, 0.35 * (Lx / Nx) / (um + 1e-9))
    s_tf.step(dt); s_st.step(dt); step += 1
    if s_tf.sim_time >= nxt:
        Btf.append(b_tf['g'].T.copy()); Bst.append(b_st['g'].T.copy())
        Wtf.append(w_tf['g'].T.copy()); Wst.append(w_st['g'].T.copy())
        T.append(s_tf.sim_time)
        D.append(max(np.max(np.abs(b_tf['g'] - b_st['g'])), np.max(np.abs(w_tf['g'] - w_st['g']))))
        nxt += save_dt
print(f"ran {step} steps to t={s_tf.sim_time:.1f} in {time.perf_counter()-t0:.0f}s, "
      f"{len(T)} frames, final max|diff|={D[-1]:.1e}")

ext = [0, Lx, -1, 1]
wscale = np.percentile(np.abs(np.array(Wtf)), 99)              # symmetric vorticity scale
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
bkw = dict(origin='lower', extent=ext, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
wkw = dict(origin='lower', extent=ext, cmap='PRGn', vmin=-wscale, vmax=wscale, aspect='auto')
imB1 = ax[0, 0].imshow(Btf[0], **bkw); imB2 = ax[0, 1].imshow(Bst[0], **bkw)
imW1 = ax[1, 0].imshow(Wtf[0], **wkw); imW2 = ax[1, 1].imshow(Wst[0], **wkw)
ax[0, 0].set_title('bandqr  (tau-free)'); ax[0, 1].set_title('Dedalus  (standard tau + SuperLU)')
ax[0, 0].set_ylabel('buoyancy\n\ny'); ax[1, 0].set_ylabel('vorticity\n\ny')
for a in ax[1, :]: a.set_xlabel('x')
sup = fig.suptitle(''); fig.tight_layout(rect=[0, 0, 1, 0.94])
def upd(i):
    imB1.set_data(Btf[i]); imB2.set_data(Bst[i]); imW1.set_data(Wtf[i]); imW2.set_data(Wst[i])
    sup.set_text(f'2D Rayleigh-Benard    t = {T[i]:5.1f}    max|difference| = {D[i]:.1e}')
    return imB1, imB2, imW1, imW2, sup
FuncAnimation(fig, upd, frames=len(T), interval=50).save('rbc_taufree_vs_standard.mp4', dpi=110, fps=20)
print("wrote rbc_taufree_vs_standard.mp4")
