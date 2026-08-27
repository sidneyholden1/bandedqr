"""Side-by-side cutaway animation: standard-tau Dedalus (left) vs tau-free banded
ride RideQRRect (right), internally-heated ball convection, 128x64x96, Ra=1e6.

Uses the EXACT cutaway construction of the example's plot_ball.py (outer T(r=1)
face + two meridional faces T(phi=0), T(phi=3pi/2); magma; per-frame +/-3 std
normalisation from the meridional data).  Both runs are their own faithful
plot_ball render; placing them side by side shows the tau-free banded-ride
solution is the same convection as standard tau -- identical through onset, then
diverging only chaotically at late time (a different-but-convergent
discretisation, not solver error).

    python plot_ball_sidebyside.py            # writes ball_ihc_tau_vs_v4.mp4
"""
import matplotlib; matplotlib.use('Agg')
import glob, re, h5py, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

R = 1.0
phis, phie = 0.0, 3*np.pi/2          # the two meridional cut faces

def build_s2_vertices(phi, theta):
    phi = phi.ravel(); phi_vert = np.concatenate([phi, [2*np.pi]]); phi_vert -= phi_vert[1]/2
    theta = theta.ravel(); theta_mid = (theta[:-1]+theta[1:])/2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    return phi_vert, theta_vert

def build_spherical_vertices(phi, theta, r, Ri, Ro):
    phi_vert, theta_vert = build_s2_vertices(phi, theta)
    r = r.ravel(); r_mid = (r[:-1]+r[1:])/2
    r_vert = np.concatenate([[Ri], r_mid, [Ro]])
    return phi_vert, theta_vert, r_vert

def s2c(phi, theta, r):
    phi, theta, r = np.meshgrid(phi, theta, r, indexing='ij')
    return np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])

def load(dirpath, prefix):
    files = sorted(glob.glob(f"{dirpath}/{prefix}_s*.h5"),
                   key=lambda f: int(re.search(r'_s(\d+)\.h5', f).group(1)))
    O, S, E, times = [], [], [], []
    phi = theta = r = None
    for fn in files:
        with h5py.File(fn, 'r') as f:
            do, ds, de = f['tasks']['T(r=1)'], f['tasks']['T(phi=0)'], f['tasks']['T(phi=3/2*pi)']
            if phi is None:
                phi = do.dims[1][0][:].ravel(); theta = do.dims[2][0][:].ravel(); r = ds.dims[3][0][:].ravel()
            for i in range(do.shape[0]):
                O.append(do[i, :, :, 0]); S.append(ds[i, 0, :, :]); E.append(de[i, 0, :, :])
                times.append(float(f['scales/sim_time'][i]))
    return np.array(O), np.array(S), np.array(E), np.array(times), phi, theta, r

# --- load both runs ---
Ol, Sl, El, tl, phi, theta, r = load('slices', 'slices')            # standard tau
Or_, Sr, Er, tr, _, _, _      = load('slices_tf3', 'slices_tf3')    # v4 banded ride
nf = min(len(tl), len(tr))
print(f"standard frames={len(tl)}  v4 frames={len(tr)}  using {nf}")

# --- fixed cutaway geometry (identical for both panels) ---
phi_vert, theta_vert, r_vert = build_spherical_vertices(phi, theta, r, 0, R)
pick = (phis < phi_vert) * (phi_vert < phie)
phi_vert_o = np.concatenate([[phis], phi_vert[pick], [phie]])
xo, yo, zo = s2c(phi_vert_o, theta_vert, [R])[:, :, :, 0]
xs, ys, zs = s2c([phis], theta_vert, r_vert)[:, 0, :, :]
xe, ye, ze = s2c([phie], theta_vert, r_vert)[:, 0, :, :]
xp = np.concatenate([xe.T[::-1], xs.T[1:]], axis=0)
yp = np.concatenate([ye.T[::-1], ys.T[1:]], axis=0)
zp = np.concatenate([ze.T[::-1], zs.T[1:]], axis=0)
dphi = phi[1] - phi[0]
phi_pick = (phis-dphi/2 < phi) * (phi < phie+dphi/2)
cmap = plt.cm.magma; norm = Normalize(-3, 3)

def faces(O, S, E, i):                         # per-frame +/- 3 std, exactly like plot_ball.py
    do = O[i][phi_pick]; ds, de = S[i], E[i]
    mer = np.concatenate([ds, de], axis=0); mu, sd = mer.mean(), mer.std()
    fo = cmap(norm((do-mu)/sd)); fs = cmap(norm((ds-mu)/sd)); fe = cmap(norm((de-mu)/sd))
    fp = np.concatenate([fe.transpose((1,0,2))[::-1], fs.transpose((1,0,2))], axis=0)
    return fo, fp

fig = plt.figure(figsize=(13, 7))
axL = fig.add_axes([0.0, 0.02, 0.5, 0.9], projection='3d')
axR = fig.add_axes([0.5, 0.02, 0.5, 0.9], projection='3d')
def setup(ax, title):
    ax.set_box_aspect((1,1,1)); ax.set_xlim(-0.7,0.7); ax.set_ylim(-0.7,0.7); ax.set_zlim(-0.7,0.7)
    ax.axis('off'); ax.view_init(azim=-75); ax.set_title(title, fontsize=13, pad=-2)
setup(axL, 'standard tau  (Dedalus, SuperLU)'); setup(axR, 'tau-free  (banded ride, RideQRRect)')

foL, fpL = faces(Ol, Sl, El, 0); foR, fpR = faces(Or_, Sr, Er, 0)
srL = axL.plot_surface(xo, yo, zo, facecolors=foL, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
spL = axL.plot_surface(xp, yp, zp, facecolors=fpL, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
srR = axR.plot_surface(xo, yo, zo, facecolors=foR, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
spR = axR.plot_surface(xp, yp, zp, facecolors=fpR, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
sup = fig.suptitle('', fontsize=14, y=0.97)

def upd(i):
    foL, fpL = faces(Ol, Sl, El, i); foR, fpR = faces(Or_, Sr, Er, i)
    srL.set_facecolors(foL.reshape(-1,4)); spL.set_facecolors(fpL.reshape(-1,4))
    srR.set_facecolors(foR.reshape(-1,4)); spR.set_facecolors(fpR.reshape(-1,4))
    sup.set_text(f'Internally-heated ball convection   128x64x96  Ra=1e6   t = {tl[i]:5.2f}')
    return ()

FuncAnimation(fig, upd, frames=nf, interval=90).save('ball_ihc_tau_vs_v4.mp4', dpi=120, fps=12)
print(f"wrote ball_ihc_tau_vs_v4.mp4 ({nf} frames)")
