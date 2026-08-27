"""Three-panel cutaway animation: standard-tau Dedalus (left) vs tau-free banded
ride at Nr=96 (middle) vs tau-free banded ride at Nr=160 (right), internally-heated
ball convection, Ra=1e6.

Same EXACT cutaway construction as the example's plot_ball.py (outer T(r=1) face +
meridional T(phi=0), T(phi=3pi/2) faces; magma; per-frame +/-3 std).  Each panel
builds its own geometry from its own radial grid, so the Nr=160 panel's finer
meridional mesh renders correctly.  The middle panel rings at the boundary around
t~13 (under-resolved thermal boundary layer); the right panel shows that ring
vanishing when the boundary layer is resolved -- a spectral-convergence artifact,
not solver error.

    python plot_ball_3way.py            # writes ball_ihc_3way_tau_v4_hires.mp4
"""
import matplotlib; matplotlib.use('Agg')
import glob, re, h5py, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

R = 1.0
phis, phie = 0.0, 3*np.pi/2

def build_s2_vertices(phi, theta):
    phi = phi.ravel(); phi_vert = np.concatenate([phi, [2*np.pi]]); phi_vert -= phi_vert[1]/2
    theta = theta.ravel(); theta_mid = (theta[:-1]+theta[1:])/2
    return phi_vert, np.concatenate([[np.pi], theta_mid, [0]])

def build_spherical_vertices(phi, theta, r, Ri, Ro):
    phi_vert, theta_vert = build_s2_vertices(phi, theta)
    r = r.ravel(); r_mid = (r[:-1]+r[1:])/2
    return phi_vert, theta_vert, np.concatenate([[Ri], r_mid, [Ro]])

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

def geometry(phi, theta, r):
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
    return dict(xo=xo, yo=yo, zo=zo, xp=xp, yp=yp, zp=zp, phi_pick=phi_pick)

cmap = plt.cm.magma; norm = Normalize(-3, 3)
def faces(O, S, E, i, phi_pick):
    do = O[i][phi_pick]; ds, de = S[i], E[i]
    mer = np.concatenate([ds, de], axis=0); mu, sd = mer.mean(), mer.std()
    fo = cmap(norm((do-mu)/sd)); fs = cmap(norm((ds-mu)/sd)); fe = cmap(norm((de-mu)/sd))
    fp = np.concatenate([fe.transpose((1,0,2))[::-1], fs.transpose((1,0,2))], axis=0)
    return fo, fp

# --- load all three ---
panels = [
    ('standard tau  (Nr=96, SuperLU)',      load('slices', 'slices')),
    ('tau-free ride  (Nr=96)',               load('slices_tf3', 'slices_tf3')),
    ('tau-free ride  (Nr=160, resolved)',    load('slices_tf4', 'slices_tf4')),
]
nf = min(len(p[1][3]) for p in panels)
tref = panels[0][1][3]
print(f"frames: {[len(p[1][3]) for p in panels]}  using {nf}", flush=True)

fig = plt.figure(figsize=(18, 6.6))
arts = []
for k, (title, (O, S, E, t, phi, theta, r)) in enumerate(panels):
    ax = fig.add_axes([k/3.0, 0.02, 1/3.0, 0.9], projection='3d')
    ax.set_box_aspect((1,1,1)); ax.set_xlim(-0.7,0.7); ax.set_ylim(-0.7,0.7); ax.set_zlim(-0.7,0.7)
    ax.axis('off'); ax.view_init(azim=-75); ax.set_title(title, fontsize=13, pad=-2)
    g = geometry(phi, theta, r)
    fo, fp = faces(O, S, E, 0, g['phi_pick'])
    sr = ax.plot_surface(g['xo'], g['yo'], g['zo'], facecolors=fo, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
    sp = ax.plot_surface(g['xp'], g['yp'], g['zp'], facecolors=fp, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
    arts.append((O, S, E, g, sr, sp))
sup = fig.suptitle('', fontsize=15, y=0.98)

def upd(i):
    for (O, S, E, g, sr, sp) in arts:
        fo, fp = faces(O, S, E, i, g['phi_pick'])
        sr.set_facecolors(fo.reshape(-1, 4)); sp.set_facecolors(fp.reshape(-1, 4))
    sup.set_text(f'Internally-heated ball convection   Ra=1e6   t = {tref[i]:5.2f}')
    return ()

FuncAnimation(fig, upd, frames=nf, interval=90).save('ball_ihc_3way_tau_v4_hires.mp4', dpi=120, fps=12)
print(f"wrote ball_ihc_3way_tau_v4_hires.mp4 ({nf} frames)", flush=True)
