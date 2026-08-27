"""Cutaway-ball 3D animation for the tau-free faithful run.

Reproduces the exact example's plot_ball.py visualization (outer T(r=1) surface + two
meridional faces T(phi=0), T(phi=3pi/2), matplotlib-3D, magma, +/-3 std) from the
MPI-gathered slices in /tmp/ball_cutaway_tf.npz.  Writes ball_ihc_taufree_cutaway.mp4.
"""
import matplotlib; matplotlib.use('Agg')
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

d = np.load('/tmp/ball_cutaway_tf.npz')
r1, p0, p32, times = d['r1'], d['p0'], d['p32'], d['times']   # r1:(Nf,Nphi,Nth)  p0/p32:(Nf,Nth,Nr)
gphi, gth, gr = d['gphi'], d['gth'], d['gr']
R = 1.0

def s2_vertices(phi, theta):
    pv = np.concatenate([phi, [2*np.pi]]); pv -= pv[1]/2
    tm = (theta[:-1]+theta[1:])/2; tv = np.concatenate([[np.pi], tm, [0]])
    return pv, tv
def sph_vertices(phi, theta, r):
    pv, tv = s2_vertices(phi, theta); rm = (r[:-1]+r[1:])/2
    rv = np.concatenate([[0.0], rm, [R]]); return pv, tv, rv
def to_cart(phi, theta, r):
    P, Th, Rr = np.meshgrid(phi, theta, r, indexing='ij')
    return (Rr*np.sin(Th)*np.cos(P), Rr*np.sin(Th)*np.sin(P), Rr*np.cos(Th))

# normalization from the meridional faces (std), like the example
allmer = np.concatenate([p0.ravel(), p32.ravel()])
mu, sd = allmer.mean(), allmer.std()
norm = lambda x: np.clip((x-mu)/sd, -3, 3)
cmap = plt.cm.magma; sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_clim(-3, 3)

# outer sphere (r=R): grid (phi, theta)
pv, tv = s2_vertices(gphi, gth)
xo, yo, zo = to_cart(pv, tv, np.array([R]))
xo, yo, zo = xo[:, :, 0], yo[:, :, 0], zo[:, :, 0]
# meridional faces: (theta, r) at phi=0 and phi=3pi/2
_, tvm, rvm = sph_vertices(gphi, gth, gr)
def mer_cart(phi0):
    X, Y, Z = to_cart(np.array([phi0]), tvm, rvm); return X[0], Y[0], Z[0]
xs, ys, zs = mer_cart(0.0)
xe, ye, ze = mer_cart(1.5*np.pi)

fig = plt.figure(figsize=(6.4, 6.6)); ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=-60)
def facecolors_outer(i): return cmap(sm.norm(norm(r1[i])))           # (Nphi,Nth)
def facecolors_mer(sl, i): return cmap(sm.norm(norm(sl[i])))          # (Nth,Nr)
surf_o = ax.plot_surface(xo, yo, zo, facecolors=facecolors_outer(0), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
surf_s = ax.plot_surface(xs, ys, zs, facecolors=facecolors_mer(p0, 0), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
surf_e = ax.plot_surface(xe, ye, ze, facecolors=facecolors_mer(p32, 0), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
ttl = ax.set_title('')
def upd(i):
    global surf_o, surf_s, surf_e
    surf_o.remove(); surf_s.remove(); surf_e.remove()
    surf_o = ax.plot_surface(xo, yo, zo, facecolors=facecolors_outer(i), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    surf_s = ax.plot_surface(xs, ys, zs, facecolors=facecolors_mer(p0, i), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    surf_e = ax.plot_surface(xe, ye, ze, facecolors=facecolors_mer(p32, i), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ttl.set_text(f'tau-free (RectQR) internally-heated ball convection\n128x64x96  Ra=1e6  adaptive dt   t = {times[i]:5.2f}')
    return ()
FuncAnimation(fig, upd, frames=len(times), interval=90).save('ball_ihc_taufree_cutaway.mp4', dpi=120, fps=12)
print(f"wrote ball_ihc_taufree_cutaway.mp4 ({len(times)} frames)")
