"""Render the equatorial-slice temperature movie from the tau-free ball IHC run.

Reads /tmp/ball_frames.npz (frames of T on the equatorial plane, gathered across MPI
ranks by internally_heated_convection_taufree.py) and writes ball_ihc_taufree.mp4 --
a full disk in (x,y) = (r cos phi, r sin phi).
"""
import matplotlib; matplotlib.use('Agg')
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

d = np.load('/tmp/ball_frames.npz')
frames, times, gphi, gr = d['frames'], d['times'], d['gphi'], d['gr']   # frames: (Nf, Nphi, Nr)
pp = np.concatenate([gphi, [2*np.pi]])                                   # close the disk
PHI, R = np.meshgrid(pp, gr, indexing='ij')
X, Y = R*np.cos(PHI), R*np.sin(PHI)
def disk(fr): return np.vstack([fr, fr[0:1, :]])                         # wrap phi

# color scale from the fluctuation about the spherical mean
vmax = np.percentile(np.abs(frames - frames.mean(axis=1, keepdims=True)), 99)
fig, ax = plt.subplots(figsize=(6.2, 6.4))
im = ax.pcolormesh(X, Y, disk(frames[0]), cmap='RdBu_r', vmin=frames.min(), vmax=frames.max(), shading='gouraud')
ax.set_aspect('equal'); ax.axis('off')
ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, color='k', lw=1.2))
ttl = ax.set_title('')
def upd(i):
    im.set_array(disk(frames[i]).ravel())
    ttl.set_text(f'Internally-heated ball convection (tau-free RectQR, 128x64x96, Ra=1e6)\n'
                 f'equatorial T   t = {times[i]:5.2f}')
    return im, ttl
FuncAnimation(fig, upd, frames=len(times), interval=90).save('ball_ihc_taufree.mp4', dpi=120, fps=12)
print(f"wrote ball_ihc_taufree.mp4 ({len(times)} frames)")
