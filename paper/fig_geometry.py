#!/usr/bin/env python3
"""Figure geometry: model geometry of the terrain study.  Plan view (wire, terrain contours, receiver profiles)
and cross-section along y = 0 of hill and depression for H = 30 and 90 m (s = 150 m).
Usage: python3 fig_geometry.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from style import plt, save, GH_RC, GH_C as C
from terrasphere_csem.terrain import terrain_height
import cases as FL

plt.rcParams.update(GH_RC)
h = lambda x, y, H: terrain_height(x, y, H, FL.S_HILL)
yw = FL.sources('wire_halfspace')[0].pos[1]                       # wire along x at y = -1500 m

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
a = ax[0]
x = np.linspace(-1000, 1000, 401); X, Y = np.meshgrid(x, np.linspace(-1700, 800, 501))
cs = a.contour(X, Y, h(X, Y, 30.0), levels=[1, 5, 10, 20, 28], colors=C['flat'], linewidths=0.6)
a.clabel(cs, fmt='%d m', fontsize=6)
xa, xb = -500.0, 500.0                                               # wire ends (cases.py)
a.plot([xa, xb], [yw, yw], color='k', lw=2.0); a.plot([xa, xb], [yw, yw], 'ks', ms=4)
a.text(0, yw - 120, 'grounded wire, 1 A', ha='center', fontsize=7)
a.plot([0, 0], [-600, 600], color=C['third'], lw=1.4); a.plot([-600, 600], [0, 0], color=C['third'], lw=1.4, ls='--')
a.text(20, 640, 'y-profile', fontsize=7); a.text(380, 40, 'x-profile', fontsize=7)
a.set_aspect('equal'); a.set_xlabel('x (m)'); a.set_ylabel('y (m)'); a.set_title('plan view (H = 30 m)', fontsize=8.5)
b = ax[1]
xs = np.linspace(-700, 700, 701)
for H, ls in [(30.0, '-'), (90.0, ':')]:
    b.plot(xs, h(xs, 0, H), color=C['hill'], ls=ls, label=f'hill, H = {H:.0f} m')
    b.plot(xs, h(xs, 0, -H), color=C['depression'], ls=ls, label=f'depression, H = {H:.0f} m')
b.axhline(0, color=C['flat'], lw=0.8)
b.text(-650, -120, r'$\sigma$ = 0.01 S/m', fontsize=7); b.text(-650, 110, r'air, $\sigma$ = 0', fontsize=7)
b.set_ylim(-150, 150); b.set_xlabel('x (m)'); b.set_ylabel('z (m)')
b.set_title(r'section y = 0: $h = \pm H\,\exp(-r^2/2s^2)$, s = 150 m', fontsize=8.5)
b.legend(fontsize=6.5, loc='upper right')
fig.tight_layout()
save(fig, 'geometry', sys.argv[1])
