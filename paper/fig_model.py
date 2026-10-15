#!/usr/bin/env python3
"""Figure model: model.  Plan view (sources: wire, grounded dipole, loop around the hill; terrain contours; sphere outline;
receiver profiles) and cross-section y = 0 (hill and depression, layer interface, sphere).
Usage: python3 fig_model.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from style import plt, C, save

s = 150.0
h = lambda x, y, H: H * np.exp(-(x ** 2 + y ** 2) / (2 * s ** 2))
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1.0, 1.3]))
a = ax[0]
x = np.linspace(-1500, 1500, 401); X, Y = np.meshgrid(x, np.linspace(-1700, 1500, 401))
cs = a.contour(X, Y, h(X, Y, 30.0), levels=[1, 10, 20, 28], colors=C['flat'], linewidths=0.6)
a.plot([-500, 500], [-1500, -1500], color='k', lw=2.0); a.plot([-500, 500], [-1500, -1500], 'ks', ms=3)
a.text(0, -1640, 'grounded wire / dipole', ha='center', fontsize=6.5)
A = 1300.0
a.plot([-A, A, A, -A, -A], [-A, -A, A, A, -A], color=C['third'], lw=1.0, ls='--')
a.text(-A + 40, A - 150, 'loop (optional)', color=C['third'], fontsize=6.5)
t = np.linspace(0, 2 * np.pi, 100); a.plot(60 * np.cos(t), 60 * np.sin(t), color=C['sphere'], lw=1.2)
a.plot([0, 0], [-600, 600], color=C['hill'], lw=1.0); a.plot([-600, 600], [0, 0], color=C['hill'], lw=1.0, ls=':')
a.set_aspect('equal'); a.set_xlabel('x (m)'); a.set_ylabel('y (m)'); a.set_title('plan view', fontsize=8.5)
b = ax[1]
xs = np.linspace(-700, 700, 701)
b.fill_between(xs, -330, 300, color='white')
b.plot(xs, h(xs, 0, 30.0), color=C['hill'], label='hill, H = 30 m')
b.plot(xs, h(xs, 0, -30.0), color=C['depression'], label='depression, H = -30 m')
b.axhline(0, color=C['flat'], lw=0.6, ls=':')
b.axhline(-300, color=C['layer'], lw=1.2); b.text(-680, -290, r'interface, $\sigma_2$ below', fontsize=6.5)
b.text(-680, -60, r'top layer $\sigma_1$', fontsize=6.5); b.text(-680, 40, 'air', fontsize=6.5)
b.add_patch(plt.Circle((0, -140), 60, fill=False, color=C['sphere'], lw=1.2))
b.text(70, -150, r'sphere $b$, $\sigma_{\rm sph}$', color=C['sphere'], fontsize=6.5)
b.set_xlim(-650, 650); b.set_ylim(-340, 120); b.set_aspect('equal'); b.set_xlabel('x (m)'); b.set_ylabel('z (m)')
b.set_title(r'section y = 0: $h=H\,\exp(-r^2/2s^2)$, s = 150 m', fontsize=8.5); b.legend(loc='upper right')
fig.tight_layout()
save(fig, 'model', sys.argv[1])
