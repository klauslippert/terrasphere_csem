#!/usr/bin/env python3
"""Figure elfe_setup: setup of the comparison with elfe3D (tests/validation/3D/elfe3d_topo and elfe3d_sphere).
Left: plan view with the four sources T1-T4 (cases/<case>/control.toml), the x- and y-profile of receivers, the
Gaussian terrain (circles at s and 2s) and the outline of the sphere.  Right: section y = 0 with the terrain of
+-30 m and +-90 m (to scale), the layer interfaces with their resistivities and the sphere (only with the 90 m hill).
Usage: python3 fig_elfe_setup.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
VAL = os.path.join(os.path.dirname(HERE), 'tests', 'validation'); sys.path.insert(0, VAL)
from style import plt, save, GH_RC, COL, GH_C
from terrasphere_csem.config import Config
from plotting import source_polylines

TOPO = os.path.join(VAL, '3D', 'elfe3d_topo', 'cases')
SPHERE = os.path.join(VAL, '3D', 'elfe3d_sphere', 'cases', 'T1_wire_inline')
CASES = [('T1_wire_inline', 'T1 wire in line'), ('T2_wire_broadside', 'T2 wire broadside'),
         ('T3_loop_offset', 'T3 offset loop'), ('T4_loop_large', 'T4 large loop')]

plt.rcParams.update(GH_RC)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
for (c, lab), col in zip(CASES, COL):
    st = Config(os.path.join(TOPO, c))
    for k, s in enumerate(source_polylines(st)):
        s = np.array(s); a1.plot(s[:, 0], s[:, 1], '-', color=col, lw=1.6, label=lab if k == 0 else None)
px, py, prof = st.receivers()
a1.plot(px, py, '.', ms=1.5, color='k', label='receivers')
for r, ls in ((st.s, '-'), (2 * st.s, '--')):
    a1.add_patch(plt.Circle((0, 0), r, fill=False, color='0.4', ls=ls, lw=0.7))
sp = Config(SPHERE).sphere; rs = float(sp['radius']); xc, yc, dc = (float(v) for v in sp['center'])
a1.add_patch(plt.Circle((xc, yc), rs, fill=False, color=GH_C['third'], lw=1.0, ls=':'))
a1.set_aspect('equal'); a1.set_xlim(-2000, 1500); a1.set_ylim(-1500, 1500)
a1.set_xlabel('x (m)'); a1.set_ylabel('y (m)'); a1.set_title('plan view')
a1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=6.5)
x = np.linspace(-800, 800, 800)
for H, col, ls in ((90.0, GH_C['hill'], '-'), (30.0, GH_C['hill'], '--'), (-30.0, GH_C['valley'], '--'),
                   (-90.0, GH_C['valley'], '-')):
    a2.plot(x, st.height(x, 0 * x, H), ls, color=col, lw=1.0, label=f'{"hill" if H > 0 else "valley"} {abs(H):g} m')
dep, sig = list(st.layers.depths[1:]), list(st.layers.sigma_g)
for z in dep:
    a2.axhline(-z, color='0.5', lw=0.7, ls='--')
for j, (zt, zb) in enumerate(zip([0.0] + dep, dep + [dep[-1] + 300.0])):
    a2.text(-780, -(zt + zb) / 2, r'%.3g $\Omega$m' % (1 / sig[j]), fontsize=7, va='center')
a2.add_patch(plt.Circle((xc, -dc), rs, color=GH_C['third'], alpha=0.3, lw=0))
a2.text(xc, -dc, r'sphere %.3g $\Omega$m' % (1 / float(sp['conductivity'])), fontsize=6.5, ha='center', va='center')
a2.set_aspect('equal'); a2.set_xlim(-800, 800); a2.set_ylim(-dep[-1] - 300, 150)
a2.set_xlabel('x (m)'); a2.set_ylabel('z (m)'); a2.set_title('section y = 0 (to scale)')
a2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.5)
fig.tight_layout()
save(fig, 'elfe_setup', sys.argv[1], png=False)
