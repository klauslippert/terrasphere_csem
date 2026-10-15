#!/usr/bin/env python3
"""Figure onedim: 1-D fields from our kernels vs empymod along a surface profile, three-layer earth.
Usage: python3 fig_onedim.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
TESTS = os.path.join(os.path.dirname(HERE), 'tests')   # reference solutions of the tests
sys.path += [os.path.join(TESTS, d) for d in ('verification', os.path.join('verification', 'empymod'), 'reference')]
from style import plt, save, LS_RC, COL
plt.rcParams.update(LS_RC)
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.onedim import primary_field
from test_onedim import emp_fields, AIR
depths = [0.0, 80.0, 250.0]; sig = [0.02, 0.002, 0.05]; st = LayerStack(depths, sig); f = 100.0; w = 2 * np.pi * f
xs = np.linspace(-600, 600, 25); y0 = 150.0
pts = [(x, y0, 0.0) for x in xs]; pts_d = [(x, y0, 120.0) for x in xs]
fig, ax = plt.subplots(2, 3, figsize=(10, 4.6), sharex=True)
data = {}
for ic, (typ, psi, zs, pp, lab) in enumerate((('HED', 0.3, 0.0, pts, 'HED at surface, rx at surface'), ('VMD', 0.0, 30.0, pts_d, 'VMD at 30 m, rx at 120 m depth'))):
    d = Dipole(typ, (0.0, 0.0, zs), 1.0, psi)
    E, H = primary_field(st, [d], pp, w); Ee, He = emp_fields(typ, d.pos, pp, depths, [AIR] + [1 / s for s in sig], f, psi)
    for j, (comp, ours, ref, nm) in enumerate(((0, E, Ee, '$|E_x|$ (V/m)'), (1, E, Ee, '$|E_y|$ (V/m)'), (2, H, He, '$|H_z|$ (A/m)'))):
        a = ax[ic, j]
        a.semilogy(xs, np.abs(ours[:, comp]), '-', color=COL[0], label='terrasphere_csem'); a.semilogy(xs, np.abs(ref[:, comp]), 'o', ms=3, mfc='none', color=COL[1], label='empymod')
        a2 = a.twinx(); a2.semilogy(xs, np.abs(ours[:, comp] - ref[:, comp]) / np.abs(ref[:, comp]), ':', color=COL[2]); a2.set_ylim(1e-10, 1e-3); a2.grid(False)
        if j == 2: a2.set_ylabel('rel. diff. (dotted)', color=COL[2])
        a.set_ylabel(nm); a.set_title(lab if j == 1 else '', fontsize=8)
        data[f'{typ}_{comp}_ours'] = ours[:, comp]; data[f'{typ}_{comp}_emp'] = ref[:, comp]
    ax[ic, 0].legend(fontsize=7)
for a in ax[1]: a.set_xlabel('$x$ (m), profile at $y=150$ m')
save(fig, 'onedim', sys.argv[1], png=False)
