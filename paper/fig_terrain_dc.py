#!/usr/bin/env python3
"""Figure terrain_dc: direct-current check of the terrain series.  E_x and E_y along the x-profile (y = 0) over
hill and valley (H = +-90 m) from the H-formulation (terrain.coefficients at f = 1e-6 Hz, lines) and the independent
scalar-potential expansion (terrain_dc.coefficients, markers); lower row: difference between the two relative to the
profile maximum.  Grounded wire over the half-space 0.01 S/m (cases.py 'wire_halfspace'), 12 orders, 512^2 x 10 m.
Usage: python3 fig_terrain_dc.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from style import plt, save, GH_RC, GH_C as C
from terrasphere_csem import terrain, terrain_dc
from terrasphere_csem.terrain import partial_sum
import cases as FL

plt.rcParams.update(GH_RC)
g = terrain.Grid(FL.sources('wire_halfspace'), n=512, dx=10.0, s=FL.S_HILL)
CE, _ = terrain.coefficients([1e-6], g, FL.ground('wire_halfspace'), nord=12)
CD = terrain_dc.coefficients(g, FL.ground('wire_halfspace'), 12)
sel = np.abs(g.x) <= 600.0
x = g.x[sel]
fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
for j, (k, lab) in enumerate([(0, '$E_x$'), (1, '$E_y$')]):
    a, b = ax[0, j], ax[1, j]
    a.plot(x, partial_sum(CE[0], 0.0)[k, 1, sel].real * 1e9, color=C['flat'], label='flat')
    for H, col, nm in [(90.0, C['hill'], 'hill'), (-90.0, C['valley'], 'valley')]:
        vh = partial_sum(CE[0], H)[k, 1, sel].real; vd = partial_sum(CD, H)[k, 1, sel].real
        a.plot(x, vh * 1e9, color=col, label=f'{nm}, H-formulation')
        a.plot(x[::4], vd[::4] * 1e9, 'o', ms=3, mfc='none', color=col, label=f'{nm}, potential')
        b.semilogy(x, np.abs(vh - vd) / np.abs(vd).max(), color=col)
    a.set_ylabel(f'{lab} (nV/m)'); a.set_title(f'{lab} along y = 0, direct current', fontsize=8.5)
    b.set_ylabel('|difference| / max'); b.set_xlabel('x (m)'); b.set_ylim(1e-12, 1e-4)
ax[0, 0].legend(loc='best')
save(fig, 'terrain_dc', sys.argv[1])
