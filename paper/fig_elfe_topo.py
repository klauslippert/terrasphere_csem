#!/usr/bin/env python3
"""Figure elfe_topo: comparison with elfe3D over topography.  Case T1 (grounded wire in line, tests/validation/3D/
elfe3d_topo/cases/T1_wire_inline), x-profile, hill and valley of 90 m: Re and Im of E_x, E_z (zero on a flat
surface: the terrain effect alone) and H_y at 0.1 Hz, 10 Hz and 1 kHz; series of this package (results/fields.npz,
lines) and elfe3D (every second receiver, circles; band nf below 100 Hz, hf1k above, H = -H_elfe3D).
Usage: python3 fig_elfe_topo.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
V3D = os.path.join(os.path.dirname(HERE), 'tests', 'validation', '3D'); sys.path.insert(0, V3D)
from style import plt, save, GH_RC, COL
from terrasphere_csem.config import Config
from compare_elfe3d import model_name
from report_elfe3d_sphere import elfe_run

CASE = os.path.join(V3D, 'elfe3d_topo', 'cases', 'T1_wire_inline')
COMPS = [(0, '$E_x$', 1e9, 'nV/m'), (2, '$E_z$', 1e9, 'nV/m'), (4, '$H_y$', 1e6, r'$\mu$A/m')]
PROF = 'x'

plt.rcParams.update(GH_RC)
st = Config(CASE)
d = np.load(os.path.join(CASE, 'results', 'fields.npz'))
ser = np.concatenate([d['E'], d['H']], axis=2)                      # (nh, nf, 6, P), as written by terrasphere_csem
sel = d['profile'] == PROF; x = d['px'][sel]
fig, ax = plt.subplots(4, 3, figsize=(7.2, 7.6), sharex=True)
for im, H in enumerate((90.0, -90.0)):
    ih = list(d['heights']).index(H)
    for jf, f in enumerate(d['freqs']):
        el = elfe_run(st, model_name(H), f, len(d['px']))
        for jc, (k, lab, sc, unit) in enumerate(COMPS):
            for ir, (part, fn) in enumerate((('Re', np.real), ('Im', np.imag))):
                a = ax[2 * im + ir, jc]
                a.plot(x, sc * fn(ser[ih, jf, k, sel]), color=COL[jf], label=f'{f:g} Hz' if (im, ir, jc) == (0, 0, 0) else None)
                a.plot(x[::2], sc * fn(el[k, sel][::2]), 'o', ms=2.6, mfc='none', mew=0.8, color=COL[jf])
                a.set_title(f'{"hill" if H > 0 else "valley"} {abs(H):g} m: {part} {lab}')
                a.set_ylabel(f'{part} {lab} ({unit})')
for a in ax[-1]:
    a.set_xlabel('x (m)')
ax[0, 0].plot([], [], 'k-', label='series'); ax[0, 0].plot([], [], 'ko', ms=2.6, mfc='none', label='elfe3D')
ax[0, 0].legend(loc='best')
fig.tight_layout()
save(fig, 'elfe_topo', sys.argv[1], png=False)
