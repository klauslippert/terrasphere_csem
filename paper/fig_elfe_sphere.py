#!/usr/bin/env python3
"""Figure elfe_sphere: comparison with elfe3D of the sphere anomaly under the 90 m hill.  Case T1 (grounded wire in
line, tests/validation/3D/elfe3d_sphere/cases/T1_wire_inline: sphere 10 Ohm m, radius 180 m, centre 400 m deep),
y-profile: Re and Im of the anomaly A = V(sphere) - V(no sphere) of E_y, E_z and H_z at 0.1 Hz, 10 Hz and 1 kHz.
Series (lines): results/fields.npz of the case minus the hill of 90 m in elfe3d_topo/cases/T1_wire_inline; elfe3D
(every second receiver, circles): run with the sphere minus the reference run on the same mesh (sphere region with
the host resistivity), H = -H_elfe3D.
Usage: python3 fig_elfe_sphere.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
V3D = os.path.join(os.path.dirname(HERE), 'tests', 'validation', '3D'); sys.path.insert(0, V3D)
from style import plt, save, GH_RC, COL
from terrasphere_csem.config import Config
from compare_elfe3d import model_name
from report_elfe3d_sphere import elfe_run

CASE = os.path.join(V3D, 'elfe3d_sphere', 'cases', 'T1_wire_inline')
BASE = os.path.join(V3D, 'elfe3d_topo', 'cases', 'T1_wire_inline')
COMPS = [(1, '$E_y$', 1e9, 'nV/m'), (2, '$E_z$', 1e9, 'nV/m'), (5, '$H_z$', 1e6, r'$\mu$A/m')]
PROF = 'y'

plt.rcParams.update(GH_RC)
st = Config(CASE)
d = np.load(os.path.join(CASE, 'results', 'fields.npz')); b = np.load(os.path.join(BASE, 'results', 'fields.npz'))
H = float(d['heights'][0]); jb = list(b['heights']).index(H)
assert np.allclose(d['freqs'], b['freqs']) and np.allclose(d['px'], b['px']) and np.allclose(d['py'], b['py'])
A = np.concatenate([d['E'][0], d['H'][0]], axis=1) - np.concatenate([b['E'][jb], b['H'][jb]], axis=1)   # (nf, 6, P)
sel = d['profile'] == PROF; y = d['py'][sel]
fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True)
for jf, f in enumerate(d['freqs']):
    e, e0 = elfe_run(st, model_name(H), f, len(d['px'])), elfe_run(st, model_name(H) + 'ref', f, len(d['px']))
    for jc, (k, lab, sc, unit) in enumerate(COMPS):
        for ir, (part, fn) in enumerate((('Re', np.real), ('Im', np.imag))):
            a = ax[ir, jc]
            a.plot(y, sc * fn(A[jf, k, sel]), color=COL[jf], label=f'{f:g} Hz' if (ir, jc) == (0, 0) else None)
            a.plot(y[::2], sc * fn((e - e0)[k, sel][::2]), 'o', ms=2.6, mfc='none', mew=0.8, color=COL[jf])
            a.set_title(f'sphere anomaly: {part} {lab}'); a.set_ylabel(f'{part} {lab} ({unit})')
for a in ax[-1]:
    a.set_xlabel('y (m)')
ax[0, 0].plot([], [], 'k-', label='series'); ax[0, 0].plot([], [], 'ko', ms=2.6, mfc='none', label='elfe3D')
fig.tight_layout()
fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=5)
save(fig, 'elfe_sphere', sys.argv[1], png=False)
