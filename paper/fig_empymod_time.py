#!/usr/bin/env python3
"""Figure empymod_time: switch-off transients over the flat three-layer earth, series of this package (lines) against
empymod (circles, every fourth time), the strong component of each source on the x-profile (y = 0, five stations):
the electric field along the wire for the grounded wires T1 (in line) and T2 (broadside), dB_z/dt for the loops T3
(offset) and T4 (large).  |V| on log axes, solid/open circles where V > 0, dashed/filled circles where V < 0.
Data: tests/validation/1D/empymod_time/cases/<case>/{results,empymod/out}/transients.npz; all components, both
profiles and the deviation tables are in the report of that folder.
Usage: python3 fig_empymod_time.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
V1D = os.path.join(os.path.dirname(HERE), 'tests', 'validation', '1D'); sys.path.insert(0, V1D)
from style import plt, save, GH_RC, COL
from compare_empymod_time import load

CASES = [('T1_wire_inline', 'T1 wire in line', 'E', 0, '$E_x$', 1e3, 'mV/m'),
         ('T2_wire_broadside', 'T2 wire broadside', 'E', 1, '$E_y$', 1e3, 'mV/m'),
         ('T3_loop_offset', 'T3 offset loop', 'dBdt', 2, r'$\partial_t B_z$', 1.0, r'V/m$^2$'),
         ('T4_loop_large', 'T4 large loop', 'dBdt', 2, r'$\partial_t B_z$', 1.0, r'V/m$^2$')]

plt.rcParams.update(GH_RC)
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.6), sharex=True)
for a, (c, title, field, k, lab, sc, unit) in zip(ax, CASES):
    ser, emp, t = load(os.path.join(V1D, 'empymod_time', 'cases', c))
    d = np.load(os.path.join(V1D, 'empymod_time', 'cases', c, 'results', 'transients.npz'))
    sel = np.where(d['profile'] == 'x')[0]
    for js, p in enumerate(sel):
        u, v, col = sc * ser[field][:, k, p], sc * emp[field][:, k, p], COL[js % len(COL)]
        m = np.arange(len(t)) % 4 == js % 4
        a.loglog(t, np.where(u > 0, u, np.nan), '-', color=col, lw=1.0, label=f'x = {d["px"][p]:g} m')
        a.loglog(t, np.where(u < 0, -u, np.nan), '--', color=col, lw=1.0)
        a.loglog(t[m & (v > 0)], v[m & (v > 0)], 'o', ms=2.4, mfc='none', mew=0.7, color=col)
        a.loglog(t[m & (v < 0)], -v[m & (v < 0)], 'o', ms=2.4, mew=0.7, color=col)
    a.set_title(f'{title}: {lab}'); a.set_xlabel('t (s)'); a.set_ylabel(f'|{lab}| ({unit})')
ax[0].plot([], [], 'k-', label='series'); ax[0].plot([], [], 'ko', ms=2.4, mfc='none', label='empymod')
fig.tight_layout()
fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=7)
save(fig, 'empymod_time', sys.argv[1], png=False)
