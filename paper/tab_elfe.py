#!/usr/bin/env python3
"""Table elfe_table: deviations of elfe3D from the series over the 90 m hill (tests/validation/3D/elfe3d_topo), per
source, frequency (0.1 Hz, 10 Hz, 1 kHz) and component: median and 90th percentile of |V_elfe3D - V_series| / max|V_series|
over the receivers of both profiles, in per cent (the measure of compare_elfe3d.py).  A component whose series value
stays below 1e-6 of the largest E (H) component on the profiles vanishes by symmetry and is marked as such.
Writes <output-dir>/elfe_table.tex (rows of a LaTeX tabular).
Usage: python3 tab_elfe.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
V3D = os.path.join(os.path.dirname(HERE), 'tests', 'validation', '3D'); sys.path.insert(0, V3D)
from terrasphere_csem.config import Config
from compare_elfe3d import model_name, COMPONENTS
from report_elfe3d_sphere import elfe_run, ZERO

CASES = [('T1_wire_inline', 'T1'), ('T2_wire_broadside', 'T2'), ('T3_loop_offset', 'T3'), ('T4_loop_large', 'T4')]
H, FREQS = 90.0, (0.1, 10.0, 1000.0)

rows = []
for c, lab in CASES:
    folder = os.path.join(V3D, 'elfe3d_topo', 'cases', c)
    st = Config(folder); d = np.load(os.path.join(folder, 'results', 'fields.npz'))
    ih = list(d['heights']).index(H); P = len(d['px'])
    for f in FREQS:
        jf = int(np.argmin(np.abs(d['freqs'] - f)))
        s = np.concatenate([d['E'][ih, jf], d['H'][ih, jf]], axis=0)        # (6, P)
        e = elfe_run(st, model_name(H), f, P)
        cells = []
        for k, cc in enumerate(COMPONENTS):
            if np.abs(s[k]).max() < ZERO * np.abs(s[3 * (k // 3):3 * (k // 3) + 3]).max():
                cells.append(r'\multicolumn{1}{c}{$\approx0$}'); continue
            dev = np.abs(e[k] - s[k]) / np.abs(s[k]).max()
            cells.append(f'{100 * np.median(dev):.2f}/{100 * np.percentile(dev, 90):.1f}')
        rows.append(f'{lab} & {f:g}\\,Hz & ' + ' & '.join(cells) + r'\\')
os.makedirs(sys.argv[1], exist_ok=True)
open(os.path.join(sys.argv[1], 'elfe_table.tex'), 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
