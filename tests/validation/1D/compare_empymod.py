#!/usr/bin/env python3
"""Flat model: series of this package (<folder>/results/fields.npz, height 0) against empymod
(<folder>/empymod/out/fields.npz, tests/validation/1D/run_empymod.py).  Per frequency and component |V_series - V_empymod| /
max|V_empymod| (maximum over both profiles): median, 90th percentile and maximum, in
<folder>/comparison/empymod_out.txt.  E_z vanishes on the flat surface in both (series: exactly; empymod 1 mm below
the surface: < 1e-4 of max|E|); it is listed as max|E_z| / max|E|.
Usage: python3 compare_empymod.py <folder>"""
import os, sys
import numpy as np

COMPONENTS = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']


def load(folder):
    """Series (nf, 6, P) of height 0, empymod (nf, 6, P), frequencies and the series file."""
    s = np.load(os.path.join(folder, 'results', 'fields.npz'))
    e = np.load(os.path.join(folder, 'empymod', 'out', 'fields.npz'))
    ih = list(s['heights']).index(0.0)
    assert np.allclose(s['px'], e['px']) and np.allclose(s['py'], e['py']) and np.allclose(s['freqs'], e['freqs'])
    return (np.concatenate([s['E'][ih], s['H'][ih]], axis=1), np.concatenate([e['E'], e['H']], axis=1),
            s['freqs'], s)


def deviations(ser, emp):
    """{(f index, component): (median, 90th percentile, max)} of |ser - emp| / max|emp|; E_z: (max|E_z| / max|E|,)."""
    out = {}
    for i in range(ser.shape[0]):
        for k, c in enumerate(COMPONENTS):
            if c == 'Ez':
                out[(i, c)] = (np.abs(emp[i, 2]).max() / np.abs(emp[i, :3]).max(),)
                continue
            dev = np.abs(ser[i, k] - emp[i, k]) / np.abs(emp[i, k]).max()
            out[(i, c)] = (np.median(dev), np.percentile(dev, 90), dev.max())
    return out


def main():
    """Write the comparison table of the folder (see the module docstring)."""
    folder = os.path.abspath(sys.argv[1])
    ser, emp, fr, _ = load(folder)
    dv = deviations(ser, emp)
    lines = [f'# {os.path.basename(folder)}: series (height 0) vs empymod; |dV| / max|V_empymod| in per cent: '
             'median / 90th percentile / max (Ez: max|Ez_empymod| / max|E|)']
    for i, f in enumerate(fr):
        row = []
        for c in COMPONENTS:
            v = dv[(i, c)]
            row.append(f'{c} {100 * v[0]:.1e}' if c == 'Ez' else f'{c} {100 * v[0]:.1e}/{100 * v[1]:.1e}/{100 * v[2]:.1e}')
        lines.append(f'f = {f:8g} Hz  ' + '  '.join(row))
    os.makedirs(os.path.join(folder, 'comparison'), exist_ok=True)
    open(os.path.join(folder, 'comparison', 'empymod_out.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
