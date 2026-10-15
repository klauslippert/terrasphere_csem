#!/usr/bin/env python3
"""Flat model in the time domain: series of this package (<folder>/results/transients.npz, height 0) against empymod
(<folder>/empymod/out/transients.npz, tests/validation/1D/run_empymod_time.py), components E_x, E_y, dB/dt x, y, z
(E_z vanishes on the earth side of the flat surface and is not compared).  Two measures per component, each over all
receivers of both profiles:
  curve:     max_t |V_series - V_empymod| / max_t |V_empymod| per receiver (relative to the maximum of its curve),
             median and maximum over the receivers, and its maximum per decade of time;
  pointwise: max |V_series - V_empymod| / |V_empymod| over the times where |V_empymod| > THRESHOLD max_t |V_empymod|.
A receiver where a component vanishes by symmetry (max_t |V_empymod| below ZERO of the largest E resp. dB/dt component
over all receivers) is left out of both measures; for it the larger of max_t |V| of the two codes is given, relative
to that same reference ('zero': largest such value, '-' if there is none).
Written to <folder>/comparison/empymod_out.txt.
Usage: python3 compare_empymod_time.py <folder>"""
import os, sys
import numpy as np

COMPONENTS = [('E', 0, 'Ex'), ('E', 1, 'Ey'), ('dBdt', 0, 'dBx/dt'), ('dBdt', 1, 'dBy/dt'), ('dBdt', 2, 'dBz/dt')]
THRESHOLD = 1e-6
ZERO = 1e-6


def load(folder):
    """Series and empymod transients {field: (nt, 3, P)} of height 0, times."""
    s = np.load(os.path.join(folder, 'results', 'transients.npz'))
    e = np.load(os.path.join(folder, 'empymod', 'out', 'transients.npz'))
    ih = list(s['heights']).index(0.0)
    assert np.allclose(s['px'], e['px']) and np.allclose(s['py'], e['py']) and np.allclose(s['t'], e['t'], rtol=1e-9)
    assert str(s['signal']) == str(e['signal'])
    return {f: s[f][ih] for f in ('E', 'dBdt')}, {f: e[f] for f in ('E', 'dBdt')}, s['t']


def deviations(ser, emp, t):
    """{component: dict(curve, pointwise (receivers with a non-vanishing component), decades {decade: max curve
    measure}, zero (level of the vanishing receivers or None))}."""
    out = {}
    dec = np.floor(np.log10(t) + 1e-9).astype(int)
    for field, k, name in COMPONENTS:
        ref = np.abs(emp[field][:, :2 if field == 'E' else 3]).max()
        a, b = ser[field][:, k], emp[field][:, k]                    # (nt, P)
        mx = np.abs(b).max(0)
        z = mx < ZERO * ref
        zero = max(mx[z].max(), np.abs(a[:, z]).max()) / ref if z.any() else None
        a, b, mx = a[:, ~z], b[:, ~z], mx[~z]
        d = np.abs(a - b) / mx
        ok = np.abs(b) > THRESHOLD * mx
        pw = np.where(ok, np.abs(a - b) / np.where(ok, np.abs(b), 1.0), 0.0).max(0)
        out[name] = dict(curve=d.max(0), pointwise=pw, decades={int(q): d[dec == q].max() for q in np.unique(dec)},
                         n=int((~z).sum()), zero=zero)
    return out


def main():
    """Write the comparison table of the folder (see the module docstring)."""
    folder = os.path.abspath(sys.argv[1])
    ser, emp, t = load(folder)
    dv = deviations(ser, emp, t)
    decs = sorted(next(iter(dv.values()))['decades'])
    lines = [f'# {os.path.basename(folder)}: series (height 0) vs empymod, time domain {t[0]:.3g}..{t[-1]:.3g} s; '
             f'curve: max_t |dV| / max_t |V_empymod| per receiver (median / max over receivers); pointwise: '
             f'max |dV| / |V_empymod| where |V_empymod| > {THRESHOLD:g} of its maximum (max over receivers); n: receivers '
             f'compared; zero: level of the receivers where the component vanishes by symmetry (< {ZERO:g} of the largest '
             f'E resp. dB/dt)',
             f'{"component":10s} {"n":>3s} {"curve med":>10s} {"curve max":>10s} {"pointwise":>10s} {"zero":>8s}   '
             f'curve max per decade: ' +
             ' '.join(f'1e{q:d}s' for q in decs)]
    for name, v in dv.items():
        zs = '-' if v['zero'] is None else f'{v["zero"]:.0e}'
        lines.append(f'{name:10s} {v["n"]:3d} {np.median(v["curve"]):10.1e} {v["curve"].max():10.1e} '
                     f'{v["pointwise"].max():10.1e} {zs:>8s}   ' +
                     ' '.join(f'{v["decades"][q]:7.1e}' for q in decs))
    os.makedirs(os.path.join(folder, 'comparison'), exist_ok=True)
    open(os.path.join(folder, 'comparison', 'empymod_out.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
