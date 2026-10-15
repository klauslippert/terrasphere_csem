#!/usr/bin/env python3
"""Degree convergence of the sphere near two interfaces (review R4-02): the case T1_wire_inline of elfe3d_sphere
(sphere b = 180 m, centre 400 m deep, interfaces at 200 and 600 m, 90 m hill) with sphere degrees N = 12, 15, 20,
without the box correction (it does not depend on N).  The change of the sphere anomaly relative to N = 12 is given
per frequency and component, relative to the maximum of the production anomaly (case with sphere minus the same case
of elfe3d_topo without it).
Usage: python3 check_degree.py setup   -> writes N12/, N15/, N20/ with control.toml next to this script
       python3 -m terrasphere_csem N12 (N15, N20; about 15-30 min each)
       python3 check_degree.py eval    -> prints the table (stored in degree.txt)"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, '..', 'cases', 'T1_wire_inline')
TOPO = os.path.join(HERE, '..', '..', 'elfe3d_topo', 'cases', 'T1_wire_inline')
DEGREES = (12, 15, 20)
COMP = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']


def setup():
    """Write N<deg>/control.toml from the case control file with the degree changed and the box correction off."""
    src = open(os.path.join(CASE, 'control.toml')).read()
    for n in DEGREES:
        os.makedirs(os.path.join(HERE, f'N{n}'), exist_ok=True)
        txt = src.replace('degree       = 12', f'degree       = {n}')
        txt = txt.replace('box_correction = true', 'box_correction = false')
        txt = txt.replace('name = "T1_wire_inline"', f'name = "T1_N{n}"')
        open(os.path.join(HERE, f'N{n}', 'control.toml'), 'w').write(txt)


def fields(folder, H=None):
    """(nf, 6, P) of E and H of a results folder, height H (default: the only one)."""
    d = np.load(os.path.join(folder, 'results', 'fields.npz'))
    i = 0 if H is None else list(d['heights']).index(H)
    return np.concatenate([d['E'][i], d['H'][i]], axis=1), d


def evaluate():
    """Print max|V_N - V_12| / max|A| per frequency and component, on the y-profile and on both profiles."""
    prod, d = fields(CASE)
    A = prod - fields(TOPO, 90.0)[0]
    runs = {n: fields(os.path.join(HERE, f'N{n}'))[0] for n in DEGREES}
    for name, sel in (('y-profile', d['profile'] == 'y'), ('both profiles', np.ones(len(d['profile']), bool))):
        print(f'== {name}: max|V_N - V_12| / max|A| for N = 15 / 20 (anomaly max|A| / max|V| in brackets)')
        for i, f in enumerate(d['freqs']):
            row = []
            for c in range(6):
                a = np.abs(A[i, c, sel]).max()
                ch = [np.abs(runs[n][i, c, sel] - runs[12][i, c, sel]).max() / a for n in DEGREES[1:]]
                size = a / np.abs(prod[i, c, sel]).max()
                row.append(f"{COMP[c]} " + '/'.join(f'{x:.1e}' for x in ch) + f" ({size:.3f})")
            print(f'f = {f:g} Hz: ' + '  '.join(row))


if __name__ == '__main__':
    {'setup': setup, 'eval': evaluate}[sys.argv[1]]()
