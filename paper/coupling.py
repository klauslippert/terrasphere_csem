#!/usr/bin/env python3
"""Validation of the sphere-terrain coupling: coefficients on the two summit profiles for the cases
'wire_layered' (terrain over two layers) and 'wire_sphere' (the same with the sphere) at a few frequencies,
production grid (N = 512, 10 m, 20 orders), and the low-frequency box check with N = 1024.  From these:
convergence of the coupled series in H, box-size dependence, and coupling versus superposition
(total - terrain only - sphere only + flat).
Usage: python3 coupling.py <out.npz>"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from terrasphere_csem import terrain
import cases as FL

FREQS = [3.0, 30.0, 300.0, 3000.0]


def compute(case, freqs, n, nord):
    """Coefficients (CE, CH, grid x) for the named case at the given frequencies on a grid of n points per side."""
    q = FL.sources(case)
    g = terrain.Grid(q, n=n, dx=10.0, s=FL.S_HILL)
    kg = FL.sphere(case, g)
    CE, CH = [], []
    for f in freqs:
        t0 = time.time()
        ce, ch = terrain.coefficients([f], g, FL.ground(case), nord=nord, sphere=kg)
        CE.append(ce[0]); CH.append(ch[0])
        print(f"{case} N={n} f={f:g} Hz: {time.time() - t0:.0f} s", flush=True)
    return np.array(CE), np.array(CH), g.x


def main():
    """Compute the coupling check coefficients and save them (see the module docstring)."""
    out = sys.argv[1]
    res = dict(freqs=np.array(FREQS))
    for case in ('wire_layered', 'wire_sphere'):
        CE, CH, x = compute(case, FREQS, 512, 20)
        res[f'CE_{case}'] = CE; res[f'CH_{case}'] = CH; res['x'] = x
        CE, CH, x = compute(case, [3.0], 1024, 12)
        res[f'CE1024_{case}'] = CE; res[f'CH1024_{case}'] = CH; res['x1024'] = x
        CE, CH, _ = compute(case, [3.0], 512, 12)
        res[f'CE512_{case}'] = CE; res[f'CH512_{case}'] = CH
        np.savez_compressed(out, **res)
    print("saved", out, flush=True)


if __name__ == '__main__':
    main()
