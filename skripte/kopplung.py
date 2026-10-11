#!/usr/bin/env python3
"""Validation of the sphere-terrain coupling (KH-014): coefficients on the two summit profiles for the cases
'draht_schichten' (terrain over two layers) and 'draht_kugel' (the same with the sphere) at a few frequencies,
production grid (N = 512, 10 m, 20 orders), and the low-frequency box check with N = 1024.  From these:
convergence of the coupled series in H, box-size dependence, and coupling versus superposition
(total - terrain only - sphere only + flat).
Usage: python3 kopplung.py <out.npz>"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from terrasphere_csem import huegel
import faelle as FL

FREQS = [3.0, 30.0, 300.0, 3000.0]


def rechne(fall, freqs, n, nord):
    q = FL.quellen(fall)
    g = huegel.Gitter(q, n=n, dx=10.0, s=FL.S_HILL)
    kg = FL.kugel(fall, g)
    CE, CH = [], []
    for f in freqs:
        t0 = time.time()
        ce, ch = huegel.koeffizienten([f], g, FL.boden(fall), nord=nord, kugel=kg)
        CE.append(ce[0]); CH.append(ch[0])
        print(f"{fall} N={n} f={f:g} Hz: {time.time() - t0:.0f} s", flush=True)
    return np.array(CE), np.array(CH), g.x


def main():
    out = sys.argv[1]
    res = dict(freqs=np.array(FREQS))
    for fall in ('draht_schichten', 'draht_kugel'):
        CE, CH, x = rechne(fall, FREQS, 512, 20)
        res[f'CE_{fall}'] = CE; res[f'CH_{fall}'] = CH; res['x'] = x
        CE, CH, x = rechne(fall, [3.0], 1024, 12)
        res[f'CE1024_{fall}'] = CE; res[f'CH1024_{fall}'] = CH; res['x1024'] = x
        CE, CH, _ = rechne(fall, [3.0], 512, 12)
        res[f'CE512_{fall}'] = CE; res[f'CH512_{fall}'] = CH
        np.savez_compressed(out, **res)
    print("saved", out, flush=True)


if __name__ == '__main__':
    main()
