#!/usr/bin/env python3
"""Hill series coefficients on the two profiles for a list of frequencies -> npz (from gausshill
rechne_frequenzen.py, GH-014; source as a named case, KH-004).
Usage: python3 koeffizienten.py <fall> <out.npz> <freqs: 'elfe3d' | 'filter' | comma list> [nord] [N] [dx] [kc]
  fall: see faelle.py ('draht', 'dipol', 'rechteck', 'kreis', 'draht_schichten', 'draht_kugel')
  'elfe3d': the elfe3D frequencies (0.1 Hz .. 10 kHz, 5 per decade)
  'filter': all abscissae of the Key-201 sine/cosine filter for the lagged time grid 1e-5 .. 10 s (<= KH_FMAX, 2 MHz)
Saves CE, CH (nf, nord+1, 3, 2, N) [e^{-iwt}], x (N), freqs, sigma, s, the numerical parameters and the sources."""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from terrasphere_csem import huegel, zeit
from terrasphere_csem.quellen import hed_liste
import faelle as FL


def frequenzen(arg):
    if arg == 'elfe3d':
        return np.array([float(f'{v:.6g}') for v in 10 ** np.arange(-1.0, 4.0001, 0.2)])
    if arg == 'filter':
        f = zeit.filter_frequenzen(zeit.zeitachse(FL.T_MIN, FL.T_MAX))
        return f[f <= float(os.environ.get('KH_FMAX', '2e6'))]      # above: series does not converge (GH-015)
    return np.array([float(v) for v in arg.split(',')])


def main():
    fall, out, fr = sys.argv[1], sys.argv[2], frequenzen(sys.argv[3])
    nord = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 512
    dx = float(sys.argv[6]) if len(sys.argv) > 6 else 10.0
    kc = float(sys.argv[7]) if len(sys.argv) > 7 else 0.2
    q = FL.quellen(fall)
    g = huegel.Gitter(q, n=n, dx=dx, s=FL.S_HILL)
    bo = FL.boden(fall); kg = FL.kugel(fall, g)
    t0 = time.time()
    CE = np.zeros((len(fr), nord + 1, 3, 2, n), complex); CH = np.zeros_like(CE)
    blk = 20
    for i0 in range(0, len(fr), blk):
        fb = fr[i0:i0 + blk]
        ce, ch = huegel.koeffizienten(fb, g, bo, nord=nord, kc=kc, kugel=kg)
        CE[i0:i0 + len(fb)] = ce; CH[i0:i0 + len(fb)] = ch
        print(f"{i0 + len(fb)}/{len(fr)} frequencies, {time.time() - t0:.0f} s", flush=True)
    src = np.array([[d.pos[0], d.pos[1], d.moment, d.psi] for d in hed_liste(q)])
    np.savez_compressed(out, CE=CE, CH=CH, x=g.x, freqs=fr, sigma=FL.SIGMA, s=FL.S_HILL, nord=nord, n=n, dx=dx,
                        kc=kc, fall=fall, quellen=src, zeitkonvention='exp(-iwt)')
    print("saved", out, flush=True)


if __name__ == '__main__':
    main()
