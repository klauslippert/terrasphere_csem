#!/usr/bin/env python3
"""Hill series coefficients on the two profiles for a list of frequencies -> npz
(source as a named case).
Usage: python3 coefficients.py <case> <out.npz> <freqs: 'elfe3d' | 'filter' | comma list> [nord] [N] [dx] [kc]
  case: see cases.py ('wire', 'dipole', 'rectangle', 'circle', 'wire_layered', 'wire_sphere')
  'elfe3d': the elfe3D frequencies (0.1 Hz .. 10 kHz, 5 per decade)
  'filter': all abscissae of the Key-201 sine/cosine filter for the lagged time grid 1e-5 .. 10 s (<= TERRASPHERE_FMAX, 2 MHz)
Saves CE, CH (nf, nord+1, 3, 2, N) [e^{-iwt}], x (N), freqs, sigma, s, the numerical parameters and the sources."""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from terrasphere_csem import terrain, timedomain
from terrasphere_csem.sources import hed_list
import cases as FL


def frequencies(arg):
    """Frequencies [Hz] from the command-line argument: 'elfe3d', 'filter' or a comma-separated list."""
    if arg == 'elfe3d':
        return np.array([float(f'{v:.6g}') for v in 10 ** np.arange(-1.0, 4.0001, 0.2)])
    if arg == 'filter':
        f = timedomain.filter_frequencies(timedomain.time_axis(FL.T_MIN, FL.T_MAX))
        return f[f <= float(os.environ.get('TERRASPHERE_FMAX', '2e6'))]      # above: series does not converge
    return np.array([float(v) for v in arg.split(',')])


def main():
    """Compute and save the series coefficients of one case (see the module docstring)."""
    case, out, fr = sys.argv[1], sys.argv[2], frequencies(sys.argv[3])
    nord = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 512
    dx = float(sys.argv[6]) if len(sys.argv) > 6 else 10.0
    kc = float(sys.argv[7]) if len(sys.argv) > 7 else 0.2
    q = FL.sources(case)
    g = terrain.Grid(q, n=n, dx=dx, s=FL.S_HILL)
    bo = FL.ground(case); kg = FL.sphere(case, g)
    t0 = time.time()
    CE = np.zeros((len(fr), nord + 1, 3, 2, n), complex); CH = np.zeros_like(CE)
    blk = 20
    for i0 in range(0, len(fr), blk):
        fb = fr[i0:i0 + blk]
        ce, ch = terrain.coefficients(fb, g, bo, nord=nord, kc=kc, sphere=kg)
        CE[i0:i0 + len(fb)] = ce; CH[i0:i0 + len(fb)] = ch
        print(f"{i0 + len(fb)}/{len(fr)} frequencies, {time.time() - t0:.0f} s", flush=True)
    src = np.array([[d.pos[0], d.pos[1], d.moment, d.psi] for d in hed_list(q)])
    np.savez_compressed(out, CE=CE, CH=CH, x=g.x, freqs=fr, sigma=FL.SIGMA, s=FL.S_HILL, nord=nord, n=n, dx=dx,
                        kc=kc, case=case, sources=src, time_convention='exp(-iwt)')
    print("saved", out, flush=True)


if __name__ == '__main__':
    main()
