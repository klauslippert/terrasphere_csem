#!/usr/bin/env python3
"""Comparison elfe3D <-> series for one computation folder (KH-031).  Reads <folder>/ergebnis/felder.npz and the
elfe3D outputs <folder>/elfe3d/<model>_<band>/<out>/{electric,magnetic}_fields_receiver_line.txt (rows: frequency,
Re/Im of three components per receiver, frequency-major; receivers in the order of steuer.toml).
Conventions (Appendix elfe3D of the paper): E_elfe3D = E_ours, H_elfe3D = -H_ours (both e^{-iwt}).
Measures per model, frequency and component (relative to the profile maximum over both profiles):
  total field   |V_elfe - V_series| / max|V_series|                         (median, 90th percentile)
  terrain effect (model - flat) of both codes, |dV_elfe - dV_series| / max|dV_series|; size max|dV|/max|V|
E_z of the flat model is zero in the series (flat surface, earth side) and is skipped there.
Writes <folder>/vergleich/elfe3d_<out>.txt and .npz.
Usage: python3 vergleich_elfe3d.py <folder> [out-dir name, default out]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from terrasphere_csem.steuerung import Steuerung

KOMP = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']


def modellname(H):
    return 'flach' if H == 0 else (f'huegel{abs(H):g}' if H > 0 else f'tal{abs(H):g}')


def fertig(run):
    # elfe3D writes its output files only at the very end; an aborted or running run leaves zeros/partial files (KH-037)
    log = os.path.join(os.path.dirname(run.rstrip('/')) if os.path.basename(run.rstrip('/')) == 'out' else run, 'elfe3d.log')
    return os.path.isfile(log) and 'CPU time to run elfe3D' in open(log, errors='replace').read()


def lies(run, P):
    res = []
    for fn in ('electric_fields_receiver_line.txt', 'magnetic_fields_receiver_line.txt'):
        A = np.atleast_2d(np.loadtxt(os.path.join(run, fn))).reshape(-1, P, 7)
        res.append(A[:, :, 1::2] + 1j * A[:, :, 2::2])
    return A[:, 0, 0], np.concatenate(res, axis=2)                    # f (nf,), V (nf, P, 6)


def main():
    st = Steuerung(sys.argv[1]); outname = sys.argv[2] if len(sys.argv) > 2 else 'out'
    d = np.load(os.path.join(st.ordner, 'ergebnis', 'felder.npz'))
    fr = d['freqs']; P = len(d['px'])
    ser = np.concatenate([d['E'], -d['H']], axis=2).transpose(0, 1, 3, 2)   # (nh, nf, P, 6) in elfe3D convention
    hoehen = list(d['hoehen'])
    elfe = {}                                                          # (model, f) -> (P, 6)
    for H in hoehen:
        for band in ('nf', 'hf1k', 'hf100k'):                         # one frequency per refined run (KH-032)
            run = os.path.join(st.ordner, 'elfe3d', f'{modellname(H)}_{band}', outname if band != 'nf' else 'out')
            if not os.path.isfile(os.path.join(run, 'magnetic_fields_receiver_line.txt')) or not fertig(run):
                continue
            fe, V = lies(run, P)
            for i, f in enumerate(fe):
                if not np.any(V[i]):                                   # elfe3D writes zeros for unfinished frequencies
                    continue
                key = (H, float(f'{f:.6g}'))
                if band != 'nf' or key not in elfe:                     # refined runs win over the nf run
                    elfe[key] = V[i]
    zeilen = [f"# {st.name}: elfe3D ({outname}) vs series ({st.num['verfahren']}); median / 90th percentile in per cent"]
    tab = {}
    for f in fr:
        fk = float(f'{f:.6g}'); jf = int(np.argmin(np.abs(fr - f)))
        for ih, H in enumerate(hoehen):
            if (H, fk) not in elfe:
                continue
            e, s = elfe[(H, fk)], ser[ih, jf]
            row = []
            for k, c in enumerate(KOMP):
                if H == 0 and c == 'Ez':
                    row.append('   -  '); continue
                dev = np.abs(e[:, k] - s[:, k]) / np.abs(s[:, k]).max()
                tab[('total', H, fk, c)] = (np.median(dev), np.percentile(dev, 90))
                row.append(f"{100 * np.median(dev):5.2f}/{100 * np.percentile(dev, 90):5.1f}")
            zeilen.append(f"total  {modellname(H):9s} f = {f:8.3g} Hz  " + '  '.join(f'{c} {r}' for c, r in zip(KOMP, row)))
            if H != 0 and (0.0, fk) in elfe:
                de = e - elfe[(0.0, fk)]; ds = s - ser[hoehen.index(0.0), jf]
                row = []
                for k, c in enumerate(KOMP):
                    ref = np.abs(ds[:, k]).max(); size = ref / max(np.abs(s[:, k]).max(), 1e-300)
                    dev = np.abs(de[:, k] - ds[:, k]) / ref
                    tab[('effekt', H, fk, c)] = (np.median(dev), np.percentile(dev, 90), size)
                    row.append(f"{100 * np.median(dev):5.1f}/{100 * np.percentile(dev, 90):5.0f} ({100 * size:.1f}%)")
                zeilen.append(f"effect {modellname(H):9s} f = {f:8.3g} Hz  " + '  '.join(f'{c} {r}' for c, r in zip(KOMP, row)))
    os.makedirs(os.path.join(st.ordner, 'vergleich'), exist_ok=True)
    open(os.path.join(st.ordner, 'vergleich', f'elfe3d_{outname}.txt'), 'w').write('\n'.join(zeilen) + '\n')
    np.savez(os.path.join(st.ordner, 'vergleich', f'elfe3d_{outname}.npz'),
             schluessel=np.array([str(k) for k in tab]), werte=np.array([list(v) + [np.nan] * (3 - len(v)) for v in tab.values()]))
    print('\n'.join(zeilen))


if __name__ == '__main__':
    main()
