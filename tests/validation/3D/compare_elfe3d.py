#!/usr/bin/env python3
"""Comparison elfe3D <-> series for one computation folder.  Reads <folder>/results/fields.npz and the
elfe3D outputs <folder>/elfe3d/<model>_<band>/<out>/{electric,magnetic}_fields_receiver_line.txt (rows: frequency,
Re/Im of three components per receiver, frequency-major; receivers in the order of control.toml).
Conventions (elfe3D appendix of the paper): E_elfe3D = E_ours, H_elfe3D = -H_ours (both e^{-iwt}).
Measures per model, frequency and component (relative to the profile maximum over both profiles):
  total field   |V_elfe - V_series| / max|V_series|                         (median, 90th percentile)
Only models with terrain (hill, valley) have elfe3D runs; the flat model is validated against empymod.
Writes <folder>/comparison/elfe3d_<out>.txt and .npz.
Usage: python3 compare_elfe3d.py <folder> [out-dir name, default out]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.abspath(os.path.join(HERE, '..', '..', '..')), os.path.dirname(HERE), HERE]   # package, plotting.py, this folder
import numpy as np
from terrasphere_csem.config import Config

COMPONENTS = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']


def model_name(H):
    """Name of the model of height H [m] used in the elfe3D folder names: 'flat', 'hill<H>' or 'valley<H>'."""
    return 'flat' if H == 0 else (f'hill{abs(H):g}' if H > 0 else f'valley{abs(H):g}')


def is_finished(run):
    """True if the elfe3D log next to the run folder reports a completed run."""
    # elfe3D writes its output files only at the very end; an aborted or running run leaves zeros/partial files
    log = os.path.join(os.path.dirname(run.rstrip('/')) if os.path.basename(run.rstrip('/')) == 'out' else run, 'elfe3d.log')
    return os.path.isfile(log) and 'CPU time to run elfe3D' in open(log, errors='replace').read()


def read_run(run, P):
    """Read the receiver files of an elfe3D run with P receivers: frequencies (nf,) and complex fields (nf, P, 6) = E, H."""
    res = []
    for fn in ('electric_fields_receiver_line.txt', 'magnetic_fields_receiver_line.txt'):
        A = np.atleast_2d(np.loadtxt(os.path.join(run, fn))).reshape(-1, P, 7)
        res.append(A[:, :, 1::2] + 1j * A[:, :, 2::2])
    return A[:, 0, 0], np.concatenate(res, axis=2)                    # f (nf,), V (nf, P, 6)


def main():
    """Compare the elfe3D results with the series results of the folder given on the command line."""
    st = Config(sys.argv[1]); outname = sys.argv[2] if len(sys.argv) > 2 else 'out'
    d = np.load(os.path.join(st.folder, 'results', 'fields.npz'))
    fr = d['freqs']; P = len(d['px'])
    ser = np.concatenate([d['E'], -d['H']], axis=2).transpose(0, 1, 3, 2)   # (nh, nf, P, 6) in elfe3D convention
    heights = list(d['heights'])
    elfe = {}                                                          # (model, f) -> (P, 6)
    for H in heights:
        for band in ('nf', 'hf1k'):                                   # one frequency per refined run
            run = os.path.join(st.folder, 'elfe3d', f'{model_name(H)}_{band}', outname if band != 'nf' else 'out')
            if not os.path.isfile(os.path.join(run, 'magnetic_fields_receiver_line.txt')) or not is_finished(run):
                continue
            fe, V = read_run(run, P)
            for i, f in enumerate(fe):
                if not np.any(V[i]):                                   # elfe3D writes zeros for unfinished frequencies
                    continue
                key = (H, float(f'{f:.6g}'))
                if band != 'nf' or key not in elfe:                     # refined runs win over the nf run
                    elfe[key] = V[i]
    report_lines = [f"# {st.name}: elfe3D ({outname}) vs series ({st.numerics['method']}); median / 90th percentile in per cent"]
    tab = {}
    for f in fr:
        fk = float(f'{f:.6g}'); jf = int(np.argmin(np.abs(fr - f)))
        for ih, H in enumerate(heights):
            if (H, fk) not in elfe:
                continue
            e, s = elfe[(H, fk)], ser[ih, jf]
            row = []
            for k, c in enumerate(COMPONENTS):
                dev = np.abs(e[:, k] - s[:, k]) / np.abs(s[:, k]).max()
                tab[('total', H, fk, c)] = (np.median(dev), np.percentile(dev, 90))
                row.append(f"{100 * np.median(dev):5.2f}/{100 * np.percentile(dev, 90):5.1f}")
            report_lines.append(f"total  {model_name(H):9s} f = {f:8.3g} Hz  " + '  '.join(f'{c} {r}' for c, r in zip(COMPONENTS, row)))
    os.makedirs(os.path.join(st.folder, 'comparison'), exist_ok=True)
    open(os.path.join(st.folder, 'comparison', f'elfe3d_{outname}.txt'), 'w').write('\n'.join(report_lines) + '\n')
    np.savez(os.path.join(st.folder, 'comparison', f'elfe3d_{outname}.npz'),
             keys=np.array([str(k) for k in tab]), values=np.array([list(v) + [np.nan] * (3 - len(v)) for v in tab.values()]))
    print('\n'.join(report_lines))


if __name__ == '__main__':
    main()
