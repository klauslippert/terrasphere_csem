#!/usr/bin/env python3
"""Figure per model with terrain (hill / valley; the flat model has no elfe3D run): series (lines) against elfe3D
(symbols) on the x- and y-profile, all six components; real and imaginary part in separate panels, all frequencies
with an elfe3D run in one panel (colour), physical units (E in nV/m, H in uA/m, per 1 A; no normalisation).
H with the elfe3D sign.
elfe3D results: band nf (out) for f < 100 Hz, band hf (<hf-out>, default out; falls back to the nf run) else.
Usage: python3 plot_profiles_freq.py <folder> [hf-out-dir name]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from terrasphere_csem.config import Config
from compare_elfe3d import read_run, model_name, is_finished

COMPONENTS = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
COLORS = ['#1f5fa8', '#2a9d5c', '#d9731a', '#8e44ad', '#c0392b']


def elfe_field(st, H, f, P, hfout):
    """elfe3D fields (6, P) of model height H at frequency f from the best available run, and its origin; (None, None) if none."""
    # priority: current runs, then earlier ones ('hf' = two-frequency refined run, valid at 100 kHz only;
    # nf/alt_f0.1_1_1000Hz = coarse nf run with 0.1/1/1000 Hz)
    runs = ([('nf', 'out'), ('nf', 'alt_f0.1_1_1000Hz/out')] if f < 100 else
            [('hf1k', hfout), ('hf', 'out'), ('nf', 'out'), ('nf', 'alt_f0.1_1_1000Hz/out')])
    for band, o in runs:
        run = os.path.join(st.folder, 'elfe3d', f'{model_name(H)}_{band}', o)
        if os.path.isfile(os.path.join(run, 'magnetic_fields_receiver_line.txt')) and is_finished(run) and \
                os.path.getsize(os.path.join(run, 'magnetic_fields_receiver_line.txt')) > 0:
            fe, V = read_run(run, P)
            # refined runs: one row per refinement step, only the last one holds the result (others zero)
            ok = [i for i in range(len(fe)) if abs(fe[i] / f - 1) < 1e-6 and np.any(V[i])]
            if ok:
                return V[ok[-1]].T, f'{band}/{o}'
    return None, None


def main():
    """Plot series against elfe3D for all frequencies, one figure per model (see the module docstring)."""
    st = Config(sys.argv[1]); hfout = sys.argv[2] if len(sys.argv) > 2 else 'out'
    d = np.load(os.path.join(st.folder, 'results', 'fields.npz'))
    fr = d['freqs']; P = len(d['px'])
    ser = np.concatenate([d['E'], -d['H']], axis=2)                  # (nh, nf, 6, P), elfe3D convention
    prof = d['profile']; pos = {'x': d['px'], 'y': d['py']}
    out = []
    for ih, H in enumerate(d['heights']):
        if H == 0:                                                     # flat: validated against empymod
            continue
        fig, ax = plt.subplots(4, 6, figsize=(16, 9.5), sharex=True)
        sources = []
        for jf, f in enumerate(fr):
            el, origin = elfe_field(st, H, f, P, hfout)
            if el is None:                                             # only frequencies with an elfe3D run
                continue
            sources.append(f'{f:g} Hz: {origin}')
            for k, c in enumerate(COMPONENTS):
                sc = 1e9 if k < 3 else 1e6
                for ip, p in enumerate(['x', 'y']):
                    sel = prof == p
                    for ir, (part, fn) in enumerate([('Re', np.real), ('Im', np.imag)]):
                        a = ax[2 * ip + ir, k]
                        a.plot(pos[p][sel], sc * fn(ser[ih, jf, k, sel]), '-', color=COLORS[jf], lw=1.1,
                               label=f'{f:g} Hz' if (k == 0 and ip == 0 and ir == 0) else None)
                        a.plot(pos[p][sel][::2], sc * fn(el[k, sel][::2]), 'o', ms=2.8, mfc='none', color=COLORS[jf])
                        a.set_title(f'{part} {c}, {p}-profile', fontsize=8.5)
                        a.tick_params(labelsize=7)
        for a in ax[-1]:
            a.set_xlabel('position (m)', fontsize=8)
        for r in range(4):
            ax[r, 0].set_ylabel('E (nV/m)', fontsize=8); ax[r, 3].set_ylabel(r'H ($\mu$A/m)', fontsize=8)
        ax[0, 0].legend(fontsize=7)
        lab = f'hill {H:g} m' if H > 0 else f'valley {-H:g} m'
        fig.suptitle(f'{st.name}, {lab}: series (lines) vs elfe3D (circles, every 2nd receiver); 1 A source\n'
                     + '; '.join(sources), fontsize=9.5)
        fig.tight_layout()
        os.makedirs(os.path.join(st.folder, 'comparison'), exist_ok=True)
        o = os.path.join(st.folder, 'comparison', f'profile_{model_name(H)}.png')
        fig.savefig(o, dpi=140); fig.savefig(o.replace('.png', '.pdf')); plt.close(fig)
        out.append(o); print(o)


if __name__ == '__main__':
    main()
