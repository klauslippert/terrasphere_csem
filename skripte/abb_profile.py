#!/usr/bin/env python3
"""Figure: series (lines) against elfe3D (symbols) on the x- and y-profile of one computation folder, one frequency,
all six components, models flat / hill / valley; real part solid with circles, imaginary part dashed with squares.
elfe3D convention e^{-iwt}; our H is shown with the elfe3D sign (H_elfe3D = -H_ours) so that both lie on top.
Usage: python3 abb_profile.py <folder> <f in Hz> [out-dir name for the hf band, default out]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from terrasphere_csem.steuerung import Steuerung
from vergleich_elfe3d import lies, modellname

KOMP = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
FARBE = {0.0: '0.45', 30.0: '#1f5fa8', -30.0: '#d9731a'}


def main():
    st = Steuerung(sys.argv[1]); f = float(sys.argv[2]); outname = sys.argv[3] if len(sys.argv) > 3 else 'out'
    d = np.load(os.path.join(st.ordner, 'ergebnis', 'felder.npz'))
    fr = d['freqs']; jf = int(np.argmin(np.abs(fr - f))); P = len(d['px'])
    ser = np.concatenate([d['E'], -d['H']], axis=2)[:, jf]          # (nh, 6, P), elfe3D convention
    prof = d['profil']; px, py = d['px'], d['py']
    band = 'nf' if f < 100 else 'hf'
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.6), sharex=True)
    for ih, H in enumerate(d['hoehen']):
        run = os.path.join(st.ordner, 'elfe3d', f'{modellname(H)}_{band}', outname if band == 'hf' else 'out')
        el = None
        if os.path.isfile(os.path.join(run, 'magnetic_fields_receiver_line.txt')):
            fe, V = lies(run, P); i = int(np.argmin(np.abs(fe - f)))
            if abs(fe[i] / f - 1) < 1e-6:
                el = V[i].T                                            # (6, P)
        for row, (p, pos) in enumerate([('x', px), ('y', py)]):
            sel = prof == p
            for k, c in enumerate(KOMP):
                a = ax[row, k]; sc = 1e9 if k < 3 else 1e6
                lab = 'flat' if H == 0 else (f'hill {H:g} m' if H > 0 else f'valley {-H:g} m')
                a.plot(pos[sel], sc * ser[ih, k, sel].real, '-', color=FARBE[H], lw=1.1, label=f'{lab} (series)')
                a.plot(pos[sel], sc * ser[ih, k, sel].imag, '--', color=FARBE[H], lw=1.1)
                if el is not None:
                    a.plot(pos[sel][::2], sc * el[k, sel][::2].real, 'o', ms=3, mfc='none', color=FARBE[H])
                    a.plot(pos[sel][::2], sc * el[k, sel][::2].imag, 's', ms=3, mfc='none', color=FARBE[H])
                a.set_title(f'{c}, {p}-profile', fontsize=9)
                if k == 0 or k == 3:
                    a.set_ylabel('nV/m' if k < 3 else r'$\mu$A/m', fontsize=8)
                if row == 1:
                    a.set_xlabel(f'position (m)', fontsize=8)
                a.tick_params(labelsize=7)
    ax[0, 0].legend(fontsize=6.5, loc='best')
    fig.suptitle(f'{st.name}, f = {f:g} Hz: series (lines; Re solid, Im dashed) vs elfe3D (symbols; Re circles, '
                 f'Im squares, every 2nd receiver)', fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(st.ordner, 'vergleich'), exist_ok=True)
    out = os.path.join(st.ordner, 'vergleich', f'profile_{f:g}Hz.png')
    fig.savefig(out, dpi=150); fig.savefig(out.replace('.png', '.pdf'))
    print(out)


if __name__ == '__main__':
    main()
