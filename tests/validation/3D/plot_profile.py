#!/usr/bin/env python3
"""Figure: series (lines) against elfe3D (symbols) on the x- and y-profile of one computation folder, one frequency,
all six components, models flat / hill / depression; real part solid with circles, imaginary part dashed with squares.
Series exactly as written by this package (results/fields.npz); elfe3D converted to its convention (H = -H_elfe3D).
Usage: python3 plot_profile.py <folder> <f in Hz> [out-dir name for the hf band, default out]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.abspath(os.path.join(HERE, '..', '..', '..')), os.path.dirname(HERE), HERE]   # package, plotting.py, this folder
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from terrasphere_csem.config import Config
from compare_elfe3d import read_run, model_name, TO_OURS

COMPONENTS = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
COLOR = {0.0: '0.45', 30.0: '#1f5fa8', -30.0: '#d9731a'}


def main():
    """Plot series against elfe3D for one frequency (see the module docstring)."""
    st = Config(sys.argv[1]); f = float(sys.argv[2]); outname = sys.argv[3] if len(sys.argv) > 3 else 'out'
    d = np.load(os.path.join(st.folder, 'results', 'fields.npz'))
    fr = d['freqs']; jf = int(np.argmin(np.abs(fr - f))); P = len(d['px'])
    ser = np.concatenate([d['E'], d['H']], axis=2)[:, jf]           # (nh, 6, P), as written by terrasphere_csem
    prof = d['profile']; px, py = d['px'], d['py']
    band = 'nf' if f < 100 else 'hf'
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.6), sharex=True)
    for ih, H in enumerate(d['heights']):
        run = os.path.join(st.folder, 'elfe3d', f'{model_name(H)}_{band}', outname if band == 'hf' else 'out')
        el = None
        if os.path.isfile(os.path.join(run, 'magnetic_fields_receiver_line.txt')):
            fe, V = read_run(run, P); i = int(np.argmin(np.abs(fe - f)))
            if abs(fe[i] / f - 1) < 1e-6:
                el = TO_OURS[:, None] * V[i].T                         # (6, P), convention of this package
        for row, (p, pos) in enumerate([('x', px), ('y', py)]):
            sel = prof == p
            for k, c in enumerate(COMPONENTS):
                a = ax[row, k]; sc = 1e9 if k < 3 else 1e6
                lab = 'flat' if H == 0 else (f'hill {H:g} m' if H > 0 else f'depression {-H:g} m')
                a.plot(pos[sel], sc * ser[ih, k, sel].real, '-', color=COLOR[H], lw=1.1, label=f'{lab} (series)')
                a.plot(pos[sel], sc * ser[ih, k, sel].imag, '--', color=COLOR[H], lw=1.1)
                if el is not None:
                    a.plot(pos[sel][::2], sc * el[k, sel][::2].real, 'o', ms=3, mfc='none', color=COLOR[H])
                    a.plot(pos[sel][::2], sc * el[k, sel][::2].imag, 's', ms=3, mfc='none', color=COLOR[H])
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
    os.makedirs(os.path.join(st.folder, 'comparison'), exist_ok=True)
    out = os.path.join(st.folder, 'comparison', f'profile_{f:g}Hz.png')
    fig.savefig(out, dpi=150); fig.savefig(out.replace('.png', '.pdf'))
    print(out)


if __name__ == '__main__':
    main()
