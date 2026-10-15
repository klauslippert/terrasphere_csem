#!/usr/bin/env python3
"""Figure poles: (a) free-space time constants vs permeability; (b) the TE denominator along the negative imaginary w axis
for increasing host conductivity: zeros (poles) only in free space.
Usage: python3 fig_poles.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from style import plt, save, LS_RC, COL
plt.rcParams.update(LS_RC)
from terrasphere_csem.poles import poles_freespace, te_denominator
from terrasphere_csem import MU0
b, ss = 180.0, 1.0
mus = np.linspace(1, 10, 37); fig, ax = plt.subplots(1, 2, figsize=(8.5, 3))
for n in (1, 2, 3):
    for s_ in (1, 2):
        tau = np.array([poles_freespace(n, ss, b, mu, 2)[0][s_ - 1] for mu in mus])
        ax[0].plot(mus, tau * 1e3, '-' if s_ == 1 else '--', color=COL[n - 1], label=f'$n={n}$, $s={s_}$')
ax[0].set_xlabel('$\\mu_r$'); ax[0].set_ylabel('$\\tau_{n,\\nu}$ (ms)'); ax[0].set_title('free space, $b=180$ m, $\\sigma_{\\rm sph}=1$ S/m', fontsize=8); ax[0].legend(fontsize=7, ncol=2)
s_ax = np.geomspace(10, 1e4, 400)
for i, sh in enumerate((1e-9, 1e-4, 1e-3, 1e-2)):
    den = np.array([abs(te_denominator(-1j * s, 1, sh, ss, b)) for s in s_ax])
    ax[1].loglog(s_ax, den, '-', color=COL[i], label=f'$\\sigma_h = {sh:g}$ S/m')
for s_ in (1, 2, 3, 4, 5, 6):
    ax[1].axvline(np.pi ** 2 * s_ ** 2 / (MU0 * ss * b ** 2), color='k', ls=':', lw=0.7)
ax[1].set_xlabel('$s = i\\omega$ (1/s), negative imaginary $\\omega$ axis'); ax[1].set_ylabel('TE denominator, $n=1$'); ax[1].set_title('$|\\mu_r q_r(x) - \\mathsf{m} D_1(\\mathsf{m}x)|$; dotted: free-space $1/\\tau_{1,s}$', fontsize=8); ax[1].legend(fontsize=7)
save(fig, 'poles', sys.argv[1], png=False)
