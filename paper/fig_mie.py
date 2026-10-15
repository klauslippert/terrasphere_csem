#!/usr/bin/env python3
"""Figure mie: Mie T-matrix.  (a) residual of the tangential continuity of E and H at r = b for the unscaled
coefficients (mie_direct), n = 1..30, four cases (mu_r, sigma_sph/sigma_h, omega); (b) magnitudes of the scaled
coefficients (mie_scaled) divided by b^(2n+1) for three frequencies.  Sphere b = 180 m, host 0.01 S/m.
Usage: python3 fig_mie.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from style import plt, save, LS_RC, COL
from terrasphere_csem import MU0
from terrasphere_csem.mie import mie_scaled, mie_direct
from terrasphere_csem.vswf import MN_point

plt.rcParams.update(LS_RC)


def k_of(sig, w, mu=1.0):
    """Wavenumber sqrt(i w mu mu0 sigma) with Im k >= 0 (time dependence e^{-iwt})."""
    k = np.sqrt(1j * w * MU0 * mu * sig + 0j); return k if k.imag >= 0 else -k


b, sh = 180.0, 0.01; th, ph = 0.9, 0.4
rhat = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
tan = lambda v: v - np.dot(v, rhat) * rhat
cases = [(1.0, 1.0, 3e3), (3.0, 1.0, 3e3), (1.0, 0.1, 3e3), (1.0, 1.0, 1e5)]
ns = np.arange(1, 31); res = np.zeros((len(cases), len(ns)))
for ic, (mu_r, ss, w) in enumerate(cases):
    k, ks = k_of(sh, w), k_of(ss, w, mu_r)
    for i_n, n in enumerate(ns):
        an, bn = mie_direct(n, k, ks, b, mu_r); m = 1
        M1, N1 = MN_point(n, m, k, b, th, ph, 1); M3, N3 = MN_point(n, m, k, b, th, ph, 3)
        Ms, Ns = MN_point(n, m, ks, b, th, ph, 1)
        # TE: outside M1 - b_n M3 against d * Ms inside (d from the largest tangential component)
        Eo = M1 - bn * M3; Ho = k * (N1 - bn * N3); d = tan(Eo)[np.argmax(np.abs(tan(Ms)))] / tan(Ms)[np.argmax(np.abs(tan(Ms)))]
        e1 = max(np.linalg.norm(tan(Eo) - d * tan(Ms)) / np.linalg.norm(tan(Eo)),
                 np.linalg.norm(tan(Ho) - ks / mu_r * d * tan(Ns)) / np.linalg.norm(tan(Ho)))
        # TM: outside N1 - a_n N3 against d * Ns inside
        Eo = N1 - an * N3; Ho = k * (M1 - an * M3); d = tan(Eo)[np.argmax(np.abs(tan(Ns)))] / tan(Ns)[np.argmax(np.abs(tan(Ns)))]
        e2 = max(np.linalg.norm(tan(Eo) - d * tan(Ns)) / np.linalg.norm(tan(Eo)),
                 np.linalg.norm(tan(Ho) - ks / mu_r * d * tan(Ms)) / np.linalg.norm(tan(Ho)))
        res[ic, i_n] = max(e1, e2)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
for ic, (mu_r, ss, w) in enumerate(cases):
    ax[0].semilogy(ns, np.maximum(res[ic], 1e-17), 'o-', ms=3, color=COL[ic],
                   label=f'$\\mu_r={mu_r}$, $\\sigma_{{\\rm sph}}/\\sigma_h={ss/sh:.0f}$, $\\omega={w:.0e}$')
ax[0].set_xlabel('$n$'); ax[0].set_ylabel('tangential E, H continuity residual at $r=b$'); ax[0].legend(fontsize=7)
ax[0].set_ylim(1e-17, 1e-10)
for ic, w in enumerate((100.0, 3e3, 1e5)):
    k, ks = k_of(sh, w), k_of(1.0, w); a, bb = mie_scaled(30, k, ks, b)
    ax[1].semilogy(ns, np.abs(bb[1:]) / b ** (2 * ns + 1), '-', color=COL[ic], label=f'$|b_n|/|kb|^{{2n+1}}$, $\\omega={w:.0e}$')
    ax[1].semilogy(ns, np.abs(a[1:]) / b ** (2 * ns + 1), '--', color=COL[ic], label=f'$|a_n|/|kb|^{{2n+1}}$')
ax[1].set_xlabel('$n$'); ax[1].set_ylabel('scaled Mie coefficients'); ax[1].legend(fontsize=7, ncol=2)
save(fig, 'mie', sys.argv[1], png=False)
