#!/usr/bin/env python3
"""Figure identities: numerical proof of the four spectral identities, max relative error vs degree n for several orders m.
Usage: python3 fig_identities.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
TESTS = os.path.join(os.path.dirname(HERE), 'tests')   # reference solutions of the tests
sys.path += [os.path.join(TESTS, d) for d in ('verification', os.path.join('verification', 'empymod'), 'reference')]
from style import plt, save, LS_RC, COL
plt.rcParams.update(LS_RC)
from scipy.special import jv, lpmv, spherical_jn, spherical_yn, gammaln
from terrasphere_csem.spectral import lam_grid, u_of, W_closed, AB_regular_scaled, outgoing_spectrum_scaled
from terrasphere_csem.legendre import scaled_pnm
from terrasphere_csem.vswf import MN_point
from test_spectral import h1, sph, W_beta_quadrature, PTS, K
ns = range(1, 17); ms = [0, 1, 2, 5]
E = {q: np.full((len(ms), len(ns)), np.nan) for q in ('I', 'II', 'III', 'IV')}
for im, m in enumerate(ms):
    for i_n, n in enumerate(ns):
        if n < m: continue
        # (I)
        worst = 0
        for (rho, phi, z) in PTS:
            r, th, _ = sph(rho, phi, z); lhs = h1(n, K * r) * lpmv(m, n, np.cos(th)) * np.exp(1j * m * phi)
            lam, wl = lam_grid(60.0 / abs(z), 300); u = u_of(lam, K); y = (1j if z > 0 else -1j) * u
            S, _ = scaled_pnm(n, m, y, K, lam)
            rhs = (1j) ** (m - n - 1) / K * np.sum(wl * lam / u * S[n] / K ** n * jv(m, lam * rho) * np.exp(-u * abs(z))) * np.exp(1j * m * phi)
            worst = max(worst, abs(lhs - rhs) / abs(lhs))
        E['I'][im, i_n] = worst
        # (II) truncated at n: use the n-th term's contribution? -> show the full-series error with nmax = 40 (indep. of n); instead show term-wise: skip
        # (IV)
        worst = 0
        for (rho, phi, z) in PTS[:2]:
            r, th, _ = sph(rho, phi, z); M3, N3 = MN_point(n, m, K, r, th, phi, 3)
            lam, wl = lam_grid(60.0 / abs(z), 300); u = u_of(lam, K); kz = (1j if z > 0 else -1j) * u
            aM_TE, aM_TM, aN_TE, aN_TM = outgoing_spectrum_scaled(n, m, kz, K, lam)
            Mi = np.zeros(3, complex); Ni = np.zeros(3, complex)
            for i in range(len(lam)):
                WTE, WTM = W_closed(m, lam[i], K, kz[i], rho, phi, z)
                Mi += wl[i] * (aM_TE[n, i] * WTE[:, 0] + aM_TM[n, i] * WTM[:, 0]) / K ** n
                Ni += wl[i] * (aN_TE[n, i] * WTE[:, 0] + aN_TM[n, i] * WTM[:, 0]) / K ** n
            worst = max(worst, np.linalg.norm(Mi - M3) / np.linalg.norm(M3), np.linalg.norm(Ni - N3) / np.linalg.norm(N3))
        E['IV'][im, i_n] = worst
# (II), (III): error of the series truncated at nmax = n, at a point (convergence with n)
lam0 = 0.9; u0 = u_of(np.array([lam0]), K)[0]; kz0 = 1j * u0
pt = (0.6, 0.7, 0.4); r, th, ph = sph(*pt)
for im, m in enumerate(ms):
    for i_n, nmax in enumerate(ns):
        if nmax < max(m, 1): continue
        S, _ = scaled_pnm(nmax, m, np.array([kz0]), K, np.array([lam0]))
        nn = np.arange(nmax + 1)
        fac = np.where(nn >= m, (2 * nn + 1) * (1j) ** (nn - m) * np.exp(gammaln(np.maximum(nn - m, 0) + 1) - gammaln(np.maximum(nn + m, 0) + 1)), 0)
        lhs = jv(m, lam0 * pt[0]) * np.exp(1j * m * pt[1]) * np.exp(1j * kz0 * pt[2])
        rhs = sum(fac[k] * S[k, 0] / K ** k * spherical_jn(k, K * r) * lpmv(m, k, np.cos(th)) for k in range(m, nmax + 1)) * np.exp(1j * m * ph)
        E['II'][im, i_n] = abs(lhs - rhs) / abs(lhs)
        A_TE, B_TE, A_TM, B_TM = AB_regular_scaled(nmax, m, np.array([kz0]), K, np.array([lam0]))
        ref_TE, ref_TM = W_beta_quadrature(m, lam0, kz0, *pt)
        sTE = np.zeros(3, complex); sTM = np.zeros(3, complex)
        for k in range(max(m, 1), nmax + 1):
            M, N = MN_point(k, m, K, r, th, ph, 1)
            sTE += A_TE[k, 0] / K ** k * M + B_TE[k, 0] / K ** k * N; sTM += A_TM[k, 0] / K ** k * M + B_TM[k, 0] / K ** k * N
        E['III'][im, i_n] = max(np.linalg.norm(sTE - ref_TE) / np.linalg.norm(ref_TE), np.linalg.norm(sTM - ref_TM) / np.linalg.norm(ref_TM))
fig, ax = plt.subplots(1, 4, figsize=(10.5, 2.7))
titles = {'I': '(I) outgoing scalar wave', 'II': '(II) scalar plane wave', 'III': '(III) vector plane wave', 'IV': '(IV) outgoing vector waves'}
for a, q in zip(ax, ('I', 'II', 'III', 'IV')):
    for im, m in enumerate(ms):
        a.semilogy(list(ns), np.maximum(E[q][im], 1e-17), 'o-', ms=3, color=COL[im], label=f'$m={m}$')
    a.set_title(titles[q], fontsize=8); a.set_xlabel('$n$ resp. $n_{max}$'); a.set_ylim(1e-17, 1)
ax[0].set_ylabel('max. relative error'); ax[0].legend()
save(fig, 'identities', sys.argv[1], png=False)
