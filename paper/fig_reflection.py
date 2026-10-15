#!/usr/bin/env python3
"""Figure reflection: reflection operator (sphere -> layers -> sphere) vs empymod, m = 0 and +-1, relative error vs frequency.
Usage: python3 fig_reflection.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
TESTS = os.path.join(os.path.dirname(HERE), 'tests')   # reference solutions of the tests
sys.path += [os.path.join(TESTS, d) for d in ('verification', os.path.join('verification', 'empymod'), 'reference')]
from style import plt, save, LS_RC, COL
plt.rcParams.update(LS_RC)
from terrasphere_csem import MU0
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sphere import SphereModel, Sphere
from terrasphere_csem.vswf import MN_point
from test_onedim import emp_fields, AIR
from test_reflection import closed_dipole_E
depths = [0.0, 150.0]; sig = [0.02, 0.004]; zc = 260.0; sig_h = sig[1]
st = LayerStack(depths, sig); mod = SphereModel(st, Sphere(30.0, 1.0), (0.0, 0.0, zc), nmax=20)
freqs = np.geomspace(3, 3e3, 7); Rr = 0.4 * mod.h_up
pts = [(Rr * np.sin(t) * np.cos(p), Rr * np.sin(t) * np.sin(p), Rr * np.cos(t)) for t, p in ((0.5, 0.3), (1.3, 2.0), (2.2, 4.0), (2.8, 5.5))]
fitpts = [(60.0, 20.0, 50.0), (-30.0, 70.0, -40.0), (10.0, -50.0, 80.0), (-70.0, -20.0, -30.0)]
lam, wl = mod.grid([], [])
cases = (('VMD', 0.0, (0,), 'M'), ('HMD', 0.4, (-1, 1), 'M'), ('HED', 0.4, (-1, 1), 'N'))
err = np.zeros((len(cases), len(freqs))); ratio = np.zeros_like(err)
for jf, f in enumerate(freqs):
    w = 2 * np.pi * f; k = np.sqrt(1j * w * MU0 * sig_h + 0j); k = k if k.imag >= 0 else -k
    kk, ks, u, R, T = mod._prepare(w, lam)
    for ic, (typ, psi, ms, kind) in enumerate(cases):
        Ec = closed_dipole_E(typ, psi, k, sig_h, w, fitpts); cols = []
        for m in ms:
            col = []
            for (x, y, z) in fitpts:
                r = np.linalg.norm([x, y, z]); th = np.arccos(z / r); ph = np.arctan2(y, x); M3, N3 = MN_point(1, m, k, r, th, ph, 3); col.append(M3 if kind == 'M' else N3)
            cols.append(np.concatenate(col))
        coef = np.linalg.lstsq(np.array(cols).T, Ec.reshape(-1), rcond=None)[0]
        Erefl = np.zeros((len(pts), 3), complex); N = mod.N
        for cm, m in zip(coef, ms):
            ch, Rh, spectra = mod._order(m, w, lam, wl, kk, u, R, []); col = 0 if kind == 'M' else N
            for ip, (x, y, z) in enumerate(pts):
                r = np.linalg.norm([x, y, z]); th = np.arccos(z / r); ph = np.arctan2(y, x)
                for n in range(max(1, abs(m)), N + 1):
                    cM = Rh[n - 1, col] / kk ** n / kk ** 2; cN = Rh[N + n - 1, col] / kk ** n / kk ** 2
                    M1, N1 = MN_point(n, m, k, r, th, ph, 1); Erefl[ip] += cm * (cM * M1 + cN * N1)
        gpts = [(x, y, zc - z) for (x, y, z) in pts]
        E_lay, _ = emp_fields(typ, (0.0, 0.0, zc), gpts, depths, [AIR] + [1 / s_ for s_ in sig], f, psi); E_ful, _ = emp_fields(typ, (0.0, 0.0, zc), gpts, [], [1 / sig_h], f, psi)
        Eemp = E_lay - E_ful
        err[ic, jf] = np.max(np.abs(Erefl - Eemp)) / np.max(np.abs(Eemp)); ratio[ic, jf] = np.max(np.abs(Eemp)) / np.max(np.abs(E_ful))
    print(f"f = {f:.1f} Hz done")
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
labs = ['VMD $\\to M_{1,0}$ (TE, $m=0$)', 'HMD $\\to M_{1,\\pm1}$ (TE$\\to$TE/TM)', 'HED $\\to N_{1,\\pm1}$ (TM$\\to$TM/TE)']
for ic in range(3):
    ax[0].loglog(freqs, err[ic], 'o-', ms=3, color=COL[ic], label=labs[ic]); ax[1].loglog(freqs, ratio[ic], 'o-', ms=3, color=COL[ic])
ax[0].set_xlabel('frequency (Hz)'); ax[0].set_ylabel('max rel. error of reflected field'); ax[0].legend(fontsize=7)
ax[1].set_xlabel('frequency (Hz)'); ax[1].set_ylabel('reflected / direct field'); ax[1].set_title('$|E_{refl}|/|E_{dir}|$ at $r = 0.4\\,h_\\uparrow$', fontsize=8)
save(fig, 'reflection', sys.argv[1], png=False)
