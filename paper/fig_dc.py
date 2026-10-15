#!/usr/bin/env python3
"""Figure dc: direct-current limit of the grounded dipole: secondary field along a profile vs the point-source series.
Usage: python3 fig_dc.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
TESTS = os.path.join(os.path.dirname(HERE), 'tests')   # reference solutions of the tests
sys.path += [os.path.join(TESTS, d) for d in ('verification', os.path.join('verification', 'empymod'), 'reference')]
from style import plt, save, LS_RC, COL
plt.rcParams.update(LS_RC)
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver
from test_dc import E_s_dc
sig_h, sig_s, b, zc = 0.01, 0.5, 50.0, 300.0
st = LayerStack([0.0], [sig_h], sigma_air=sig_h); mod = SphereModel(st, Sphere(b, sig_s), (0.0, 0.0, zc), nmax=12)
src = (120.0, -60.0, 100.0); psi = 0.8; w = 1e-3
xs = np.linspace(-300, 300, 21); zt = 420.0
ours = np.zeros((len(xs), 3), complex); ref = np.zeros((len(xs), 3))
for i, x in enumerate(xs):
    ours[i] = [mod.scattered_field(w, [Dipole('HED', src, 1.0, psi)], [Receiver('E', (x, 40.0, zt), e)])[0] for e in np.eye(3)]
    ref[i] = E_s_dc((x, 40.0, zc - zt), (src[0], src[1], zc - src[2]), psi, b, sig_s, sig_h)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
for j, nm in enumerate(('$E_x$', '$E_y$', '$E_z$')):
    ax[0].plot(xs, np.real(ours[:, j]) * 1e9, '-', color=COL[j], label=f'{nm} terrasphere_csem, $\\omega=10^{{-3}}$ s$^{{-1}}$'); ax[0].plot(xs, ref[:, j] * 1e9, 'o', ms=3, mfc='none', color=COL[j])
ax[0].set_xlabel('$x$ (m), profile at $y = 40$ m, 120 m below the centre'); ax[0].set_ylabel('secondary $E$ (nV/m per A m)'); ax[0].legend(fontsize=7)
ax[1].semilogy(xs, np.max(np.abs(ours - ref), axis=1) / np.max(np.abs(ref)), 'o-', ms=3, color=COL[0], label='|terrasphere_csem $-$ DC series| / max')
ax[1].semilogy(xs, np.max(np.abs(np.imag(ours)), axis=1) / np.max(np.abs(ref)), 's--', ms=3, color=COL[1], label='|Im| / max (residual induction)')
ax[1].set_xlabel('$x$ (m)'); ax[1].legend(fontsize=7)
save(fig, 'dc', sys.argv[1], png=False)
