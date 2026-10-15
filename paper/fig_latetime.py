#!/usr/bin/env python3
"""Figure latetime: Laplace (Talbot) route vs sine-filter route for the primary field (Rx 0.7 a) and the sphere part.
Usage: python3 fig_latetime.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
TESTS = os.path.join(os.path.dirname(HERE), 'tests')   # reference solutions of the tests
sys.path += [os.path.join(TESTS, d) for d in ('verification', os.path.join('verification', 'empymod'), 'reference')]
from style import plt, save, LS_RC, COL
plt.rcParams.update(LS_RC)
from scipy.special import ellipk, ellipe, jv
import empymod_loop as ES
from singh_multipole import step_off, k_of
from terrasphere_csem import MU0
AIR = 2e14   # resistivity of the air for empymod (Ohm m)
from terrasphere_csem.spectral import lam_grid
from terrasphere_csem.timedomain import talbot, switch_off_talbot, model_transient
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver
A, sh = 200.0, 0.01; Ar = 0.7 * A
kk2 = 4 * A * Ar / (A + Ar) ** 2; kk = np.sqrt(kk2); M_free = MU0 * np.sqrt(A * Ar) * ((2 / kk - kk) * ellipk(kk2) - (2 / kk) * ellipe(kk2))
lam, wl = lam_grid(400 * np.pi / A, 400, 24); JJ = jv(1, lam * A) * jv(1, lam * Ar)
def V_delta(w):
    k = k_of(sh, w); u = np.sqrt(lam ** 2 - k * k + 0j); u = np.where(np.real(u) < 0, -u, u)
    return 2 * np.pi * Ar * 1j * w * MU0 * A * np.sum(wl * (lam / (lam + u) - 0.5) * JJ)
T = np.geomspace(1e-3, 1, 13)
tal = {M: talbot(lambda s: -V_delta(1j * s) / s + M_free, T, M=M) for M in (16, 24, 32)}
filt = step_off(lambda w: np.array([V_delta(wi) for wi in w]), T)
emp = ES.V_off_time(A, T, depth=[0], res=[AIR, 1 / sh], a_rx=Ar)
mod = SphereModel(LayerStack([0.0], [sh]), Sphere(180.0, 1.0), (0.0, 0.0, 250.0), nmax=15)
tx = [Dipole('LOOP', (0.0, 0.0, 0.0), 1.0, radius=A)]; rx = [Receiver('LOOP', (0.0, 0.0, 0.0), radius=A)]
Ts = np.geomspace(1e-3, 1e-1, 7)
tal_s = {M: switch_off_talbot(lambda w: mod.scattered_field(w, tx, rx), Ts, M=M)[:, 0] for M in (16, 24, 32)}
filt_s = model_transient(mod, tx, rx, Ts).switch_off()[:, 0]          # exact filter evaluation (KSA-025)
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3))
for i, M in enumerate((16, 24, 32)):
    ax[0].semilogx(T, tal[M] / filt - 1, 'o-', ms=3, color=COL[i], label=f'Talbot $M={M}$ / filter $-1$')
    ax[1].semilogx(Ts, tal_s[M] / filt_s - 1, 'o-', ms=3, color=COL[i], label=f'Talbot $M={M}$ / filter $-1$')
ax[0].semilogx(T, emp / filt - 1, 's--', ms=3, color=COL[3], label='empymod / filter $-1$')
ax[0].set_title('primary half-space voltage, Rx radius 0.7 a', fontsize=8); ax[0].set_xlabel('time (s)'); ax[0].set_ylabel('relative deviation'); ax[0].set_yscale('symlog', linthresh=1e-6); ax[0].legend(fontsize=6.5)
ax[1].set_title('sphere part, coaxial loop (1:100 design)', fontsize=8); ax[1].set_xlabel('time (s)'); ax[1].set_yscale('symlog', linthresh=1e-6); ax[1].legend(fontsize=6.5)
ax2 = ax[1].twinx(); ax2.loglog(Ts, np.abs(filt_s), 'k:', lw=0.8); ax2.set_ylabel('$|V_s|$ (V/A), dotted'); ax2.grid(False)
save(fig, 'latetime', sys.argv[1], png=False)
