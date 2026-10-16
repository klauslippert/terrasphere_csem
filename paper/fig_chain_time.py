#!/usr/bin/env python3
"""Figure chain_time: verification chain in the time domain (1:100 design, coaxial loop): terrasphere_csem vs the single-order references of
tests/reference (half-space: lee1981_coaxial, full space: singh_multipole), Nabighian 1970 (free space), and Lee 1983 (dipole approximation).
Usage: python3 fig_chain_time.py <output-dir>"""
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
from terrasphere_csem.timedomain import model_transient, time_axis
import lee1981_coaxial as L2, singh_multipole as S1
from nabighian_multipole import Vs_multipole
from lee1983_reference import V_s as lee_Vs
A, H, B, sh, ss = 200.0, 250.0, 180.0, 0.01, 1.0
T = time_axis(2e-4, 5e-2)[::2]          # lagged grid: exact filter evaluation (KSA-025)
tx = [Dipole('LOOP', (0.0, 0.0, 0.0), 1.0, radius=A)]; rx = [Receiver('LOOP', (0.0, 0.0, 0.0), radius=A)]
out = {}
for name, st in (('halfspace', LayerStack([0.0], [sh])), ('fullspace', LayerStack([0.0], [sh], sigma_air=sh)), ('freespace', LayerStack([0.0], [1e-9], sigma_air=1e-9))):
    mod = SphereModel(st, Sphere(B, ss), (0.0, 0.0, H), nmax=15)
    out[name] = model_transient(mod, tx, rx, T).switch_off()[:, 0]; print(name, 'done')
ref = {'halfspace': L2.Vs_time(T, A, H, B, sh, ss, nmax=15), 'fullspace': S1.Vs_time(T, A, H, B, sh, ss, nmax=15), 'freespace': Vs_multipole(T, A, H, B, ss, nmax=30)}
lee = np.array([lee_Vs(A, B, H, 0.0, 1.0, sh, ss, t)['total'] for t in T])
tau1 = 4e-7 * np.pi * ss * B ** 2 / np.pi ** 2
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3))
for i, (name, lab) in enumerate((('halfspace', 'half-space (reference: Lee 1981 physics, all $n$)'), ('fullspace', 'full space (reference: Singh 1973)'), ('freespace', 'free space (Nabighian 1970)'))):
    ax[0].loglog(T / tau1, np.abs(out[name]), '-', color=COL[i], label='terrasphere_csem, ' + lab.split(' (')[0]); ax[0].loglog(T / tau1, np.abs(ref[name]), 'o', ms=3, mfc='none', color=COL[i], label=lab.split(': ')[-1].rstrip(')') if ': ' in lab else lab.split(' (')[1].rstrip(')'))
    ax[1].semilogx(T / tau1, out[name] / ref[name], 'o-', ms=3, color=COL[i], label=lab.split(' (')[0])
ax[0].loglog(T / tau1, np.abs(lee), 'k--', label='Lee 1983 ($n=1$, static kernels)')
ax[1].semilogx(T / tau1, out['halfspace'] / lee, 'k--', label='half-space / Lee 1983')
ax[0].set_xlabel('$t/\\tau_1$'); ax[0].set_ylabel('$|V_s|$ (V/A), switch-off'); ax[0].legend(fontsize=6.5)
t_lee = 5 * sh * 4e-7 * np.pi * H ** 2
ax[1].axvline(t_lee / tau1, color='k', ls=':', lw=0.8); ax[1].text(t_lee / tau1 * 1.05, 2.5, 'Lee 1983 valid $\\rightarrow$', fontsize=6.5)
ax[1].set_yscale('log'); ax[1].set_ylim(0.3, 4); ax[1].set_xlabel('$t/\\tau_1$'); ax[1].set_ylabel('terrasphere_csem / reference'); ax[1].legend(fontsize=7, loc='upper right')
save(fig, 'chain_time', sys.argv[1], png=False)
