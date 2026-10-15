#!/usr/bin/env python3
"""Figure multipole: degree-by-degree content of the coaxial response (C' design, large sphere, and 1:100 design): share of
degrees n >= 2 vs time -> validity of the dipole approximation.
Usage: python3 fig_multipole.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from style import plt, save, LS_RC, COL
plt.rcParams.update(LS_RC)
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver
from terrasphere_csem.timedomain import time_axis, Transient
designs = {"C' (a 400, b 225, d 500, host 10 kOhm m, sphere 0.86 S/m)": (400.0, 500.0, 225.0, 1e-4, 0.8606),
           "1:100 (a 200, b 180, d 250, host 100 Ohm m, sphere 1 S/m)": (200.0, 250.0, 180.0, 0.01, 1.0)}
T = time_axis(3e-4, 1e-1)[::2]; fig, ax = plt.subplots(1, 2, figsize=(8.5, 3)); data = {}
for ic, (lab, (A, H, B, sh, ss)) in enumerate(designs.items()):
    mod = SphereModel(LayerStack([0.0], [sh]), Sphere(B, ss), (0.0, 0.0, H), nmax=12)
    tx = [Dipole('LOOP', (0.0, 0.0, 0.0), 1.0, radius=A)]; rx = [Receiver('LOOP', (0.0, 0.0, 0.0), radius=A)]
    N = mod.N
    def Vdeg(wi):
        loes = mod.solve(wi, tx, rx); v = np.zeros(N, complex)
        if not loes[0]:
            return v
        for n in range(1, N + 1):
            so = {m: np.where(np.isin(np.arange(2 * N), [n - 1, N + n - 1]), s_, 0) for m, (s_, *_r) in loes[0].items()}
            v[n - 1] = mod.scattered_field(wi, tx, rx, solution=loes, s_override=so)[0]
        return v
    vt = Transient(Vdeg, T).switch_off(); tot = vt.sum(1); tau1 = 4e-7 * np.pi * ss * B ** 2 / np.pi ** 2
    ax[ic].semilogx(T / tau1, tot / tot, 'k-', lw=0.8)
    for n in range(1, 6):
        ax[ic].semilogx(T / tau1, vt[:, n - 1] / tot, '-', color=COL[n - 1], label=f'$n={n}$')
    ax[ic].semilogx(T / tau1, 1 - vt[:, 0] / tot, 'k--', label='$n\\geq2$ (missing in a dipole model)')
    ax[ic].set_xlabel('$t/\\tau_1$'); ax[ic].set_ylabel('share of $V_s$'); ax[ic].set_title(lab, fontsize=7); ax[ic].set_ylim(-0.2, 1.1); ax[ic].legend(fontsize=6.5)
    data[f'design{ic}_Vn'] = vt; data[f'design{ic}_tau1'] = tau1; print(lab, 'done')
save(fig, 'multipole', sys.argv[1], png=False)
