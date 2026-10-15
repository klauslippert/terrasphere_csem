#!/usr/bin/env python3
"""Figure derivatives: analytic derivatives vs central finite differences as a function of the step (FD error ~ h^2).
Usage: python3 fig_derivatives.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from style import plt, save, LS_RC, COL
plt.rcParams.update(LS_RC)
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver
st = LayerStack([0.0, 120.0], [0.02, 0.005]); zc = 200.0; kug = dict(b=60.0, sigma_s=2.0, mu_r=1.5); xc, yc = 15.0, -25.0
tx = [Dipole('HED', (150.0, 80.0, 0.0), 1.0, 0.3), Dipole('LOOP', (-100.0, 50.0, 0.0), 1.0, radius=80.0)]
rx = [Receiver('E', (-220.0, 130.0, 0.0), (1, 0, 0)), Receiver('dBdt', (60.0, -90.0, 0.0), (0, 0, 1)), Receiver('LOOP', (10.0, 20.0, 0.0), radius=100.0)]
w = 2e3
def model(**kw):
    p = dict(kug); p.update({k: v for k, v in kw.items() if k in p}); c = (kw.get('xc', xc), kw.get('yc', yc), kw.get('zc', zc))
    return SphereModel(st, Sphere(p['b'], p['sigma_s'], p['mu_r']), c, nmax=10)
mod = model()
params = ['sigma_s', 'b', 'mu_r', 'xc', 'yc', 'zc']
ana = {'sigma_s': mod.derivative(w, tx, rx, 'sigma_s'), 'b': mod.derivative(w, tx, rx, 'b'), 'mu_r': mod.derivative(w, tx, rx, 'mu_r'),
       'xc': mod.derivative_position(w, tx, rx, 'x'), 'yc': mod.derivative_position(w, tx, rx, 'y'), 'zc': mod.derivative_depth(w, tx, rx)}
hs = np.geomspace(1e-5, 1e-1, 9); dev = np.zeros((len(params), len(hs)))
base = dict(kug, xc=xc, yc=yc, zc=zc)
for ip, p in enumerate(params):
    for ih, h in enumerate(hs):
        x0 = base[p]; d = h * (x0 if p in ('sigma_s', 'b', 'mu_r') else 60.0)
        fd = (model(**{p: x0 + d}).scattered_field(w, tx, rx) - model(**{p: x0 - d}).scattered_field(w, tx, rx)) / (2 * d)
        dev[ip, ih] = np.max(np.abs(fd - ana[p]) / np.abs(ana[p]))
    print(p, dev[ip])
fig, ax = plt.subplots(figsize=(5, 3))
labs = {'sigma_s': '$\\partial/\\partial\\sigma_s$', 'b': '$\\partial/\\partial b$', 'mu_r': '$\\partial/\\partial\\mu_r$', 'xc': '$\\partial/\\partial x_c$', 'yc': '$\\partial/\\partial y_c$', 'zc': '$\\partial/\\partial \\zeta_c$ (Richardson FD)'}
for ip, p in enumerate(params):
    ax.loglog(hs, np.maximum(dev[ip], 1e-16), 'o-', ms=3, color=COL[ip], label=labs[p])
ax.loglog(hs, 1e-1 * hs ** 2, 'k:', label='$\\propto h^2$')
ax.set_xlabel('relative finite-difference step $h$'); ax.set_ylabel('|FD $-$ analytic| / |analytic|'); ax.legend(fontsize=7, ncol=2)
save(fig, 'derivatives', sys.argv[1], png=False)
