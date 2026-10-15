#!/usr/bin/env python3
"""Figure convergence: series terms |c_n H^n| / |c_0| versus order n on the summit, H/s = 0.2, 0.6, 1.0, three
frequencies; E_x and H_z.  Grounded wire over the half-space 0.01 S/m (cases.py 'wire_halfspace'), 24 orders,
512^2 x 10 m.  The coefficients near the summit are written to <output-dir>/convergence.npz and reused if present.
Usage: python3 fig_convergence.py <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from style import plt, save, GH_RC, GH_C as C
from terrasphere_csem import terrain
import cases as FL

plt.rcParams.update(GH_RC)
OUT = sys.argv[1]; P = os.path.join(OUT, 'convergence.npz')
FR = [1.0, 1e3, 3e4]
if not os.path.exists(P):
    g = terrain.Grid(FL.sources('wire_halfspace'), n=512, dx=10.0, s=FL.S_HILL)
    CE, CH = terrain.coefficients(FR, g, FL.ground('wire_halfspace'), nord=24)
    i0 = g.i0
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(P, CE=CE[..., i0 - 30:i0 + 31], CH=CH[..., i0 - 30:i0 + 31], x=g.x[i0 - 30:i0 + 31], freqs=FR)
d = np.load(P)
CE, CH, x = d['CE'], d['CH'], d['x']
ip = int(np.argmin(np.abs(x)))
Hs = [30.0, 90.0, 150.0]; cols = [C['hill'], C['depression'], C['third']]
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True, sharey=True)
for j, f in enumerate(FR):
    for r, (lab, A, k) in enumerate([('$E_x$', CE, 0), ('$H_z$', CH, 2)]):
        a = ax[r, j]
        c = A[j, :, k, 1, ip]
        n = np.arange(len(c))
        for H, col in zip(Hs, cols):
            t = np.abs(c * H ** n) / np.abs(c[0])
            a.semilogy(n[1:], t[1:], 'o-', ms=2.5, color=col, label=f'H/s = {H / FL.S_HILL:.1f}')
        a.set_title(f'{lab}, f = {f:g} Hz', fontsize=8.5)
        if j == 0:
            a.set_ylabel(r'$|c_n H^n| / |c_0|$')
        if r == 1:
            a.set_xlabel('order n')
ax[0, 0].legend(loc='lower left')
ax[0, 0].set_ylim(1e-17, 3)
save(fig, 'convergence', OUT)
