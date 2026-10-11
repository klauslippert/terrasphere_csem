#!/usr/bin/env python3
"""Figure and numbers: sphere below a hill, from the coefficient file written by scripts/coupling.py.
(a) convergence of the coupled series on the summit profiles: max_n |c_n H^n| / max |c_0| for H = 30 and 90 m;
(b, c) E_x and H_z along the y-profile (x = 0) at 30 Hz: terrain effect without sphere, sphere effect without terrain,
both together, and the coupling (together minus the sum of the two), relative to the profile maximum of the flat
layered field.  Printed: coupling relative to the sphere effect, box dependence of the coupled terrain effect.
Usage: python3 fig_coupling.py <coupling.npz> <output-dir>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from style import plt, C, save
from terrasphere_csem.terrain import optimal_sum

d = np.load(sys.argv[1])
fr, x = d['freqs'], d['x']
sel = np.abs(x) <= 600.0
# the case names in the file: files written before the English renaming use 'draht_kugel' / 'draht_schichten'
K, S = ('wire_sphere', 'wire_layered') if 'CE_wire_sphere' in d.files else ('draht_kugel', 'draht_schichten')
C6 = lambda key, i: np.concatenate([d[f'CE_{key}'][i], d[f'CH_{key}'][i]], axis=1)      # (nord+1, 6, 2, N)

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
mk = ['o', 's', '^', 'D']
for H, ls in [(30.0, '-'), (90.0, ':')]:
    for i, f in enumerate(fr):
        c = C6(K, i)[:, [0, 1, 3, 4, 5]][..., sel]                    # E_z = 0 on the flat surface: excluded
        ref = np.abs(c[0]).reshape(5, -1).max(1)[:, None, None]
        t = (np.abs(c * (H ** np.arange(c.shape[0]))[:, None, None, None]) / ref).reshape(c.shape[0], -1).max(1)
        ax[0].semilogy(np.arange(1, c.shape[0]), t[1:], ls=ls, marker=mk[i], ms=2.5,
                       color=plt.cm.viridis(i / len(fr)), label=f'{f:g} Hz' if H == 30 else None)
ax[0].set_xlabel('order n'); ax[0].set_ylabel(r'max $|c_nH^n|$ / max $|c_0|$ (per comp.)')
ax[0].set_title('coupled series (solid 30 m, dotted 90 m)', fontsize=8); ax[0].legend(fontsize=6)

i30 = int(np.argmin(np.abs(fr - 30.0)))
H = 30.0
k, s = C6(K, i30), C6(S, i30)
flat = s[0]
terr = optimal_sum(s, H)[0] - s[0]
sph = k[0] - s[0]
both = optimal_sum(k, H)[0] - s[0]
coup = both - terr - sph
for a, comp, name in [(ax[1], 0, r'$E_x$'), (ax[2], 5, r'$H_z$')]:
    ref = np.abs(flat[comp, 0, sel]).max()
    for v, col, lab, lst in [(terr, C['hill'], 'hill only', '-'), (sph, C['sphere'], 'sphere only', '-'),
                             (both, 'k', 'hill + sphere', '--'), (coup, C['third'], 'coupling', '-')]:
        a.plot(x[sel], 100 * np.real(v[comp, 0, sel]) / ref, color=col, ls=lst, label=lab)
    a.set_xlabel('y (m), x = 0'); a.set_ylabel(f'Re {name} change (% of flat max)')
    a.set_title(f'{name}, 30 Hz, H = 30 m', fontsize=8)
ax[1].legend(fontsize=6)
fig.tight_layout()
save(fig, 'coupling', sys.argv[2])

for i, f in enumerate(fr):
    k, s = C6(K, i), C6(S, i)
    for H in (30.0, -30.0, 90.0):
        terr = optimal_sum(s, H)[0] - s[0]; sph = k[0] - s[0]; both = optimal_sum(k, H)[0] - s[0]
        coup = both - terr - sph
        r = [np.abs(coup[c][..., sel]).max() / np.abs(sph[c][..., sel]).max() for c in range(6)]
        rt = [np.abs(coup[c][..., sel]).max() / np.abs(terr[c][..., sel]).max() for c in range(6)]
        err = optimal_sum(k, H)[1][..., sel].max() / np.abs(k[0][..., sel]).max()
        print(f"f = {f:6g} Hz, H = {H:+.0f}: coupling / sphere effect " + ' '.join(f'{v:.2f}' for v in r)
              + " | coupling / terrain effect " + ' '.join(f'{v:.2f}' for v in rt) + f" | last term {err:.1e}")
x1 = d['x1024']; s1 = np.abs(x1) <= 600.0
for key in (S, K):
    a = np.concatenate([d[f'CE512_{key}'][0], d[f'CH512_{key}'][0]], axis=1)[..., sel]
    b = np.concatenate([d[f'CE1024_{key}'][0], d[f'CH1024_{key}'][0]], axis=1)[..., s1]
    for H in (30.0, 90.0):
        ea = optimal_sum(a, H)[0] - a[0]; eb = optimal_sum(b, H)[0] - b[0]
        e = [np.abs(ea[c] - eb[c]).max() / np.abs(eb[c]).max() for c in range(6)]
        print(f"box 5.12 vs 10.24 km, 3 Hz, {key}, H = {H:.0f}: terrain effect changes by " + ' '.join(f'{v:.1e}' for v in e))
