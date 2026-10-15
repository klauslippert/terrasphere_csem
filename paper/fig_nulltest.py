#!/usr/bin/env python3
"""Figure nulltest: null test against empymod (quasi-static, eperm = 0): sphere with the conductivity of its layer.
(a) frequency domain, sphere in the middle layer of a three-layer earth, off-axis HED, VMD and HMD sources at the
    surface, all components of E and H at surface and buried receivers: max relative deviation of the total
    response (terrasphere_csem: 1-D kernels + sphere part) from empymod, and the sphere part itself relative to the
    sphere part with contrast (sigma_s = 2 S/m);
(b) time domain, coaxial loop over the sphere in a half-space (1:100 design), receiver loop radius 0.7 a;
(c) time domain, 500 m grounded wire in the three-layer earth;
(d) first order: dV/dsigma_s at null contrast against the Born integral of two empymod fields
    (tests/verification/empymod/test_born.py).
The results are collected in <output-dir>/nulltest.npz; parts given as options (--a --b --c --d) are recomputed,
the others taken from that file; --plot only redraws.  Without options everything is computed.
Usage: python3 fig_nulltest.py <output-dir> [--a] [--b] [--c] [--d] [--plot]"""
import os, sys, time
import numpy as np
import empymod
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
TESTS = os.path.join(os.path.dirname(HERE), 'tests')   # reference solutions of the tests
sys.path += [os.path.join(TESTS, d) for d in ('verification', os.path.join('verification', 'empymod'), 'reference')]
from style import plt, save, LS_RC, COL
from terrasphere_csem import MU0
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole, grounded_wire
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver
from terrasphere_csem.onedim import loop_to_loop_emf
from terrasphere_csem.timedomain import time_axis, Transient, model_transient
from empymod_loop import polygon
from test_onedim import emp_fields, AIR
import test_born as TB

plt.rcParams.update(LS_RC)
QS = lambda n: dict(epermH=[0.0] * n, epermV=[0.0] * n)
OUT = sys.argv[1]; DATA = os.path.join(OUT, 'nulltest.npz'); os.makedirs(OUT, exist_ok=True)
opts = set(a.lstrip('-') for a in sys.argv[2:])
WANT = set() if 'plot' in opts else (opts or {'a', 'b', 'c', 'd'})
out = dict(np.load(DATA)) if os.path.exists(DATA) else {}
t0 = time.time()
# ---------------------------------------------------------------- (a)
depths = [0.0, 100.0, 300.0]; sig = [0.02, 0.005, 0.03]; st = LayerStack(depths, sig); res = [AIR] + [1 / s for s in sig]
if 'a' in WANT:
    kug0 = Sphere(60.0, sig[1]); kug1 = Sphere(60.0, 2.0); C = (100.0, 50.0, 200.0)
    srcs = [Dipole('HED', (300.0, -100.0, 0.0), 1.0, 0.4), Dipole('VMD', (-200.0, 150.0, 0.0), 1.0), Dipole('HMD', (0.0, 300.0, 0.0), 1.0, 1.2)]
    pts = [(-150.0, -80.0, 0.0), (250.0, 220.0, 0.0), (-100.0, -150.0, 350.0)]
    freqs = np.geomspace(1, 1e4, 9)
    dev = np.zeros((len(srcs), len(freqs))); sph = np.zeros((len(srcs), len(freqs)))
    for js, src in enumerate(srcs):
        for jf, f in enumerate(freqs):
            w = 2 * np.pi * f
            rx = [Receiver(t, p, e) for p in pts for t in ('E', 'H') for e in np.eye(3) if not (t == 'E' and e[2] == 1 and p[2] == 0.0)]
            m0 = SphereModel(st, kug0, C, nmax=10); tot = m0.response(w, [src], rx)
            s0 = m0.scattered_field(w, [src], rx); s1 = SphereModel(st, kug1, C, nmax=10).scattered_field(w, [src], rx)
            Ee, He = emp_fields(src.kind, src.pos, pts, depths, res, f, src.psi)
            ref = np.array([(Ee if r.kind == 'E' else He)[pts.index(tuple(r.pos))] @ r.vec for r in rx])
            dev[js, jf] = np.max(np.abs(tot - ref)) / np.max(np.abs(ref))
            sph[js, jf] = np.max(np.abs(s0)) / np.max(np.abs(s1))
        print('(a)', src.kind, 'done', f'{time.time()-t0:.0f} s')
    out.update(a_freqs=freqs, a_dev=dev, a_sph=sph)
    np.savez(DATA, **out)
# ---------------------------------------------------------------- (b)
if 'b' in WANT:
    A, H, B, sh = 200.0, 250.0, 180.0, 0.01; st_b = LayerStack([0.0], [sh]); Ar = 0.7 * A
    tx = Dipole('LOOP', (0.0, 0.0, 0.0), 1.0, radius=A); rxl = Receiver('LOOP', (0.0, 0.0, 0.0), radius=Ar)
    T = time_axis(1e-4, 1e-1)[::3]                         # every third lagged time: exact, 201 + 3K frequencies
    for lab, ss in (('null', sh), ('contrast', 1.0)):
        mod = SphereModel(st_b, Sphere(B, ss), (0.0, 0.0, H), nmax=15)
        out[f'b_{lab}_s'] = model_transient(mod, [tx], [rxl], T).switch_off()[:, 0]
    out['b_p'] = Transient(lambda wi: loop_to_loop_emf(st_b, tx, rxl, wi), T).switch_off()[:, 0]
    srcp = polygon(A, 256, z=1e-6)
    emp = np.asarray(empymod.bipole(src=srcp, rec=[Ar, 0.0, 0.02, 90.0, 0.0], depth=[0], res=[AIR, 1 / sh], freqtime=T, signal=-1, srcpts=5, strength=1.0, verb=0,
                                    htarg={'dlf': 'key_401_2009'}, ftarg={'pts_per_dec': -1}, **QS(2))).reshape(len(T), -1).sum(-1) * 2 * np.pi * Ar
    out.update(b_T=T, b_emp=emp); print('(b) done', f'{time.time()-t0:.0f} s')
    np.savez(DATA, **out)
# ---------------------------------------------------------------- (c)
if 'c' in WANT:
    Tc = time_axis(1e-4, 1e-1)[::3]
    wire = grounded_wire((-250.0, 0.0), (250.0, 0.0), 1.0, n_gauss=7)
    rxc = [Receiver('E', (0.0, 600.0, 0.0), (1, 0, 0)), Receiver('dBdt', (300.0, 350.0, 0.0), (0, 0, 1))]
    Cc = (100.0, 200.0, 200.0)
    for lab, ss in (('null', sig[1]), ('contrast', 2.0)):
        mod = SphereModel(st, Sphere(60.0, ss), Cc, nmax=12)
        trs = model_transient(mod, wire, rxc, Tc, verbose=True)
        out[f'c_{lab}_on'] = trs.switch_on(); out[f'c_{lab}_off'] = trs.switch_off(); print('(c)', lab, f'{time.time()-t0:.0f} s', flush=True)
    trp = Transient(lambda wi: SphereModel(st, Sphere(60.0, sig[1]), Cc, nmax=12).primary_field(wi, wire, rxc), Tc)
    pon, poff = trp.switch_on(), trp.switch_off()
    # empymod: the wire as a Gauss sum of point dipoles (as srcpts=7 does internally); empymod 2.6.0 returns NaN for
    # magnetic receivers when the finite bipole lies 1e-6 m below the surface, the point dipoles do not
    xg, wg = np.polynomial.legendre.leggauss(7)
    emp = lambda rec, signal, mrec: sum(wi * 250.0 * np.real(np.asarray(empymod.bipole(
        src=[xi * 250.0, 0.0, 1e-6, 0.0, 0.0], rec=rec, depth=depths, res=res, freqtime=Tc, signal=signal, strength=0,
        mrec=mrec, verb=0, htarg={'dlf': 'key_401_2009'}, ftarg={'pts_per_dec': -1}, **QS(4)))) for xi, wi in zip(xg, wg))
    ex = emp([0.0, 600.0, 1e-6, 0.0, 0.0], 1, False)
    hz = emp([300.0, 350.0, 1e-6, 0.0, 90.0], 0, True)
    out.update(c_T=Tc, c_tot0=out['c_null_on'][:, 0] + pon[:, 0], c_tot1=out['c_null_off'][:, 1] + poff[:, 1], c_emp0=ex, c_emp1=-MU0 * hz,   # switch-off dB_z/dt = -mu0 * impulse response
               c_con0=out['c_contrast_on'][:, 0], c_con1=out['c_contrast_off'][:, 1], c_p0=pon[:, 0], c_p1=poff[:, 1]); print('(c) done', flush=True)
    np.savez(DATA, **out)
# ---------------------------------------------------------------- (d)
if 'd' in WANT:
    st_d = LayerStack(TB.DEPTHS, TB.SIG); mod = SphereModel(st_d, Sphere(TB.B, TB.SIG[1]), (TB.XC, TB.YC, TB.ZC), nmax=8)
    txs = [Dipole('HED', (300.0, 100.0, 0.0), 1.0, 0.5), Dipole('VMD', (-150.0, 80.0, 0.0), 1.0)]
    rxs = [('E', (-250.0, 150.0, 0.0), (1.0, 0.0, 0.0)), ('dBdt', (100.0, 250.0, 0.0), (0.0, 0.0, 1.0))]
    fd = np.geomspace(1, 1e4, 5); q = TB.sphere_quadrature(); dd = np.zeros((4, len(fd)))
    for jf, f in enumerate(fd):
        k = 0
        for tx_ in txs:
            ours = mod.derivative(2 * np.pi * f, [tx_], [Receiver(t, p, v) for (t, p, v) in rxs], 'sigma_s')
            for ir, (t, p, v) in enumerate(rxs):
                b = TB.born(tx_, t, p, v, f, q); dd[k, jf] = abs(ours[ir] - b) / abs(b); k += 1
        print('(d)', f, f'{time.time()-t0:.0f} s')
    out.update(d_freqs=fd, d_dev=dd)
    np.savez(DATA, **out); print('all data saved')
d = np.load(DATA)
fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
a = ax[0, 0]
for js, lab in enumerate(('HED (300, $-$100, 0) m', 'VMD ($-$200, 150, 0) m', 'HMD (0, 300, 0) m')):
    a.loglog(d['a_freqs'], d['a_dev'][js], 'o-', ms=3, color=COL[js], label=f'total vs empymod, {lab}')
    a.loglog(d['a_freqs'], np.maximum(d['a_sph'][js], 1e-18), 's:', ms=3, color=COL[js], alpha=0.6)
a.plot([], [], 's:', color='k', alpha=0.6, label='sphere part (null) / sphere part ($\\sigma_{\\rm sph}=2$ S/m)')
a.set_xlabel('frequency (Hz)'); a.set_ylabel('relative deviation'); a.set_ylim(1e-18, 1e-3); a.legend(fontsize=6.3, loc='upper left')
a.set_title('(a) three layers, sphere in layer 2, all components of $\\mathbf{E}$, $\\mathbf{H}$', fontsize=8)
a = ax[0, 1]; T = d['b_T']; tot = d['b_null_s'] + d['b_p']
a.loglog(T, np.abs(tot / d['b_emp'] - 1), 'o-', ms=3, color=COL[0], label='|total ($\\sigma_{\\rm sph}=\\sigma_h$) / empymod $-$ 1|')
a.loglog(T, np.abs(d['b_contrast_s'] / (d['b_contrast_s'] + d['b_p'])), 's--', ms=3, color=COL[2], label='|sphere share of total|, $\\sigma_{\\rm sph} = 1$ S/m')
a.set_ylim(1e-6, 3); a.set_xlabel('time (s)'); a.set_ylabel('switch-off'); a.legend(fontsize=6.5, loc='lower right')
a.set_title('(b) coaxial loop, half-space (1:100 design), Rx radius $0.7a$', fontsize=8)
a = ax[1, 0]; T = d['c_T']
a.loglog(T, np.abs(d['c_tot0'] / d['c_emp0'] - 1), 'o-', ms=3, color=COL[0], label='$E_x$ (switch-on): |total / empymod $-$ 1|')
a.loglog(T, np.abs(d['c_tot1'] / d['c_emp1'] - 1), 's-', ms=3, color=COL[1], label='$dB_z/dt$ (switch-off): |total / empymod $-$ 1|')
a.loglog(T, np.abs(d['c_con0'] / d['c_tot0']), 'o--', ms=3, color=COL[0], alpha=0.5, label='$E_x$: |sphere share|, $\\sigma_{\\rm sph}=2$ S/m')
a.loglog(T, np.abs(d['c_con1'] / d['c_tot1']), 's--', ms=3, color=COL[1], alpha=0.5, label='$dB_z/dt$: |sphere share|, $\\sigma_{\\rm sph}=2$ S/m')
a.set_ylim(1e-10, 1); a.set_xlabel('time (s)'); a.set_ylabel('relative'); a.legend(fontsize=6.3, loc='center right')
a.set_title('(c) three layers, 500 m grounded wire', fontsize=8)
a = ax[1, 1]
labs = ['HED $\\to E_x$', 'HED $\\to dB_z/dt$', 'VMD $\\to E_x$', 'VMD $\\to dB_z/dt$']
for k in range(4):
    a.loglog(d['d_freqs'], np.maximum(d['d_dev'][k], 1e-16), 'o-', ms=3, color=COL[k], label=labs[k])
a.set_xlabel('frequency (Hz)'); a.set_ylabel('relative deviation'); a.set_ylim(1e-16, 1e-6); a.legend(fontsize=6.5)
a.set_title('(d) first order: $\\partial V/\\partial\\sigma_{\\rm sph}$ at null contrast vs Born integral', fontsize=8)
fig.tight_layout()
save(fig, 'nulltest', OUT, png=False)
