#!/usr/bin/env python3
"""Transients on the profiles from the hill series coefficients (from gausshill transient.py, GH-017/018/022; source
from the coefficient file, KH-004).

V(f; H) = V0(f) [closed form, all filter frequencies] + dV(f; H),   dV = S(H) - c_0 (series, f <= f_c(H)),
dV(f > f_c) = dV(f_c)  (constant high-frequency continuation; its influence is measured by repeating with the
cut-off moved down by a factor 4: '..._alt').  f_c(H): last frequency before the first one >= 1 Hz where the estimated
truncation error exceeds TOL of max|dV| (1e-3; 1e-2 for |H| > 100 m).  S = optimal truncation.
Low frequencies (GH-018): the periodic box distorts the inductive part once the skin depth exceeds the box.  With
--low a.npz,b.npz (boxes L_a < L_b, same nord, profile points every 20 m) dV for f <= F_LOW is replaced by the
Richardson extrapolation  dV_inf = (L_b^2 dV_b - L_a^2 dV_a)/(L_b^2 - L_a^2)  (box error ~ L^-2, measured t09);
--check c.npz adds the transient from box c alone for comparison.
Outputs (switch-off of a step current at t = 0) on both profiles, terrain surface, earth side: E (e_x, e_y, e_z; V/m),
dB/dt (T/s) and B (T), arrays (nt, 3, 2, P).  Flat reference V0 = layered primary + sphere (flat) response.
The coefficient files come from koeffizienten.py (e^{-iwt}); Re V does not depend on the time convention.
Usage: python3 transient.py <filter-npz> <out.npz> [H1,H2,...] [--low ref.npz,a.npz,b.npz] [--check c.npz]"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from terrasphere_csem import MU0, halbraum, zeit, huegel
from terrasphere_csem.huegel import boden_schichten
from terrasphere_csem.huegel import summe_optimal
import faelle as FL
TOL = 1e-3
F_LOW = 100.0


def taper(f, f1=3.0, f2=F_LOW):
    """1 below f1, 0 above f2, cos^2 in log f between: the box correction is faded out before the seam at F_LOW
    (above a few Hz the skin depth is well inside the production box; a jump at the seam spoils late times, GH-022)."""
    u = np.clip(np.log(np.asarray(f) / f1) / np.log(f2 / f1), 0.0, 1.0)
    return np.cos(0.5 * np.pi * u) ** 2


def lade(p):
    d = np.load(p)
    return {k: d[k] for k in ('freqs', 'x', 'CE', 'CH', 'n', 'dx', 'fall')}


def dv_spektren(d, H, xp):
    """dV (nf, 5, 2, len(xp)) and error estimate for the profile positions xp."""
    ip = np.array([int(np.argmin(np.abs(d['x'] - v))) for v in xp]); assert np.allclose(d['x'][ip], xp)
    out = np.zeros((len(d['freqs']), 6, 2, len(xp)), complex); est = np.zeros(len(d['freqs']))
    for i in range(len(d['freqs'])):
        C = np.concatenate([d['CE'][i][..., ip], d['CH'][i][..., ip]], axis=1)
        S, err = summe_optimal(C, H)
        dv = S - C[0]; out[i] = dv
        est[i] = np.max(err / (np.abs(dv).reshape(6, -1).max(1)[:, None, None] + 1e-300))
    return out, est


def main():
    args = [a for a in sys.argv[1:]]
    low = chk = None
    if '--low' in args:
        k = args.index('--low'); low = args[k + 1].split(','); del args[k:k + 2]
    if '--check' in args:
        k = args.index('--check'); chk = args[k + 1]; del args[k:k + 2]
    d = lade(args[0]); out = args[1]
    Hs = [float(v) for v in args[2].split(',')] if len(args) > 2 else [30.0, -30.0, 90.0, -90.0]
    fs = d['freqs']
    xp = d['x'][np.abs(d['x']) <= FL.PROFILE_HALF]
    fall = str(d['fall'])
    lows = [lade(p) for p in low] if low else []
    if lows:
        common = set(np.round(xp, 6))
        for L in lows:
            common &= set(np.round(L['x'], 6))
            assert str(L['fall']) == fall, "low-frequency files must belong to the same source"
        xp = np.array(sorted(common))
    t = zeit.zeitachse(FL.T_MIN, FL.T_MAX)
    fall_f = zeit.filter_frequenzen(t)
    n = len(xp)
    px = np.concatenate([np.zeros_like(xp), xp]); py = np.concatenate([xp, np.zeros_like(xp)])
    P = halbraum.oberflaeche_schichten(px, py, fall_f, boden_schichten(FL.boden(fall)), FL.quellen(fall))
    kg = FL.kugel(fall, huegel.Gitter(FL.quellen(fall), n=int(d['n']), dx=float(d['dx']), s=FL.S_HILL))
    if kg is not None:
        for j, f in enumerate(fall_f):
            pk = kg.oberflaeche(2 * np.pi * f, FL.quellen(fall), px, py)
            for c in pk:
                P[c][j] += pk[c]
    V0 = np.zeros((len(fall_f), 6, 2, n), complex)          # E_z = 0 on the flat surface (earth side)
    for k, c in [(0, 'Ex'), (1, 'Ey'), (3, 'Hx'), (4, 'Hy'), (5, 'Hz')]:
        V0[:, k, 0] = P[c][:, :n]; V0[:, k, 1] = P[c][:, n:]
    res = dict(t=t, x=xp, Hs=np.array(Hs), freqs=fall_f, low=str(low), check=str(chk), fall=fall)
    res['E_flat'] = zeit.transient(t, fall_f, V0[:, :3], art='aus')
    res['dBdt_flat'] = -zeit.transient(t, fall_f, MU0 * V0[:, 3:], art='imp')
    res['B_flat'] = zeit.transient(t, fall_f, MU0 * V0[:, 3:], art='aus')

    def transform(dV):
        V = V0 + dV
        return (zeit.transient(t, fall_f, V[:, :3], art='aus'), -zeit.transient(t, fall_f, MU0 * V[:, 3:], art='imp'),
                zeit.transient(t, fall_f, MU0 * V[:, 3:], art='aus'))

    def zusammensetzen(dvp, fcut, lowspec=None):
        dV = np.zeros_like(V0)
        jc = int(np.argmin(np.abs(fs - fcut)))
        for j, f in enumerate(fall_f):
            if lowspec is not None and f <= F_LOW * (1 + 1e-9):
                i = int(np.argmin(np.abs(lowspec[0] - f))); assert abs(lowspec[0][i] / f - 1) < 1e-8
                dV[j] = lowspec[1][i]
            elif f <= fs[jc] * (1 + 1e-9):
                i = int(np.argmin(np.abs(fs - f))); dV[j] = dvp[i]
            else:
                dV[j] = dvp[jc]
        return dV

    for H in Hs:
        dvp, est = dv_spektren(d, H, xp)
        tol = TOL if abs(H) <= 100 else 10 * TOL
        bad = (est >= tol) & (fs >= 1.0)
        ic = int(np.argmax(bad)) - 1 if bad.any() else len(fs) - 1
        fc = fs[ic]
        print(f"H = {H:+.0f} m: est < {tol:g} from 1 Hz up to f_c = {fc:.3g} Hz; max est below 1 Hz "
              f"{est[fs < 1].max():.1e}, 1 Hz..f_c {est[(fs >= 1) & (fs <= fc)].max():.1e}", flush=True)
        lowspec = None
        if lows:
            # correction mode (GH-022): lows = [reference box with the production box size but the coarse grid,
            # box a, box b]; dV = dV_production + [Richardson(a, b) - reference], all three on the same grid, so
            # that truncation and grid errors cancel and the correction vanishes smoothly towards F_LOW
            (r0, _), (a, _), (b, _) = [dv_spektren(L, H, xp) for L in lows]
            La, Lb = [L['n'] * L['dx'] for L in lows[1:]]
            fl = lows[0]['freqs']; assert all(np.allclose(fl, L['freqs']) for L in lows)
            corr = ((Lb ** 2 * b - La ** 2 * a) / (Lb ** 2 - La ** 2) - r0) * taper(fl).reshape(-1, 1, 1, 1)
            ip_ = [int(np.argmin(np.abs(fs - f))) for f in fl]
            lowspec = (fl, dvp[ip_] + corr)
            print(f"   low frequencies <= {F_LOW} Hz: production + box correction (Richardson {La:.0f}/{Lb:.0f} m "
                  f"minus {lows[0]['n'] * lows[0]['dx']:.0f} m); |corr| at {fl[-1]:.3g} Hz / max|corr| = "
                  f"{np.abs(corr[-1]).max() / (np.abs(corr).max() + 1e-300):.1e}", flush=True)
        for alt, fcut in [('', fc), ('_alt', fs[int(np.argmin(np.abs(fs - fc / 4)))])]:
            e, b_, bb = transform(zusammensetzen(dvp, fcut, lowspec))
            res[f'E{alt}_{H:+.0f}'] = e; res[f'dBdt{alt}_{H:+.0f}'] = b_; res[f'B{alt}_{H:+.0f}'] = bb
        if lows:   # also with the largest box alone instead of the extrapolation, for the error estimate
            ip_ = [int(np.argmin(np.abs(fs - f))) for f in lows[0]['freqs']]
            e, b_, _ = transform(zusammensetzen(dvp, fc, (lows[0]['freqs'], dvp[ip_] + b - r0)))
            res[f'E_boxB_{H:+.0f}'] = e; res[f'dBdt_boxB_{H:+.0f}'] = b_
        if chk:
            C = lade(chk)
            e, b_, _ = transform(zusammensetzen(dvp, fc, (C['freqs'], dv_spektren(C, H, xp)[0])))
            res[f'E_check_{H:+.0f}'] = e; res[f'dBdt_check_{H:+.0f}'] = b_
        res[f'fc_{H:+.0f}'] = fc; res[f'est_{H:+.0f}'] = est
    np.savez_compressed(out, **res)
    print('saved', out)


if __name__ == '__main__':
    main()
