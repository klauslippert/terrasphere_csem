#!/usr/bin/env python3
"""Transients on the profiles from the hill series coefficients (source from the coefficient file).

V(f; H) = V0(f) [closed form, all filter frequencies] + dV(f; H),   dV = S(H) - c_0 (series, f <= f_c(H)),
dV(f > f_c) = dV(f_c)  (constant high-frequency continuation; its influence is measured by repeating with the
cut-off moved down by a factor 4: '..._alt').  f_c(H): last frequency before the first one >= 1 Hz where the estimated
truncation error exceeds TOL of max|dV| (1e-3; 1e-2 for |H| > 100 m).  S = optimal truncation.
Low frequencies: the periodic box distorts the inductive part once the skin depth exceeds the box.  With
--low a.npz,b.npz (boxes L_a < L_b, same nord, profile points every 20 m) dV for f <= F_LOW is replaced by the
Richardson extrapolation  dV_inf = (L_b^2 dV_b - L_a^2 dV_a)/(L_b^2 - L_a^2)  (box error ~ L^-2, measured t09);
--check c.npz adds the transient from box c alone for comparison.
Outputs (switch-off of a step current at t = 0) on both profiles, terrain surface, earth side: E (e_x, e_y, e_z; V/m),
dB/dt (T/s) and B (T), arrays (nt, 3, 2, P).  Flat reference V0 = layered primary + sphere (flat) response.
The coefficient files come from coefficients.py (e^{-iwt}); Re V does not depend on the time convention.
Usage: python3 transients.py <filter-npz> <out.npz> [H1,H2,...] [--low ref.npz,a.npz,b.npz] [--check c.npz]"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from terrasphere_csem import MU0, halfspace, timedomain, terrain
from terrasphere_csem.terrain import ground_layers
from terrasphere_csem.terrain import optimal_sum
import cases as FL
TOL = 1e-3
F_LOW = 100.0


def taper(f, f1=3.0, f2=F_LOW):
    """1 below f1, 0 above f2, cos^2 in log f between: the box correction is faded out before the seam at F_LOW
    (above a few Hz the skin depth is well inside the production box; a jump at the seam spoils late times)."""
    u = np.clip(np.log(np.asarray(f) / f1) / np.log(f2 / f1), 0.0, 1.0)
    return np.cos(0.5 * np.pi * u) ** 2


def load_file(p):
    """Read the arrays needed from a coefficient file written by coefficients.py."""
    d = np.load(p)
    return {k: d[k] for k in ('freqs', 'x', 'CE', 'CH', 'n', 'dx', 'case')}


def dv_spectra(d, H, xp):
    """dV (nf, 5, 2, len(xp)) and error estimate for the profile positions xp."""
    ip = np.array([int(np.argmin(np.abs(d['x'] - v))) for v in xp]); assert np.allclose(d['x'][ip], xp)
    out = np.zeros((len(d['freqs']), 6, 2, len(xp)), complex); est = np.zeros(len(d['freqs']))
    for i in range(len(d['freqs'])):
        C = np.concatenate([d['CE'][i][..., ip], d['CH'][i][..., ip]], axis=1)
        S, err = optimal_sum(C, H)
        dv = S - C[0]; out[i] = dv
        est[i] = np.max(err / (np.abs(dv).reshape(6, -1).max(1)[:, None, None] + 1e-300))
    return out, est


def parse_args(argv):
    """Split the command line into positional arguments and the options --low (list of files) and --check (file)."""
    args = [a for a in argv]
    low = chk = None
    if '--low' in args:
        k = args.index('--low'); low = args[k + 1].split(','); del args[k:k + 2]
    if '--check' in args:
        k = args.index('--check'); chk = args[k + 1]; del args[k:k + 2]
    return args, low, chk


def flat_spectrum(case, d, xp, freqs):
    """Spectrum V0 (nf, 6, 2, n) of the flat layered earth (plus sphere) on the two profiles; E_z = 0 on the flat
    surface (earth side)."""
    n = len(xp)
    px = np.concatenate([np.zeros_like(xp), xp]); py = np.concatenate([xp, np.zeros_like(xp)])
    P = halfspace.surface_fields_layered(px, py, freqs, ground_layers(FL.ground(case)), FL.sources(case))
    kg = FL.sphere(case, terrain.Grid(FL.sources(case), n=int(d['n']), dx=float(d['dx']), s=FL.S_HILL))
    if kg is not None:
        for j, f in enumerate(freqs):
            pk = kg.surface_fields(2 * np.pi * f, FL.sources(case), px, py)
            for c in pk:
                P[c][j] += pk[c]
    V0 = np.zeros((len(freqs), 6, 2, n), complex)
    for k, c in [(0, 'Ex'), (1, 'Ey'), (3, 'Hx'), (4, 'Hy'), (5, 'Hz')]:
        V0[:, k, 0] = P[c][:, :n]; V0[:, k, 1] = P[c][:, n:]
    return V0


def transform(t, freqs, V0, dV):
    """Transients (E, dB/dt, B) of the spectrum V0 + dV."""
    V = V0 + dV
    return (timedomain.transient(t, freqs, V[:, :3], kind='off'),
            -timedomain.transient(t, freqs, MU0 * V[:, 3:], kind='imp'),
            timedomain.transient(t, freqs, MU0 * V[:, 3:], kind='off'))


def assemble(V0, freqs, fs, dvp, fcut, lowspec=None):
    """Series correction dV on the filter frequencies: the box-corrected lowspec up to F_LOW (if given), otherwise
    the series dvp (on the frequencies fs) up to the cut-off frequency fcut, and the value at fcut above it."""
    dV = np.zeros_like(V0)
    jc = int(np.argmin(np.abs(fs - fcut)))
    for j, f in enumerate(freqs):
        if lowspec is not None and f <= F_LOW * (1 + 1e-9):
            i = int(np.argmin(np.abs(lowspec[0] - f))); assert abs(lowspec[0][i] / f - 1) < 1e-8
            dV[j] = lowspec[1][i]
        elif f <= fs[jc] * (1 + 1e-9):
            i = int(np.argmin(np.abs(fs - f))); dV[j] = dvp[i]
        else:
            dV[j] = dvp[jc]
    return dV


def cutoff_frequency(fs, est, H):
    """Cut-off frequency f_c of the series for height H: the last frequency before the first one >= 1 Hz whose
    estimated truncation error is not below TOL (10 TOL for |H| > 100 m)."""
    tol = TOL if abs(H) <= 100 else 10 * TOL
    bad = (est >= tol) & (fs >= 1.0)
    ic = int(np.argmax(bad)) - 1 if bad.any() else len(fs) - 1
    fc = fs[ic]
    print(f"H = {H:+.0f} m: est < {tol:g} from 1 Hz up to f_c = {fc:.3g} Hz; max est below 1 Hz "
          f"{est[fs < 1].max():.1e}, 1 Hz..f_c {est[(fs >= 1) & (fs <= fc)].max():.1e}", flush=True)
    return fc


def low_frequency_correction(lows, fs, dvp, H, xp):
    """Box-corrected low-frequency spectrum (frequencies, dV) and the pieces (r0, b) of the correction.

    lows = [reference box with the production box size but the coarse grid, box a, box b]; dV = dV_production +
    [Richardson(a, b) - reference], all three on the same grid, so that truncation and grid errors cancel and the
    correction vanishes smoothly towards F_LOW."""
    (r0, _), (a, _), (b, _) = [dv_spectra(L, H, xp) for L in lows]
    La, Lb = [L['n'] * L['dx'] for L in lows[1:]]
    fl = lows[0]['freqs']; assert all(np.allclose(fl, L['freqs']) for L in lows)
    corr = ((Lb ** 2 * b - La ** 2 * a) / (Lb ** 2 - La ** 2) - r0) * taper(fl).reshape(-1, 1, 1, 1)
    ip_ = [int(np.argmin(np.abs(fs - f))) for f in fl]
    lowspec = (fl, dvp[ip_] + corr)
    print(f"   low frequencies <= {F_LOW} Hz: production + box correction (Richardson {La:.0f}/{Lb:.0f} m "
          f"minus {lows[0]['n'] * lows[0]['dx']:.0f} m); |corr| at {fl[-1]:.3g} Hz / max|corr| = "
          f"{np.abs(corr[-1]).max() / (np.abs(corr).max() + 1e-300):.1e}", flush=True)
    return lowspec, r0, b


def main():
    """Compute the transients from coefficient files (see the module docstring)."""
    args, low, chk = parse_args(sys.argv[1:])
    d = load_file(args[0]); out = args[1]
    Hs = [float(v) for v in args[2].split(',')] if len(args) > 2 else [30.0, -30.0, 90.0, -90.0]
    fs = d['freqs']
    xp = d['x'][np.abs(d['x']) <= FL.PROFILE_HALF]
    case = str(d['case'])
    lows = [load_file(p) for p in low] if low else []
    if lows:
        common = set(np.round(xp, 6))
        for L in lows:
            common &= set(np.round(L['x'], 6))
            assert str(L['case']) == case, "low-frequency files must belong to the same source"
        xp = np.array(sorted(common))
    t = timedomain.time_axis(FL.T_MIN, FL.T_MAX)
    all_freqs = timedomain.filter_frequencies(t)
    V0 = flat_spectrum(case, d, xp, all_freqs)
    res = dict(t=t, x=xp, Hs=np.array(Hs), freqs=all_freqs, low=str(low), check=str(chk), case=case)
    res['E_flat'] = timedomain.transient(t, all_freqs, V0[:, :3], kind='off')
    res['dBdt_flat'] = -timedomain.transient(t, all_freqs, MU0 * V0[:, 3:], kind='imp')
    res['B_flat'] = timedomain.transient(t, all_freqs, MU0 * V0[:, 3:], kind='off')
    for H in Hs:
        dvp, est = dv_spectra(d, H, xp)
        fc = cutoff_frequency(fs, est, H)
        lowspec = None
        if lows:
            lowspec, r0, b = low_frequency_correction(lows, fs, dvp, H, xp)
        for alt, fcut in [('', fc), ('_alt', fs[int(np.argmin(np.abs(fs - fc / 4)))])]:
            e, b_, bb = transform(t, all_freqs, V0, assemble(V0, all_freqs, fs, dvp, fcut, lowspec))
            res[f'E{alt}_{H:+.0f}'] = e; res[f'dBdt{alt}_{H:+.0f}'] = b_; res[f'B{alt}_{H:+.0f}'] = bb
        if lows:   # also with the largest box alone instead of the extrapolation, for the error estimate
            ip_ = [int(np.argmin(np.abs(fs - f))) for f in lows[0]['freqs']]
            e, b_, _ = transform(t, all_freqs, V0, assemble(V0, all_freqs, fs, dvp, fc, (lows[0]['freqs'], dvp[ip_] + b - r0)))
            res[f'E_boxB_{H:+.0f}'] = e; res[f'dBdt_boxB_{H:+.0f}'] = b_
        if chk:
            C = load_file(chk)
            e, b_, _ = transform(t, all_freqs, V0, assemble(V0, all_freqs, fs, dvp, fc, (C['freqs'], dv_spectra(C, H, xp)[0])))
            res[f'E_check_{H:+.0f}'] = e; res[f'dBdt_check_{H:+.0f}'] = b_
        res[f'fc_{H:+.0f}'] = fc; res[f'est_{H:+.0f}'] = est
    np.savez_compressed(out, **res)
    print('saved', out)


if __name__ == '__main__':
    main()
