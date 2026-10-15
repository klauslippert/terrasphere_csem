#!/usr/bin/env python3
"""Main program of the package (series computation): reads a control file, writes <folder of the control file>/results/:
  log.txt      control file, grids, timing, convergence, box correction
  coefficients.npz  only if [output] formats contains "coefficients": the series coefficients from which the
      fields are assembled, S(H) = sum_n c_n H^n.  The c_n do not depend on the height H, so further heights (and,
      method fft, other points of the profiles) can be evaluated from them without recomputing, and the size of
      each order shows the convergence.  Contents: freqs, method; fft: C (nf, nord + 1, 6, 2, N) = c_n of E and H
      (x, y, z) on the x- and y-profile at the grid points x (N), and the coarse runs of the box correction
      C_box<N>, x_box<N>; hankel: C_receivers (nf, nord + 1, 6, P) at the receivers.  Not needed for the fields.
  frequency domain ([frequencies]):  fields.npz / fields.csv  (as chosen in [output] formats: npz, csv)
      E, H (nh, nf, 3, P) complex, e^{-iwt}, components (x, y, z), z up, fields on the terrain (earth side);
      px, py, pz, profile, error (nh, nf) estimated truncation error relative to the largest flat E resp. H
  time domain ([time], signal off|on):  transients.npz / transients.csv
      E (V/m), dBdt (T/s), B (T) (nh, nt, 3, P) for a 1 A step of the chosen sign; t; px, py, pz, profile; f_c per H
Spectrum per height H: V = flat field (closed form + layer correction + sphere, at every frequency) + terrain part
S(H) - c_0 of the series, (i) box-corrected for f <= box_fmax by [Richardson(L_b, L_c) - L_a] of coarse runs
(20 m grid, boxes 5.12/10.24/20.48 km, 12 orders; box error ~ L^-2), (ii) in the time domain
continued as a constant above f_c, the last frequency (from 1 Hz up) whose estimated truncation error is below
`tolerance` of the largest terrain effect.
If all heights are 0, the terrain series is skipped: the fields are the flat field alone (no coefficients.npz).
Usage: python3 -m terrasphere_csem <control file>      (or <folder>, then <folder>/control.toml)"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # also as a plain script: python3 terrasphere_csem/__main__.py
import numpy as np
from terrasphere_csem import terrain, halfspace, timedomain, MU0
from terrasphere_csem.config import Config
from terrasphere_csem.terrain import optimal_sum

COMPONENTS_E, COMPONENTS_H = ['Ex', 'Ey', 'Ez'], ['Hx', 'Hy', 'Hz']


def coeff_fft(st, freqs, n, dx, kc, nord, log):
    """Series coefficients with the FFT grid (n points, spacing dx): array (nf, nord + 1, 6, 2, N) of E and H on the profiles, and x."""
    g = st.grid(n=n, dx=dx)
    kg = st.sphere_coupling(g)
    t0 = time.time()
    CE, CH = terrain.coefficients(freqs, g, st.layers, nord=nord, kc=kc, sphere=kg)
    log(f"  grid N = {n}, dx = {dx} m (box {n * dx / 1000:.2f} km), {nord} orders, low-pass {kc}: "
        f"{len(freqs)} frequencies in {time.time() - t0:.0f} s")
    return np.concatenate([CE, CH], axis=2), g.x                    # (nf, nord+1, 6, 2, N)


def coeff_hankel(st, freqs, log):
    """Series coefficients with the Hankel method: array (nf, nord + 1, 6, P) of E and H at the receivers."""
    from terrasphere_csem import terrain_hankel
    g = st.hankel_grid()
    kg = st.sphere_coupling(g)
    t0 = time.time()
    CE, CH = terrain_hankel.coefficients(freqs, g, st.layers, nord=int(st.numerics['orders']), sphere=kg)
    log(f"  Hankel series: |m| <= {g.M}, {len(g.r)} r nodes (0..{g.r_max:.0f} m), {len(g.lam)} lambda nodes "
        f"(0..{g.lam[-1]:.3f} rad/m), {st.numerics['orders']} orders: {len(freqs)} frequencies in {time.time() - t0:.0f} s")
    return np.concatenate([CE, CH], axis=2)                          # (nf, nord+1, 6, P) at the receivers


def at_receivers(C, x, px, py, prof):
    """C (..., 6, 2, N) on the lines -> (..., 6, P) at the receivers."""
    out = []
    for X, Y, p in zip(px, py, prof):
        line, v = (1, X) if p == 'x' else (0, Y)
        i = int(np.argmin(np.abs(x - v))); assert abs(x[i] - v) < 1e-6, "receiver not on the grid"
        out.append(C[..., line, i])
    return np.stack(out, axis=-1)


def flat_field(st, freqs, px, py):
    """Flat-earth field at the receivers for all frequencies: (nf, 6, P) (E_z = 0 on the flat surface)."""
    q = st.sources()
    P = halfspace.surface_fields_layered(px, py, freqs, st.layers, q)
    kg = st.sphere_coupling(st.hankel_grid() if st.numerics['method'] == 'hankel' else st.grid()) if st.sphere else None
    out = np.zeros((len(freqs), 6, len(px)), complex)
    for k, c in [(0, 'Ex'), (1, 'Ey'), (3, 'Hx'), (4, 'Hy'), (5, 'Hz')]:
        out[:, k] = P[c]
    if kg is not None:
        for i, f in enumerate(freqs):
            pk = kg.surface_fields(2 * np.pi * f, q, px, py)
            for k, c in [(0, 'Ex'), (1, 'Ey'), (3, 'Hx'), (4, 'Hy'), (5, 'Hz')]:
                out[i, k] += pk[c]
    return out


def terrain_effect(st, Cr, log):
    """Terrain part dV (nh, nf, 6, P) of the spectrum per height (optimal-sum series minus the flat field) and the
    estimated relative truncation error est (nh, nf)."""
    nh, nf, P = len(st.heights), Cr.shape[0], Cr.shape[-1]
    dV = np.zeros((nh, nf, 6, P), complex); est = np.zeros((nh, nf))
    for ih, H in enumerate(st.heights):
        for i in range(nf):
            S, err = optimal_sum(Cr[i], H)
            dV[ih, i] = S - Cr[i, 0]
            rE, rH = np.abs(Cr[i, 0, :3]).max(), np.abs(Cr[i, 0, 3:]).max()
            est[ih, i] = max(err[:3].max() / rE, err[3:].max() / rH) if H != 0 else 0.0
        log(f"  H = {H:+.0f} m: estimated truncation error (relative to the flat field), max {est[ih].max():.1e}")
    return dV, est


def apply_box_correction(st, fr, dV, res, receivers, log):
    """Correct dV in place for the periodic box of the FFT method (frequencies up to box_fmax).

    Coarse runs on three box sizes are combined by Richardson extrapolation (box error ~ L^-2); the correction is
    [Richardson(L_b, L_c) - L_a]. In the time domain it is faded out towards box_fmax. The coarse coefficients are
    stored in res."""
    nu = st.numerics
    px, py, prof = receivers
    fl = fr[fr <= float(nu['box_fmax'])]
    if not (nu['method'] == 'fft' and nu['box_correction'] and len(fl) and any(H != 0 for H in st.heights)):
        return
    Cb = []
    for nb in nu['box_n']:
        c, xb = coeff_fft(st, fl, int(nb), float(nu['box_dx']), float(nu['box_lowpass']), int(nu['box_orders']), log)
        Cb.append(at_receivers(c, xb, px, py, prof)); res[f'C_box{nb}'] = c; res[f'x_box{nb}'] = xb
    L = [int(nb) * float(nu['box_dx']) for nb in nu['box_n']]
    for ih, H in enumerate(st.heights):
        if H == 0:
            continue
        m = []
        for jf, f in enumerate(fl):
            i = int(np.argmin(np.abs(fr - f)))
            d = [optimal_sum(cb[jf], H)[0] - cb[jf][0] for cb in Cb]
            corr = (L[2] ** 2 * d[2] - L[1] ** 2 * d[1]) / (L[2] ** 2 - L[1] ** 2) - d[0]
            if st.domain == 'time':                            # fade out towards box_fmax (seam)
                u = np.clip(np.log(f / 3.0) / np.log(float(nu['box_fmax']) / 3.0), 0.0, 1.0)
                corr = corr * np.cos(0.5 * np.pi * u) ** 2
            m.append(np.abs(corr).max() / (np.abs(dV[ih, i]).max() + 1e-300))
            dV[ih, i] = dV[ih, i] + corr
        log(f"  H = {H:+.0f} m: box correction / terrain effect, max over f <= {nu['box_fmax']} Hz: {max(m):.1e}")
    log(f"  box correction: Richardson {L[1]:.0f}/{L[2]:.0f} m minus {L[0]:.0f} m")


def write_fields(st, results, fr, Cr, dV, est, header, prof, log):
    """Frequency-domain outputs (fields.npz / fields.csv): flat field plus terrain part per height."""
    P = Cr.shape[-1]
    px, py, pz = header['px'], header['py'], header['pz']
    F = Cr[None, :, 0] + dV                                     # flat field (exact at the receivers) + terrain part
    if 'npz' in st.formats:
        np.savez_compressed(os.path.join(results, 'fields.npz'), E=F[:, :, :3], H=F[:, :, 3:], freqs=fr, error=est,
                            time_convention='exp(-iwt)', units='E V/m, H A/m', **header)
    if 'csv' in st.formats:
        with open(os.path.join(results, 'fields.csv'), 'w') as fh:
            fh.write('H_m,f_Hz,profile,x_m,y_m,z_m,' + ','.join(f'{c}_{p}' for c in COMPONENTS_E + COMPONENTS_H
                                                              for p in ['re', 'im']) + '\n')
            for ih, H in enumerate(st.heights):
                for i, f in enumerate(fr):
                    for k in range(P):
                        fh.write(f"{H:g},{f:g},{prof[k]},{px[k]:g},{py[k]:g},{pz[ih, k]:.6g}," +
                                 ','.join(f"{a.real:.10e},{a.imag:.10e}" for a in F[ih, i, :, k]) + '\n')
    log(f"written: {results}: " + ', '.join(f'fields.{f}' for f in st.formats if f != 'coefficients'))


def write_transients(st, results, fr, Cr, dV, est, header, prof, px, py, log):
    """Time-domain outputs (transients.npz / transients.csv): step responses E, dB/dt, B per height.

    The spectrum of each height is the flat field at every filter frequency plus the series terrain part up to the
    cut-off frequency f_c (last frequency from 1 Hz up whose estimated truncation error is below `tolerance` of the
    largest terrain effect), continued as a constant above."""
    nu = st.numerics
    nh, nf, P = len(st.heights), len(fr), len(px)
    pz = header['pz']
    fa = st.freqs
    V0 = flat_field(st, fa, px, py)                              # (nfa, 6, P)
    t = st.t
    out = {k: np.zeros((nh, len(t), 3, P)) for k in ('E', 'dBdt', 'B')}
    fcs = []
    for ih, H in enumerate(st.heights):
        if H != 0:
            eff = np.abs(dV[ih]).reshape(nf, -1).max(1)
            rel = est[ih] * np.abs(Cr[:, 0]).reshape(nf, -1).max(1) / (eff.max() + 1e-300)
            bad = (rel >= float(nu['tolerance'])) & (fr >= 1.0)
            ic = int(np.argmax(bad)) - 1 if bad.any() else nf - 1
        else:
            ic = nf - 1
        fcs.append(fr[ic])
        V = V0.copy()
        for j, f in enumerate(fa):
            i = int(np.argmin(np.abs(fr - f)))
            V[j] += dV[ih, i] if (abs(fr[i] / f - 1) < 1e-8 and i <= ic) else dV[ih, ic]
        sgn = 1.0 if st.signal == 'off' else -1.0
        out['E'][ih] = timedomain.transient(t, fa, V[:, :3], kind=st.signal)
        out['B'][ih] = timedomain.transient(t, fa, MU0 * V[:, 3:], kind=st.signal)
        out['dBdt'][ih] = -sgn * timedomain.transient(t, fa, MU0 * V[:, 3:], kind='imp')
        log(f"  H = {H:+.0f} m: " + ("flat field" if H == 0 else f"series used up to f_c = {fr[ic]:.3g} Hz, constant above"))
    if 'npz' in st.formats:
        np.savez_compressed(os.path.join(results, 'transients.npz'), t=t, signal=st.signal, f_c=np.array(fcs),
                            units='E V/m, dBdt T/s, B T; 1 A step', **out, **header)
    if 'csv' in st.formats:
        with open(os.path.join(results, 'transients.csv'), 'w') as fh:
            fh.write('H_m,t_s,profile,x_m,y_m,z_m,' + ','.join(['Ex', 'Ey', 'Ez', 'dBx_dt', 'dBy_dt', 'dBz_dt',
                                                               'Bx', 'By', 'Bz']) + '\n')
            for ih, H in enumerate(st.heights):
                for it, tt in enumerate(t):
                    for k in range(P):
                        v = list(out['E'][ih, it, :, k]) + list(out['dBdt'][ih, it, :, k]) + list(out['B'][ih, it, :, k])
                        fh.write(f"{H:g},{tt:.6e},{prof[k]},{px[k]:g},{py[k]:g},{pz[ih, k]:.6g}," +
                                 ','.join(f"{a:.10e}" for a in v) + '\n')
    log(f"written: {results}: " + ', '.join(f'transients.{f}' for f in st.formats if f != 'coefficients'))


def main():
    """Run the series computation of the control file given on the command line (see the module docstring)."""
    st = Config(sys.argv[1])
    results = os.path.join(st.folder, 'results'); os.makedirs(results, exist_ok=True)
    logfile = open(os.path.join(results, 'log.txt'), 'w')

    def log(s):
        """Print a line and append it to the log file."""
        print(s, flush=True); logfile.write(s + '\n'); logfile.flush()
    log(f"# {st.name}  {time.strftime('%Y-%m-%d %H:%M:%S')} (server time, UTC)")
    log(open(st.file).read())
    nu = st.numerics
    px, py, prof = st.receivers()
    fr = st.freqs if st.domain == 'frequency' else st.freqs[st.freqs <= float(nu['fmax_series'])]
    res = None
    if all(H == 0 for H in st.heights):                            # flat surface: no terrain series needed
        log("  all heights 0: flat field only (closed form + layer correction, sphere if present), no terrain series")
        if st.domain == 'frequency':
            Cr = flat_field(st, fr, px, py)[:, None]                # (nf, 1, 6, P)
        else:                                                      # write_transients takes the flat field itself
            Cr = np.zeros((len(fr), 1, 6, len(px)), complex)
        dV = np.zeros((len(st.heights),) + Cr[:, 0].shape, complex); est = np.zeros((len(st.heights), len(fr)))
    elif nu['method'] == 'hankel':
        Cr = coeff_hankel(st, fr, log)                              # (nf, nord+1, 6, P), no periodic box
        res = dict(freqs=fr, C_receivers=Cr, method='hankel')
    else:
        C, x = coeff_fft(st, fr, int(nu['n']), float(nu['dx']), float(nu['lowpass']), int(nu['orders']), log)
        res = dict(freqs=fr, x=x, C=C, method='fft')
        Cr = at_receivers(C, x, px, py, prof)                     # (nf, nord+1, 6, P)
    if res is not None:
        dV, est = terrain_effect(st, Cr, log)
        apply_box_correction(st, fr, dV, res, (px, py, prof), log)
    if 'coefficients' in st.formats and res is None:
        log("  coefficients.npz not written: no terrain series (all heights 0)")
    elif 'coefficients' in st.formats:                             # only on request (see the module docstring)
        np.savez_compressed(os.path.join(results, 'coefficients.npz'), **res)
        log(f"written: {results}: coefficients.npz")
    pz = np.array([[st.height(a, b, H) for a, b in zip(px, py)] for H in st.heights])
    header = dict(heights=np.array(st.heights), px=px, py=py, pz=pz, profile=np.array(prof))
    if st.domain == 'frequency':
        write_fields(st, results, fr, Cr, dV, est, header, prof, log)
    else:
        write_transients(st, results, fr, Cr, dV, est, header, prof, px, py, log)
    logfile.close()


if __name__ == '__main__':
    main()
