#!/usr/bin/env python3
"""Series computation of one computation folder (KH-026/028).  Reads <folder>/steuer.toml, writes <folder>/ergebnis/:
  protokoll.txt      control file, grids, timing, convergence, box correction
  koeffizienten.npz  series coefficients on the summit profiles (always; basis of every other output)
  frequency domain ([frequenzen]):  felder.npz / felder.csv  (as chosen in [ausgabe] formate)
      E, H (nh, nf, 3, P) complex, e^{-iwt}, components (x, y, z), z up, fields on the terrain (earth side);
      px, py, pz, profile, fehler (nh, nf) estimated truncation error relative to the largest flat E resp. H
  time domain ([zeit], signal aus|ein):  transienten.npz / transienten.csv
      E (V/m), dBdt (T/s), B (T) (nh, nt, 3, P) for a 1 A step of the chosen sign; t; px, py, pz, profile; f_c per H
Spectrum per height H: V = flat field (closed form + layer correction + sphere, at every frequency) + terrain part
S(H) - c_0 of the series, (i) box-corrected for f <= box_fmax by [Richardson(L_b, L_c) - L_a] of coarse runs
(20 m grid, boxes 5.12/10.24/20.48 km, 12 orders; box error ~ L^-2, gausshill GH-018/022), (ii) in the time domain
continued as a constant above f_c, the last frequency (from 1 Hz up) whose estimated truncation error is below
`toleranz` of the largest terrain effect (gausshill GH-017).
Usage: python3 rechne.py <folder>"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from terrasphere_csem import huegel, halbraum, zeit, MU0
from terrasphere_csem.steuerung import Steuerung
from terrasphere_csem.huegel import summe_optimal

KOMP_E, KOMP_H = ['Ex', 'Ey', 'Ez'], ['Hx', 'Hy', 'Hz']


def koeff(st, freqs, n, dx, kc, nord, log):
    g = st.gitter(n=n, dx=dx)
    kg = st.kugel_kopplung(g)
    t0 = time.time()
    CE, CH = huegel.koeffizienten(freqs, g, st.schichten, nord=nord, kc=kc, kugel=kg)
    log(f"  grid N = {n}, dx = {dx} m (box {n * dx / 1000:.2f} km), {nord} orders, low-pass {kc}: "
        f"{len(freqs)} frequencies in {time.time() - t0:.0f} s")
    return np.concatenate([CE, CH], axis=2), g.x                    # (nf, nord+1, 6, 2, N)


def koeff_hankel(st, freqs, log):
    from terrasphere_csem import huegel_hankel
    g = st.hankelgitter()
    kg = st.kugel_kopplung(g)
    t0 = time.time()
    CE, CH = huegel_hankel.koeffizienten(freqs, g, st.schichten, nord=int(st.num['ordnungen']), kugel=kg)
    log(f"  Hankel series: |m| <= {g.M}, {len(g.r)} r nodes (0..{g.r_max:.0f} m), {len(g.lam)} lambda nodes "
        f"(0..{g.lam[-1]:.3f} rad/m), {st.num['ordnungen']} orders: {len(freqs)} frequencies in {time.time() - t0:.0f} s")
    return np.concatenate([CE, CH], axis=2)                          # (nf, nord+1, 6, P) at the receivers


def an_empfaengern(C, x, px, py, prof):
    """C (..., 6, 2, N) on the lines -> (..., 6, P) at the receivers."""
    out = []
    for X, Y, p in zip(px, py, prof):
        lin, v = (1, X) if p == 'x' else (0, Y)
        i = int(np.argmin(np.abs(x - v))); assert abs(x[i] - v) < 1e-6, "receiver not on the grid"
        out.append(C[..., lin, i])
    return np.stack(out, axis=-1)


def flachfeld(st, freqs, px, py):
    """Flat-earth field at the receivers for all frequencies: (nf, 6, P) (E_z = 0 on the flat surface)."""
    q = st.quellen()
    P = halbraum.oberflaeche_schichten(px, py, freqs, st.schichten, q)
    kg = st.kugel_kopplung(st.hankelgitter() if st.num['verfahren'] == 'hankel' else st.gitter()) if st.kugel else None
    out = np.zeros((len(freqs), 6, len(px)), complex)
    for k, c in [(0, 'Ex'), (1, 'Ey'), (3, 'Hx'), (4, 'Hy'), (5, 'Hz')]:
        out[:, k] = P[c]
    if kg is not None:
        for i, f in enumerate(freqs):
            pk = kg.oberflaeche(2 * np.pi * f, q, px, py)
            for k, c in [(0, 'Ex'), (1, 'Ey'), (3, 'Hx'), (4, 'Hy'), (5, 'Hz')]:
                out[i, k] += pk[c]
    return out


def main():
    st = Steuerung(sys.argv[1])
    erg = os.path.join(st.ordner, 'ergebnis'); os.makedirs(erg, exist_ok=True)
    prot = open(os.path.join(erg, 'protokoll.txt'), 'w')

    def log(s):
        print(s, flush=True); prot.write(s + '\n'); prot.flush()
    log(f"# {st.name}  {time.strftime('%Y-%m-%d %H:%M:%S')} (server time, UTC)")
    log(open(os.path.join(st.ordner, 'steuer.toml')).read())
    nu = st.num
    px, py, prof = st.empfaenger()
    fr = st.freqs if st.bereich == 'frequenz' else st.freqs[st.freqs <= float(nu['fmax_reihe'])]
    if nu['verfahren'] == 'hankel':
        Cr = koeff_hankel(st, fr, log)                              # (nf, nord+1, 6, P), no periodic box
        res = dict(freqs=fr, C_empfaenger=Cr, verfahren='hankel')
    else:
        C, x = koeff(st, fr, int(nu['n']), float(nu['dx']), float(nu['tiefpass']), int(nu['ordnungen']), log)
        res = dict(freqs=fr, x=x, C=C, verfahren='fft')
        Cr = an_empfaengern(C, x, px, py, prof)                     # (nf, nord+1, 6, P)
    nh, nf, P = len(st.hoehen), len(fr), len(px)
    # terrain part per height, error estimate
    dV = np.zeros((nh, nf, 6, P), complex); est = np.zeros((nh, nf))
    for ih, H in enumerate(st.hoehen):
        for i in range(nf):
            S, err = summe_optimal(Cr[i], H)
            dV[ih, i] = S - Cr[i, 0]
            rE, rH = np.abs(Cr[i, 0, :3]).max(), np.abs(Cr[i, 0, 3:]).max()
            est[ih, i] = max(err[:3].max() / rE, err[3:].max() / rH) if H != 0 else 0.0
        log(f"  H = {H:+.0f} m: estimated truncation error (relative to the flat field), max {est[ih].max():.1e}")
    # box correction
    fl = fr[fr <= float(nu['box_fmax'])]
    if nu['verfahren'] == 'fft' and nu['boxkorrektur'] and len(fl) and any(H != 0 for H in st.hoehen):
        Cb = []
        for nb in nu['box_n']:
            c, xb = koeff(st, fl, int(nb), float(nu['box_dx']), float(nu['box_tiefpass']), int(nu['box_ordnungen']), log)
            Cb.append(an_empfaengern(c, xb, px, py, prof)); res[f'C_box{nb}'] = c; res[f'x_box{nb}'] = xb
        L = [int(nb) * float(nu['box_dx']) for nb in nu['box_n']]
        for ih, H in enumerate(st.hoehen):
            if H == 0:
                continue
            m = []
            for jf, f in enumerate(fl):
                i = int(np.argmin(np.abs(fr - f)))
                d = [summe_optimal(cb[jf], H)[0] - cb[jf][0] for cb in Cb]
                corr = (L[2] ** 2 * d[2] - L[1] ** 2 * d[1]) / (L[2] ** 2 - L[1] ** 2) - d[0]
                if st.bereich == 'zeit':                            # fade out towards box_fmax (seam, GH-022)
                    u = np.clip(np.log(f / 3.0) / np.log(float(nu['box_fmax']) / 3.0), 0.0, 1.0)
                    corr = corr * np.cos(0.5 * np.pi * u) ** 2
                m.append(np.abs(corr).max() / (np.abs(dV[ih, i]).max() + 1e-300))
                dV[ih, i] = dV[ih, i] + corr
            log(f"  H = {H:+.0f} m: box correction / terrain effect, max over f <= {nu['box_fmax']} Hz: {max(m):.1e}")
        log(f"  box correction: Richardson {L[1]:.0f}/{L[2]:.0f} m minus {L[0]:.0f} m")
    np.savez_compressed(os.path.join(erg, 'koeffizienten.npz'), **res)
    pz = np.array([[st.h(a, b, H) for a, b in zip(px, py)] for H in st.hoehen])
    kopf = dict(hoehen=np.array(st.hoehen), px=px, py=py, pz=pz, profil=np.array(prof))
    if st.bereich == 'frequenz':
        F = Cr[None, :, 0] + dV                                     # flat field (exact at the receivers) + terrain part
        if 'npz' in st.formate:
            np.savez_compressed(os.path.join(erg, 'felder.npz'), E=F[:, :, :3], H=F[:, :, 3:], freqs=fr, fehler=est,
                                zeitkonvention='exp(-iwt)', einheiten='E V/m, H A/m', **kopf)
        if 'csv' in st.formate:
            with open(os.path.join(erg, 'felder.csv'), 'w') as fh:
                fh.write('H_m,f_Hz,profil,x_m,y_m,z_m,' + ','.join(f'{c}_{p}' for c in KOMP_E + KOMP_H
                                                                  for p in ['re', 'im']) + '\n')
                for ih, H in enumerate(st.hoehen):
                    for i, f in enumerate(fr):
                        for k in range(P):
                            fh.write(f"{H:g},{f:g},{prof[k]},{px[k]:g},{py[k]:g},{pz[ih, k]:.6g}," +
                                     ','.join(f"{a.real:.10e},{a.imag:.10e}" for a in F[ih, i, :, k]) + '\n')
        log(f"written: {erg}: koeffizienten.npz, " + ', '.join(f'felder.{f}' for f in st.formate))
    else:
        fa = st.freqs
        V0 = flachfeld(st, fa, px, py)                              # (nfa, 6, P)
        t = st.t
        out = {k: np.zeros((nh, len(t), 3, P)) for k in ('E', 'dBdt', 'B')}
        fcs = []
        for ih, H in enumerate(st.hoehen):
            if H != 0:
                eff = np.abs(dV[ih]).reshape(nf, -1).max(1)
                rel = est[ih] * np.abs(Cr[:, 0]).reshape(nf, -1).max(1) / (eff.max() + 1e-300)
                bad = (rel >= float(nu['toleranz'])) & (fr >= 1.0)
                ic = int(np.argmax(bad)) - 1 if bad.any() else nf - 1
            else:
                ic = nf - 1
            fcs.append(fr[ic])
            V = V0.copy()
            for j, f in enumerate(fa):
                i = int(np.argmin(np.abs(fr - f)))
                V[j] += dV[ih, i] if (abs(fr[i] / f - 1) < 1e-8 and i <= ic) else dV[ih, ic]
            sgn = 1.0 if st.signal == 'aus' else -1.0
            out['E'][ih] = zeit.transient(t, fa, V[:, :3], art=st.signal)
            out['B'][ih] = zeit.transient(t, fa, MU0 * V[:, 3:], art=st.signal)
            out['dBdt'][ih] = -sgn * zeit.transient(t, fa, MU0 * V[:, 3:], art='imp')
            log(f"  H = {H:+.0f} m: series used up to f_c = {fr[ic]:.3g} Hz, constant above")
        if 'npz' in st.formate:
            np.savez_compressed(os.path.join(erg, 'transienten.npz'), t=t, signal=st.signal, f_c=np.array(fcs),
                                einheiten='E V/m, dBdt T/s, B T; 1 A step', **out, **kopf)
        if 'csv' in st.formate:
            with open(os.path.join(erg, 'transienten.csv'), 'w') as fh:
                fh.write('H_m,t_s,profil,x_m,y_m,z_m,' + ','.join(['Ex', 'Ey', 'Ez', 'dBx_dt', 'dBy_dt', 'dBz_dt',
                                                                   'Bx', 'By', 'Bz']) + '\n')
                for ih, H in enumerate(st.hoehen):
                    for it, tt in enumerate(t):
                        for k in range(P):
                            v = list(out['E'][ih, it, :, k]) + list(out['dBdt'][ih, it, :, k]) + list(out['B'][ih, it, :, k])
                            fh.write(f"{H:g},{tt:.6e},{prof[k]},{px[k]:g},{py[k]:g},{pz[ih, k]:.6g}," +
                                     ','.join(f"{a:.10e}" for a in v) + '\n')
        log(f"written: {erg}: koeffizienten.npz, " + ', '.join(f'transienten.{f}' for f in st.formate))
    prot.close()


if __name__ == '__main__':
    main()
