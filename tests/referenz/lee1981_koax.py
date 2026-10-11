#!/usr/bin/env python3
"""STUFE 2 — Kugel im leitenden HALBRAUM unter koaxialer Schleife, alle Grade, mit Rueckkopplung ueber die
Erdoberflaeche (Physik von Lee 1981, Gl. 15-18, fuer m = 0 / TE; aus geprueften Bausteinen neu aufgebaut).
Zeitkonvention e^(-i w t), Verschiebungsstroeme vernachlaessigt, mu0 ueberall, Luft nichtleitend.
Ursprung Kugelmitte, z nach oben; Erdoberflaeche/Schleife bei z = h (Kugeltiefe), Schleifenradius a.

Bausteine (lee1981_bausteine.py, singh_mehrpol.py):
  auslaufend:  h_n(kR) P_n^1 = C_n int (lam/u) P_n^1(iu/k) J1(lam rho) e^(-uz) dlam,  C_n = (-i)^n / k
  eben->Kugel: J1 e^(ikz z) = sum (2n+1) i^(n-1)/(n(n+1)) P_n^1(kz/k) j_n(kR) P_n^1(cos th)
  Kugel je Grad: s_n = -b_n c_n (Mie TE, stabile Form)
Primaerfeld im Boden: E_phi = i w mu0 I a int lam/(lam+u) J1(lam a) J1(lam rho) e^(-u*Tiefe) dlam
Reflexion (Boden->Luft, TE): r = (u - lam)/(u + lam);  Transmission in Luft/Draht: 1 + r.
System:  c = c^p + R s,  s = -B c   ->   (I + B R) s = -B c^p ;   V_s = sum_n s_n g_n
Skaliert (keine Ueber-/Unterlaeufe): c^ = c k^n, s^ = s k^-(n+1), B^ = b_n k^-(2n+1), R^ = k^n R k^(m+1).
Fuer eine Deckschicht muessen nur r(lam) und die Transmission des Primaerfeldes ersetzt werden."""
import os, sys
import numpy as np
from scipy.special import jv
from scipy.interpolate import CubicSpline
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from singh_mehrpol import k_of, _ratios_h, _ratios_j, _logderiv_psi, _F, MU0


def Pt(nmax, y, k, lam):
    """k^n P_n^1(y/k), n = 1..nmax; y = +-i u. Rekursion in Q_n = k^n P_n(y/k)."""
    Q = [np.ones_like(y), y]
    k2 = k * k
    for n in range(1, nmax):
        Q.append(((2 * n + 1) * y * Q[n] - n * k2 * Q[n - 1]) / (n + 1))
    return np.array([n * (y * Q[n] - k2 * Q[n - 1]) / lam for n in range(1, nmax + 1)])


def Bhat(nmax, k, ks, b):
    """b_n / k^(2n+1) fuer n = 1..nmax (stabile Verhaeltnisform)."""
    x, mx, m = k * b, ks * b, ks / k
    rx = _ratios_h(np.atleast_1d(x), nmax)[:, 0]
    sx = _ratios_j(np.atleast_1d(x), nmax)[:, 0]
    Dm = _logderiv_psi(np.atleast_1d(mx), nmax)[:, 0]
    jh = (1 - np.exp(-2j * x)) / 2 / x          # (j0/h0)/x
    out = np.zeros(nmax, complex)
    for n in range(1, nmax + 1):
        jh = jh * sx[n] / (rx[n] * x * x)
        q = m * Dm[n] + n / x
        G = (q - 1.0 / sx[n]) / (q - 1.0 / rx[n])
        out[n - 1] = G * jh * b**(2 * n + 1)
    return out


def _lamgrid(h, a, npan=80, nq=24, lmax_fac=45.0):
    L = lmax_fac / h
    x, w = np.polynomial.legendre.leggauss(nq)
    e = np.linspace(0, L, npan + 1)
    lam = np.concatenate([0.5 * (e[i+1] - e[i]) * x + 0.5 * (e[i+1] + e[i]) for i in range(npan)])
    wl = np.concatenate([0.5 * (e[i+1] - e[i]) * w for _ in [0] for i in range(npan)])
    return lam, wl


def parts_at(w, a, h, b, sig_h, sig_s, nmax=15, I=1.0, refl=True):
    """Fuer eine Kreisfrequenz w: (c^p, R^, B^, g^) skaliert."""
    k, ks = k_of(sig_h, w), k_of(sig_s, w)
    lam, wl = _lamgrid(h, a)
    u = np.sqrt(lam**2 - k**2 + 0j); u = np.where(np.real(u) < 0, -u, u)
    r = (u - lam) / (u + lam) if refl else np.zeros_like(u)
    Pup, Pdn = Pt(nmax, 1j * u, k, lam), Pt(nmax, -1j * u, k, lam)
    n = np.arange(1, nmax + 1)
    pref = (2 * n + 1) * (1j)**(n - 1) / (n * (n + 1))
    J1a = jv(1, lam * a)
    K0 = 1j * w * MU0 * I * a
    cp = K0 * pref * (Pdn * (wl * lam / (lam + u) * J1a * np.exp(-u * h))).sum(1)
    kC = (-1j)**n
    Rm = (Pdn * (wl * lam / u * r * np.exp(-2 * u * h))) @ Pup.T          # [n, m]
    Rh = pref[:, None] * Rm * kC[None, :]
    g = 2 * np.pi * a * kC * (Pup * (wl * lam / u * (1 + r) * J1a * np.exp(-u * h))).sum(1)
    return cp, Rh, Bhat(nmax, k, ks, b), g


def Vs_freq(w, a, h, b, sig_h, sig_s, nmax=15, refl=True, by_degree=False):
    w = np.atleast_1d(w).astype(float); out = np.zeros(len(w), complex); deg = np.zeros((nmax, len(w)), complex)
    for i, wi in enumerate(w):
        k = k_of(sig_h, wi)
        if np.imag(k) * (h - b) > 60:            # Feld der Kugel an der Oberflaeche ~ e^(-60): null
            continue
        cp, Rh, Bh, g = parts_at(wi, a, h, b, sig_h, sig_s, nmax, refl=refl)
        Mx = np.eye(nmax) + Bh[:, None] * Rh
        s = np.linalg.solve(Mx, -Bh * cp)
        deg[:, i] = s * g; out[i] = deg[:, i].sum()
    return (out, deg) if by_degree else out


def Vs_time(t, a, h, b, sig_h, sig_s, nmax=15, refl=True, nw=260):
    """Ausschaltantwort ueber Spline in log w (Re V), dann digitaler Sinusfilter."""
    t = np.atleast_1d(t)
    wmin, wmax = _F.base[0] / t.max(), _F.base[-1] / t.min()
    wg = np.geomspace(wmin, wmax, nw)
    Vg = Vs_freq(wg, a, h, b, sig_h, sig_s, nmax, refl)
    sp = CubicSpline(np.log(wg), np.real(Vg))
    res = np.zeros(len(t))
    for i, ti in enumerate(t):
        ww = _F.base / ti
        res[i] = -(2 / np.pi) * np.sum(sp(np.log(ww)) / ww * _F.sin) / ti
    return res


def _tests():
    import singh_mehrpol as S1
    from lee_mehrpol import Vs_mehrpol
    from lee1983_referenz import V_p as lee_Vp
    A, H, B, SIG = 400.0, 500.0, 225.0, 0.8606
    print("TEST 1  Freiraum-Grenzfall im Frequenzbereich: Halbraum (sigma_h 1e-9) gegen Stufe 1 (Vollraum 1e-9)")
    for w in (10.0, 1e3, 1e5):
        v2 = Vs_freq(w, A, H, B, 1e-9, SIG)[0]; v1 = S1.Vs_freq(w, A, H, B, 1e-9, SIG)[0]
        print(f"   w = {w:8.0f}: Stufe 2 {v2: .6e}   Stufe 1 {v1: .6e}   rel {abs(v2-v1)/abs(v1):.1e}")
    print("\nTEST 2  Reziprozitaet im leitenden Halbraum: g^_n = -4 pi n(n+1) c^_n/((2n+1) w mu0)")
    for sh, w in ((1e-2, 300.0), (1e-3, 3e3)):
        cp, Rh, Bh, g = parts_at(w, A, H, B, sh, SIG)
        n = np.arange(1, 16); rec = -4 * np.pi * n * (n + 1) * cp / ((2 * n + 1) * w * MU0)
        print(f"   sigma_h {sh:.0e}, w {w:6.0f}: max rel. Abw. ueber n = 1..15: {np.max(np.abs(g-rec)/np.abs(rec)):.1e}")
    print("\nTEST 3  Zeitbereich, Freiraum-Grenzfall gegen lee_mehrpol.py (Nabighian)")
    T = np.geomspace(6.62e-4, 8.65e-2, 6)
    v = Vs_time(T, A, H, B, 1e-9, SIG); ref = Vs_mehrpol(T, A, H, B, SIG)
    for i, t in enumerate(T):
        print(f"   t = {t*1e3:7.3f} ms   Stufe 2 {v[i]: .6e}   Nabighian {ref[i]: .6e}   Verh. {v[i]/ref[i]:.5f}")
    print("\nTEST 4  Lees eigene Zahlen (Lee 1983 Abb. 7/8): Schleife 100 m, Kugel 50 m, Boden 0,01 S/m, V_total")
    for (sig_s, tt, refs) in ((5.0, 10e-3, (6.28e-6, 3.63e-6, 2.10e-6)), (2.0, 5e-3, (1.17e-5, 1.0e-5, 9.2e-6))):
        for D, rf in zip((75.0, 100.0, 150.0), refs):
            vs = Vs_time(np.array([tt]), 100.0, D, 50.0, 0.01, sig_s)[0]
            vt = vs + lee_Vp(100.0, 1.0, 0.01, tt)
            print(f"   Kugel {sig_s} S/m, t {tt*1e3:4.0f} ms, D {D:5.0f} m: Stufe 2 {vt:.3e}   Lee {rf:.3e}   Verh. {vt/rf:.3f}")


if __name__ == '__main__':
    _tests()
