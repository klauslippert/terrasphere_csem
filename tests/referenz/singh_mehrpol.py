#!/usr/bin/env python3
"""STUFE 1 — Mehrpol-Referenz im LEITENDEN Vollraum (Singh 1973), koaxiale Schleife Tx = Rx.
Kugel (Radius b, sigma_s) und Schleife in einem homogenen leitenden Vollraum (sigma_h). Verschiebungsstroeme
vernachlaessigt, mu ueberall mu0. Achsensymmetrisch -> nur TE, Ordnung m = 0. Zeitkonvention e^(-i w t).

Frequenzbereich (alles je Grad n exakt):
  Ring (Radius a, Strom I) um den Kugelmittelpunkt bei Abstand c = sqrt(a^2+h^2), Polwinkel alpha (cos = h/c).
  Primaerfeld innen (r < c):  E_phi = sum_n c_n j_n(k_h r) P_n^1(cos th),
      c_n = -k_h w mu0 I c S_n h_n(k_h c),   S_n = (2n+1) sin(alpha) P_n^1(cos alpha) / (2n(n+1))
      (Radial-Green: r^2-Sprung -i w mu0 I S_n c, Wronski j h' - j' h = i/x^2).
  Kugel: Streukoeffizient TE (Mie b_n, mu gleich), Streufeld s_n = -b_n c_n mit h_n^(1)(k_h r).
  Spannung: V_s(w) = 2 pi a * sum_n s_n h_n(k_h c) P_n^1(cos alpha).
Zeitbereich: Ausschalten, V_off(t) = -(2/pi) int_0^inf Re V(w) sin(w t)/w dw  (V(0) = 0), digitaler Filter
  key_201_CosSin_2012 (empymod).
Aufruf: python singh_mehrpol.py  -> Selbsttests."""
import os, sys
import numpy as np
from scipy.special import spherical_jn, spherical_yn, lpmv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MU0 = 4e-7 * np.pi
from empymod import filters as _flt
_F = _flt.key_201_CosSin_2012()


def k_of(sig, w):
    k = np.sqrt(1j * w * MU0 * sig + 0j)
    return np.where(np.imag(k) < 0, -k, k)


def h1(n, z):
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def dh1(n, z):
    return spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True)


def mie_bn(n, kh, ks, b):
    """TE-Streukoeffizient (Bohren & Huffman b_n, mu gleich); m = ks/kh, x = kh b."""
    x, mx = kh * b, ks * b
    m = ks / kh
    psi = lambda nn, z: z * spherical_jn(nn, z)
    dpsi = lambda nn, z: spherical_jn(nn, z) + z * spherical_jn(nn, z, derivative=True)
    xi = lambda nn, z: z * h1(nn, z)
    dxi = lambda nn, z: h1(nn, z) + z * dh1(nn, z)
    num = psi(n, mx) * dpsi(n, x) - m * psi(n, x) * dpsi(n, mx)
    den = psi(n, mx) * dxi(n, x) - m * xi(n, x) * dpsi(n, mx)
    return num / den


def _ratios_h(z, nmax):
    """r_k = h_k(z)/h_{k-1}(z), k = 1..nmax (aufwaerts stabil); h_0 = -i e^{iz}/z."""
    r = np.zeros((nmax + 1,) + np.shape(z), complex)
    r[1] = 1.0 / z - 1j                     # h_1/h_0 geschlossen (ohne Ueberlauf)
    for k in range(2, nmax + 1):
        r[k] = (2 * k - 1) / z - 1.0 / r[k - 1]
    return r


def _ratios_j(z, nmax, extra=40):
    """s_k = j_k(z)/j_{k-1}(z), k = 1..nmax (abwaerts, Kettenbruch)."""
    K = nmax + extra + int(np.max(np.abs(z)))
    sk = np.zeros(np.shape(z), complex)
    out = np.zeros((nmax + 1,) + np.shape(z), complex)
    for k in range(K, 0, -1):
        sk = 1.0 / ((2 * k + 1) / z - sk)
        if k <= nmax:
            out[k] = sk
    return out


def _logderiv_psi(z, nmax, extra=40):
    """D_n(z) = psi_n'(z)/psi_n(z), n = 0..nmax, abwaerts (Bohren & Huffman 4.89)."""
    K = nmax + extra + int(np.max(np.abs(z)))
    D = np.zeros(np.shape(z), complex)
    out = np.zeros((nmax + 1,) + np.shape(z), complex)
    for n in range(K, 0, -1):
        if n <= nmax:
            out[n] = D
        D = n / z - 1.0 / (D + n / z)
    out[0] = D
    return out


def Vs_freq(w, a, h, b, sig_h, sig_s, nmax=30, I=1.0, by_degree=False):
    """V_s(w) der Kugel in der koaxialen Schleife (leitender Vollraum), stabile Verhaeltnisform:
    b_n h_n(kc)^2 = G_n * j_n(x) h_n(kc) * h_n(kc)/h_n(x),  x = k_h b, ohne Ueberlauf."""
    w = np.atleast_1d(w).astype(float)
    c = np.hypot(a, h); ca = h / c; sa = a / c
    kh, ks = k_of(sig_h, w), k_of(sig_s, w)
    x, mx, m = kh * b, ks * b, ks / kh
    zc = kh * c
    rx, rc = _ratios_h(x, nmax), _ratios_h(zc, nmax)
    sx = _ratios_j(x, nmax)
    Dm = _logderiv_psi(mx, nmax)
    # j_0(x) h_0(kc) = sin(x)/x * (-i e^{ikc}/(kc)) ueberlaufsfrei: Exponenten zusammenfassen (c > b)
    jh = -(np.exp(1j * (zc + x)) - np.exp(1j * (zc - x))) / (2j) / x * 1j / zc
    hq = (x / zc) * np.exp(1j * (zc - x))   # h_0(kc)/h_0(x)
    out = np.zeros((nmax, len(w)), complex)
    for n in range(1, nmax + 1):
        jh = jh * sx[n] * rc[n]
        hq = hq * rc[n] / rx[n]
        q = m * Dm[n] + n / x
        G = (q - 1.0 / sx[n]) / (q - 1.0 / rx[n])
        bh2 = G * jh * hq
        P = lpmv(1, n, ca)
        S = (2 * n + 1) * sa * P / (2 * n * (n + 1))
        out[n - 1] = 2 * np.pi * a * kh * w * MU0 * I * c * S * P * bh2
    return (out.sum(0), out) if by_degree else out.sum(0)


def step_off(Ffun, t):
    """V_off(t) = -(2/pi) int Re F(w) sin(wt)/w dw, F(0) = 0."""
    t = np.atleast_1d(t); res = np.zeros(len(t))
    for i, ti in enumerate(t):
        w = _F.base / ti
        res[i] = -(2 / np.pi) * np.sum(np.real(Ffun(w)) / w * _F.sin) / ti
    return res


def Vs_time(t, a, h, b, sig_h, sig_s, nmax=30):
    return step_off(lambda w: Vs_freq(w, a, h, b, sig_h, sig_s, nmax), t)


def _Vs_direkt(w, a, h, b, sig_h, sig_s, nmax):
    w = np.atleast_1d(w).astype(float); c = np.hypot(a, h); ca = h / c; sa = a / c
    kh, ks = k_of(sig_h, w), k_of(sig_s, w); tot = 0
    for n in range(1, nmax + 1):
        P = lpmv(1, n, ca); S = (2 * n + 1) * sa * P / (2 * n * (n + 1)); hn = h1(n, kh * c)
        tot = tot + 2 * np.pi * a * kh * w * MU0 * c * S * P * mie_bn(n, kh, ks, b) * hn**2
    return tot


def _tests():
    A, H, B, SIG = 400.0, 500.0, 225.0, 0.8606
    print("TEST 0  stabile Verhaeltnisform gegen direkte Mie-Formel (harmlose Argumente, n <= 8)")
    w = np.array([10.0, 300.0, 3000.0]); a1 = Vs_freq(w, A, H, B, 1e-2, SIG, nmax=8); a2 = _Vs_direkt(w, A, H, B, 1e-2, SIG, 8)
    print("   rel. Abweichung", np.abs(a1 - a2) / np.abs(a2))
    print("TEST 1  Filter: int_0^inf sin(wt)/(w(1+w^2)) dw = (pi/2)(1 - e^-t)")
    for t in (0.01, 0.3, 3.0):
        w = _F.base / t
        num = np.sum(1 / (w * (1 + w**2)) * _F.sin) / t
        print(f"   t = {t:5.2f}: Filter {num:.8f}   exakt {np.pi/2*(1-np.exp(-t)):.8f}")
    print("\nTEST 2  Quasistatischer Grenzfall (sigma_h -> 1e-9) gegen lee_mehrpol.py (Nabighian, abgenommen)")
    from lee_mehrpol import Vs_mehrpol
    T = np.geomspace(6.62e-4, 8.65e-2, 7)
    ref = Vs_mehrpol(T, A, H, B, SIG, nmax=30)
    vs = Vs_time(T, A, H, B, 1e-9, SIG)
    for i, t in enumerate(T):
        print(f"   t = {t*1e3:7.3f} ms   Singh-Weg {vs[i]: .6e}   Nabighian {ref[i]: .6e}   Verh. {vs[i]/ref[i]:.6f}")


if __name__ == '__main__':
    _tests()
