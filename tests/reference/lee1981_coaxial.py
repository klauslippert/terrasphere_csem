#!/usr/bin/env python3
"""Reference 2: independent reference implementation of the equations of Lee (1981) for a sphere in a conductive
half-space below a coaxial loop (all degrees n, order m = 0, TE, with the feedback through the earth surface), kept
for the verification tests of the sphere model (eqs. 15-18 of Lee 1981, rebuilt from checked building blocks).
Convention: time dependence e^(-i w t), displacement currents neglected, mu0 everywhere, insulating air.
Geometry: origin at the sphere centre, z up; earth surface and loop at z = h (sphere depth), loop radius a.
Units: lengths in m, angular frequency w in rad/s, conductivities in S/m, voltages in V per A of loop current.

Building blocks (this module and singh_multipole.py):
  outgoing:      h_n(kR) P_n^1 = C_n int (lam/u) P_n^1(iu/k) J1(lam rho) e^(-uz) dlam,  C_n = (-i)^n / k
  plane -> sphere: J1 e^(ikz z) = sum (2n+1) i^(n-1)/(n(n+1)) P_n^1(kz/k) j_n(kR) P_n^1(cos th)
  sphere per degree: s_n = -b_n c_n (Mie TE, stable form)
Primary field in the ground: E_phi = i w mu0 I a int lam/(lam+u) J1(lam a) J1(lam rho) e^(-u*depth) dlam
Reflection (ground -> air, TE): r = (u - lam)/(u + lam);  transmission into air/wire: 1 + r.
System:  c = c^p + R s,  s = -B c   ->   (I + B R) s = -B c^p ;   V_s = sum_n s_n g_n
Scaled (no over-/underflow): c^ = c k^n, s^ = s k^-(n+1), B^ = b_n k^-(2n+1), R^ = k^n R k^(m+1).
For a cover layer only r(lam) and the transmission of the primary field have to be replaced."""
import os, sys
import numpy as np
from scipy.special import jv
from scipy.interpolate import CubicSpline
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from singh_multipole import k_of, _ratios_h, _ratios_j, _logderiv_psi, _F, MU0


def Pt(nmax, y, k, lam):
    """k^n P_n^1(y/k), n = 1..nmax, for y = +-i u (arrays over lam); computed by recursion in Q_n = k^n P_n(y/k)."""
    Q = [np.ones_like(y), y]
    k2 = k * k
    for n in range(1, nmax):
        Q.append(((2 * n + 1) * y * Q[n] - n * k2 * Q[n - 1]) / (n + 1))
    return np.array([n * (y * Q[n] - k2 * Q[n - 1]) / lam for n in range(1, nmax + 1)])


def Bhat(nmax, k, ks, b):
    """Mie coefficients b_n / k^(2n+1) (TE, mu_r = 1), n = 1..nmax, in the stable ratio form.

    k, ks: wavenumbers of host and sphere [1/m]; b: sphere radius [m]. Returns a complex array of length nmax."""
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
    """Composite Gauss-Legendre grid in lam on [0, lmax_fac/h]: (lam, weights)."""
    L = lmax_fac / h
    x, w = np.polynomial.legendre.leggauss(nq)
    e = np.linspace(0, L, npan + 1)
    lam = np.concatenate([0.5 * (e[i+1] - e[i]) * x + 0.5 * (e[i+1] + e[i]) for i in range(npan)])
    wl = np.concatenate([0.5 * (e[i+1] - e[i]) * w for _ in [0] for i in range(npan)])
    return lam, wl


def parts_at(w, a, h, b, sig_h, sig_s, nmax=15, I=1.0, refl=True):
    """For one angular frequency w the scaled quantities (c^p, R^, B^, g^), each over the degrees n = 1..nmax.

    c^p: regular-field coefficients of the loop at the sphere; R^: reflection matrix sphere -> earth surface ->
    sphere; B^: Mie coefficients; g^: receiver weights of the outgoing waves. a: loop radius, h: sphere depth, b:
    sphere radius [m]; sig_h, sig_s: conductivities of host and sphere [S/m]; I: loop current [A]; refl=False
    switches the earth surface off (full space)."""
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
    """Scattered voltage V_s(w) [V/A] of the sphere (radius b, depth h) in the coaxial loop (radius a); with
    by_degree=True also the contributions of the single degrees."""
    w = np.atleast_1d(w).astype(float); out = np.zeros(len(w), complex); deg = np.zeros((nmax, len(w)), complex)
    for i, wi in enumerate(w):
        k = k_of(sig_h, wi)
        if np.imag(k) * (h - b) > 60:            # field of the sphere at the surface ~ e^(-60): zero
            continue
        cp, Rh, Bh, g = parts_at(wi, a, h, b, sig_h, sig_s, nmax, refl=refl)
        Mx = np.eye(nmax) + Bh[:, None] * Rh
        s = np.linalg.solve(Mx, -Bh * cp)
        deg[:, i] = s * g; out[i] = deg[:, i].sum()
    return (out, deg) if by_degree else out


def Vs_time(t, a, h, b, sig_h, sig_s, nmax=15, refl=True, nw=260):
    """Switch-off voltage V_off(t) [V/A] of Vs_freq at the times t [s]: cubic spline of Re V in log w on nw points,
    then the 201-point digital sine filter."""
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
    """Self checks against the free-space reference and Lee's published numbers (need the modules
    lee_mehrpol and lee1983_referenz, which are not part of this repository)."""
    import singh_multipole as S1
    from lee_mehrpol import Vs_multipole
    from lee1983_referenz import V_p as lee_Vp
    A, H, B, SIG = 400.0, 500.0, 225.0, 0.8606
    print("TEST 1  free-space limit in the frequency domain: half-space (sigma_h 1e-9) against reference 1 (full space 1e-9)")
    for w in (10.0, 1e3, 1e5):
        v2 = Vs_freq(w, A, H, B, 1e-9, SIG)[0]; v1 = S1.Vs_freq(w, A, H, B, 1e-9, SIG)[0]
        print(f"   w = {w:8.0f}: reference 2 {v2: .6e}   reference 1 {v1: .6e}   rel {abs(v2-v1)/abs(v1):.1e}")
    print("\nTEST 2  reciprocity in the conductive half-space: g^_n = -4 pi n(n+1) c^_n/((2n+1) w mu0)")
    for sh, w in ((1e-2, 300.0), (1e-3, 3e3)):
        cp, Rh, Bh, g = parts_at(w, A, H, B, sh, SIG)
        n = np.arange(1, 16); rec = -4 * np.pi * n * (n + 1) * cp / ((2 * n + 1) * w * MU0)
        print(f"   sigma_h {sh:.0e}, w {w:6.0f}: max rel. deviation over n = 1..15: {np.max(np.abs(g-rec)/np.abs(rec)):.1e}")
    print("\nTEST 3  time domain, free-space limit against lee_mehrpol.py (Nabighian)")
    T = np.geomspace(6.62e-4, 8.65e-2, 6)
    v = Vs_time(T, A, H, B, 1e-9, SIG); ref = Vs_multipole(T, A, H, B, SIG)
    for i, t in enumerate(T):
        print(f"   t = {t*1e3:7.3f} ms   reference 2 {v[i]: .6e}   Nabighian {ref[i]: .6e}   ratio {v[i]/ref[i]:.5f}")
    print("\nTEST 4  Lee's own numbers (Lee 1983, Figs. 7/8): loop 100 m, sphere 50 m, ground 0.01 S/m, V_total")
    for (sig_s, tt, refs) in ((5.0, 10e-3, (6.28e-6, 3.63e-6, 2.10e-6)), (2.0, 5e-3, (1.17e-5, 1.0e-5, 9.2e-6))):
        for D, rf in zip((75.0, 100.0, 150.0), refs):
            vs = Vs_time(np.array([tt]), 100.0, D, 50.0, 0.01, sig_s)[0]
            vt = vs + lee_Vp(100.0, 1.0, 0.01, tt)
            print(f"   sphere {sig_s} S/m, t {tt*1e3:4.0f} ms, D {D:5.0f} m: reference 2 {vt:.3e}   Lee {rf:.3e}   ratio {vt/rf:.3f}")


if __name__ == '__main__':
    _tests()
