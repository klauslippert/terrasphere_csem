#!/usr/bin/env python3
"""Reference 3: multipole solution for a conducting sphere in FREE SPACE, excited and measured by a coaxial loop
(Tx = Rx), quasi-static, switch-off of 1 A.  After Nabighian 1970 (Geophysics 35, 303-309, eqs. 18/19/23,
K = mu_i/mu_0 = 1).  Lee 1983 contains only the degree n = 1 of it.

Coordinates: origin at the sphere centre, z axis towards the loop; loop plane at z = h (= depth d), radius a.
c = sqrt(a^2 + h^2) = distance centre -> wire, cos(alpha) = h/c.

1. Primary field inside (r < c) as the potential U_inc = sum_n C_n r^n P_n(cos th), H = -grad U.
   Axial field H_z(z) = I a^2 / (2 (a^2 + (h-z)^2)^(3/2)) = I a^2/(2 c^3) sum_k G_k^(3/2)(cos alpha) (z/c)^k
   (Gegenbauer generating function), hence h_k = I a^2 G_k(cos alpha) / (2 c^(k+3)) and C_n = -h_{n-1}/n.
2. Sphere (Nabighian eq. 18): outside U_s = sum_n D_n F_n(t) r^-(n+1) P_n, D_n = -n/(n+1) C_n b^(2n+1),
   F_n(t) = 2(2n+1) sum_s exp(-x_s^2 t/beta^2)/x_s^2, x_s the zeros of j_{n-1}, beta^2 = mu0 sigma b^2.
3. Flux through the loop area: B_z = mu0 (n+1) D_n F_n P_{n+1}(cos th) / r^(n+2)
   (identity d/dz[P_n r^-(n+1)] = -(n+1) P_{n+1} r^-(n+2)), integrated numerically over the disc.
4. V(t) = -d Phi/dt.
Usage: python nabighian_multipole.py  -> self checks."""
import os, sys
import numpy as np
from scipy.special import eval_gegenbauer, eval_legendre, spherical_jn, ellipk, ellipe
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MU0 = 4e-7 * np.pi


def coeff_inc(a, h, nmax, I=1.0):
    """C_n (n = 1..nmax) of the primary potential of the loop about the sphere centre."""
    c = np.hypot(a, h); ca = h / c
    C = np.zeros(nmax + 1)
    for n in range(1, nmax + 1):
        hk = I * a**2 * eval_gegenbauer(n - 1, 1.5, ca) / (2.0 * c**(n + 2))
        C[n] = -hk / n
    return C


def Hz_series(C, r, th):
    """H_z from the series: d/dz (r^n P_n) = n r^(n-1) P_{n-1}."""
    ct = np.cos(th); out = 0.0
    for n in range(1, len(C)):
        out = out - C[n] * n * r**(n - 1) * eval_legendre(n - 1, ct)
    return out


def Hz_biot(a, h, I, rho, z):
    """Exact H_z of the circular loop (plane z = h, axis through rho = 0), elliptic integrals."""
    zz = z - h
    k2 = 4 * a * rho / ((a + rho)**2 + zz**2)
    K, E = ellipk(k2), ellipe(k2)
    pre = I / (2 * np.pi * np.sqrt((a + rho)**2 + zz**2))
    return pre * (K + (a**2 - rho**2 - zz**2) / ((a - rho)**2 + zz**2) * E)


def roots_jn(nu, count):
    """First `count` positive zeros of j_nu (spherical Bessel function)."""
    if nu == 0:
        return np.pi * np.arange(1, count + 1)
    xs, x0 = [], nu + 0.5
    step = 0.05
    f0 = spherical_jn(nu, x0)
    x = x0
    while len(xs) < count:
        x1 = x + step; f1 = spherical_jn(nu, x1)
        if f0 == 0:
            xs.append(x)
        elif f0 * f1 < 0:
            xs.append(brentq(lambda t: spherical_jn(nu, t), x, x1, xtol=1e-14))
        x, f0 = x1, f1
    return np.array(xs)


def dFdt(n, t, beta2, roots):
    """d/dt F_n(t), F_n = 2(2n+1) sum exp(-x^2 t/beta^2)/x^2  ->  -2(2n+1)/beta^2 sum exp(-x^2 t/beta^2)."""
    t = np.atleast_1d(t)
    return -2.0 * (2 * n + 1) / beta2 * np.exp(-np.outer(t, roots**2) / beta2).sum(1)


def flux_geom(n, a, h, D, nq=400):
    """Flux through the loop area per unit F_n for degree n with coefficient D (without F_n)."""
    x, w = np.polynomial.legendre.leggauss(nq)
    rho = 0.5 * a * (x + 1); wq = 0.5 * a * w
    r = np.hypot(rho, h); ct = h / r
    Bz = MU0 * (n + 1) * D * eval_legendre(n + 1, ct) / r**(n + 2)
    return 2 * np.pi * np.sum(Bz * rho * wq)


def Vs_multipole(t, a, h, b, sig, nmax=30, nroots=400, I=1.0, by_degree=False):
    """V_s(t) of the sphere in the coaxial loop, sum of the degrees 1..nmax.  Sign: switch-off."""
    beta2 = MU0 * sig * b**2
    C = coeff_inc(a, h, nmax, I)
    parts = []
    for n in range(1, nmax + 1):
        D = -n / (n + 1.0) * C[n] * b**(2 * n + 1)
        g = flux_geom(n, a, h, D)
        parts.append(-g * dFdt(n, t, beta2, roots_jn(n - 1, nroots)))
    parts = np.array(parts)
    return (parts.sum(0), parts) if by_degree else parts.sum(0)


def _tests():
    """Self checks: primary field series against Biot-Savart, F_n(0) = 1, degree 1 against Lee 1983."""
    A, H, B, SIG = 400.0, 500.0, 225.0, 0.8606
    print("TEST 1  primary field: series (Gegenbauer) against Biot-Savart (elliptic), points inside the sphere")
    for nmax in (5, 15, 40):
        C = coeff_inc(A, H, nmax)
        err = 0.0
        for r in (0.3 * B, 0.7 * B, 1.0 * B):
            for th in (0.0, 0.6, 1.3, 2.0, np.pi - 1e-9):
                rho, z = r * np.sin(th), r * np.cos(th)
                hb = Hz_biot(A, H, 1.0, max(rho, 1e-9), z)
                err = max(err, abs(Hz_series(C, r, th) - hb) / abs(hb))
        print(f"   n_max = {nmax:3d}: max rel. error of H_z = {err:.2e}")
    print("\nTEST 2  decay functions: F_n(0) = 2(2n+1) sum 1/x_s^2 must be 1 (Nabighian eq. 20, K = 1)")
    for n in (1, 2, 3, 6):
        x = roots_jn(n - 1, 20000)
        print(f"   n = {n}: F_n(0) = {2*(2*n+1)*np.sum(1/x**2):.6f}   first root {x[0]:.4f}  "
              f"tau_n1/tau_1 = {np.pi**2/x[0]**2:.3f}")
    print("\nTEST 3  degree n = 1 alone against Lee's term t1 (Lee 1983 eq. 15, rho = 0, same physics)")
    from lee1983_reference import V_s as lee_Vs
    T = np.geomspace(6.62e-4, 8.65e-2, 9)
    _, parts = Vs_multipole(T, A, H, B, SIG, nmax=1, by_degree=True)
    for i, t in enumerate(T):
        t1 = lee_Vs(A, B, H, 0.0, 1.0, 1e-4, SIG, t)['t1']
        print(f"   t = {t*1e3:7.3f} ms   n=1: {parts[0][i]: .6e}   Lee t1: {t1: .6e}   ratio {parts[0][i]/t1:.6f}")


if __name__ == '__main__':
    _tests()
