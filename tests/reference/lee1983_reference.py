#!/usr/bin/env python3
"""Reference 4: Lee 1983 (Geophys. Prospect. 31, 766-781): transient response of a conducting sphere in a
conducting half-space, circular loop on the surface, step current.

Notation as in Lee:
  a   loop radius [m]               I0  transmitter current [A]
  b   sphere radius [m]             d   depth of the sphere centre [m]
  rho horizontal distance loop axis <-> sphere axis [m]
  s2  conductivity of the host [S/m]      s3  conductivity of the sphere [S/m]
  beta^2 = s3*mu0*b^2 ,  alpha^2 = s2/s3 ,  k^2 = 4*a*rho/(d^2+(a+rho)^2)

This module implements the building blocks SEPARATELY and checks them against each other, because the formulas
were read from the PDF.  The closed forms of I1, I2, I3 are checked against direct quadrature of the Hankel
integrals; J and C are compared in their two expansions (large/small beta^2/t) in the overlap range.

Usage:  python lee1983_reference.py          -> self checks
"""
import numpy as np
from scipy.special import ellipk, ellipe, jv, gamma
from scipy.integrate import quad

MU0 = 4e-7 * np.pi


# --------------------------------------------------------------------------- #
# Hankel integrals I1, I2, I3  (Lee eqs. 8, 9, 10)
# --------------------------------------------------------------------------- #
def kmod2(a, rho, d):
    """k^2 = 4 a rho / (d^2 + (a+rho)^2)."""
    return 4.0 * a * rho / (d ** 2 + (a + rho) ** 2)


def I1_closed(a, rho, d):
    """I1 = a^2 int_0^inf exp(-l d) l J1(l a) J0(l rho) dl."""
    if rho == 0.0:
        return a ** 3 / (a ** 2 + d ** 2) ** 1.5
    k2 = kmod2(a, rho, d); k = np.sqrt(k2)
    K, E = ellipk(k2), ellipe(k2)
    return a ** 2 * (k ** 3 * (a ** 2 - rho ** 2 - d ** 2) * E
                     / (8 * np.pi * (a * rho) ** 1.5 * a * (1 - k2))
                     + k * K / (2 * np.pi * (a * rho) ** 0.5 * a))


def I2_closed(a, rho, d):
    """I2 = a^2 int_0^inf exp(-l d) J1(l rho) J1(l a) l dl."""
    if rho == 0.0:
        return 0.0
    k2 = kmod2(a, rho, d); k = np.sqrt(k2)
    K, E = ellipk(k2), ellipe(k2)
    return (d * k * a ** 2 / (2 * np.pi * (a * rho) ** 1.5 * (1 - k2))
            * ((1 - 0.5 * k2) * E - (1 - k2) * K))


def I3_closed(a, rho, d):
    """I3 = a int_0^inf J1(l rho) J1(l a) exp(-l d) dl."""
    if rho == 0.0:
        return 0.0
    k2 = kmod2(a, rho, d); k = np.sqrt(k2)
    K, E = ellipk(k2), ellipe(k2)
    return 2 * a / (np.pi * k * (a * rho) ** 0.5) * ((1 - 0.5 * k2) * K - E)


def _hankel(f, d, lmax_fac=400.0):
    """int_0^inf f(l) exp(-l d) dl, piecewise over intervals of the order of the zero spacing."""
    lmax = lmax_fac / max(d, 1e-9)
    n = 4000
    edges = np.linspace(0.0, lmax, n + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        v, _ = quad(f, lo, hi, limit=200, epsabs=1e-16, epsrel=1e-11)
        tot += v
    return tot


def I1_quad(a, rho, d):
    """I1 by direct quadrature."""
    return a ** 2 * _hankel(lambda l: np.exp(-l * d) * l * jv(1, l * a) * jv(0, l * rho), d)


def I2_quad(a, rho, d):
    """I2 by direct quadrature."""
    return a ** 2 * _hankel(lambda l: np.exp(-l * d) * l * jv(1, l * rho) * jv(1, l * a), d)


def I3_quad(a, rho, d):
    """I3 by direct quadrature."""
    return a * _hankel(lambda l: np.exp(-l * d) * jv(1, l * rho) * jv(1, l * a), d)


# --------------------------------------------------------------------------- #
# J and C  (Lee eqs. 13/14, appendix A.5/A.9 and A.15/A.16)
# --------------------------------------------------------------------------- #
def J_series(beta, t, nmax=4000):
    """Eq. (13) = (A.5): fast for LARGE beta^2/t."""
    n = np.arange(1, nmax + 1)
    b2 = beta ** 2
    e = np.exp(-b2 * n ** 2 / t)
    s = np.sum(e / n * (1.5 * b2 * n ** 2 / t ** 2.5 - n ** 4 * beta ** 4 / t ** 3.5))
    return (-2.0 * np.sqrt(np.pi) * beta * s
            + np.sqrt(np.pi) * beta / (2 * t ** 1.5)
            - gamma(2.5) * beta ** 3 / (3 * t ** 2.5))


def J_small(beta, t):
    """(A.9): expansion for SMALL beta^2/t (first four terms)."""
    x = beta ** 2 / t
    return (np.sqrt(np.pi) * beta ** 5 / t ** 3.5
            * (1 / 24 + x / 72 + x ** 2 / 160 + x ** 3 / 288))


def C_from_J(alpha, beta, t, J):
    """Eq. (14) = (A.15)."""
    return alpha ** 3 * ((2 * t / beta ** 2 - 2.0 / 3.0) * J
                         - beta ** 3 * np.sqrt(np.pi) / (12 * t ** 2.5))


def C_small(alpha, beta, t):
    """(A.16): expansion for SMALL beta^2/t."""
    x = beta ** 2 / t
    return (alpha ** 3 * beta ** 7 * np.sqrt(np.pi) / t ** 4.5
            * (7 / 2160 + x / 360 + 32 / 14175 * x ** 2 + 2423 / 1159200 * x ** 3))


def J_of(beta, t, switch=1.0):
    """Series for beta^2/t >= switch, otherwise the small expansion."""
    return J_series(beta, t) if beta ** 2 / t >= switch else J_small(beta, t)


def C_of(alpha, beta, t, switch=1.0):
    """C from the series of J for beta^2/t >= switch, otherwise the small expansion."""
    if beta ** 2 / t >= switch:
        return C_from_J(alpha, beta, t, J_series(beta, t))
    return C_small(alpha, beta, t)


# --------------------------------------------------------------------------- #
# V_p (half-space, p. 768) and V_s (sphere, eq. 15)
# --------------------------------------------------------------------------- #
def V_p(a, I0, s2, t, jmax=60):
    """Half-space voltage, series on p. 768."""
    u = a ** 2 * s2 * MU0 / (4 * t)
    j = np.arange(0, jmax)
    from scipy.special import factorial
    c = (2 * (-1.0) ** j * factorial(2 * j + 2)
         / ((2 * j + 5) * factorial(j) * factorial(j + 1) * factorial(j + 2)))
    return (a * MU0 * I0 * np.sqrt(np.pi) / t * np.sqrt(a ** 2 * s2 * MU0)
            / np.sqrt(4 * t) * np.sum(c * u ** (j + 1)))


def V_s(a, b, d, rho, I0, s2, s3, t, nmax=4000):
    """Sphere part after eq. (15)."""
    beta = np.sqrt(s3 * MU0 * b ** 2)
    alpha = np.sqrt(s2 / s3)
    I1, I2, I3 = I1_closed(a, rho, d), I2_closed(a, rho, d), I3_closed(a, rho, d)
    J = J_of(beta, t)
    C = C_of(alpha, beta, t)
    n = np.arange(1, nmax + 1)
    reso = np.sum(np.exp(-n ** 2 * np.pi ** 2 * t / beta ** 2))
    t1 = 3 * MU0 * b ** 3 * I0 * np.pi / (a ** 2 * beta ** 2) * (I1 ** 2 + I2 ** 2) * reso
    t2 = (a ** 2 / b ** 2) * np.sqrt(np.pi) / 4 * rho * MU0 * I0 / t ** 3.5 \
        * (s2 * MU0 * b ** 2) ** 2.5 * I3
    t3 = 2.0 / 3.0 * alpha ** 3 * a * MU0 * I0 * I1 * J
    t4 = 2.0 / 3.0 * MU0 * b ** 3 / a ** 2 * C * (I1 ** 2 + I2 ** 2)
    t5 = 4.0 / 3.0 * MU0 * I0 * b / t ** 3 * I3 ** 2 * (s2 * MU0 * b ** 2) ** 2
    return dict(total=t1 + t2 + t3 + t4 + t5, t1=t1, t2=t2, t3=t3, t4=t4, t5=t5,
                I1=I1, I2=I2, I3=I3, J=J, C=C, beta=beta, alpha=alpha)


def t_min(s2, l):
    """Lower limit of validity: s2*mu0*l^2/t <= 0.2."""
    return 5.0 * s2 * MU0 * l ** 2


# --------------------------------------------------------------------------- #
# Self checks
# --------------------------------------------------------------------------- #
def _tests():
    """Self checks: closed forms against quadrature, rho -> 0, the two expansions of J and C, Lee's Fig. 4."""
    print("=" * 78)
    print("TEST 1  I1, I2, I3: closed form against direct Hankel quadrature")
    print("=" * 78)
    ok = True
    for (a, rho, d) in [(25.0, 50.0, 100.0), (25.0, 150.0, 100.0), (25.0, 10.0, 75.0),
                        (50.0, 200.0, 150.0), (25.0, 300.0, 100.0)]:
        for nm, fc, fq in (('I1', I1_closed, I1_quad), ('I2', I2_closed, I2_quad),
                           ('I3', I3_closed, I3_quad)):
            c, q = fc(a, rho, d), fq(a, rho, d)
            rel = abs(c - q) / max(abs(q), 1e-300)
            flag = 'OK ' if rel < 1e-6 else 'DEVIATION'
            ok &= rel < 1e-6
            print(f"  a={a:5.1f} rho={rho:6.1f} d={d:6.1f}  {nm}: "
                  f"closed {c: .6e}  quadrature {q: .6e}  rel {rel:.2e}  {flag}")
    print(f"\nTEST 2  limit rho -> 0")
    a, d = 25.0, 100.0
    print(f"  I1(rho=0) closed {I1_closed(a,0.0,d):.6e}  "
          f"formula a^3/(a^2+d^2)^1.5 {a**3/(a**2+d**2)**1.5:.6e}")
    for r in (1e-3, 1e-2, 1e-1):
        print(f"  rho={r:7.0e}: I1 {I1_closed(a,r,d):.6e}  I2 {I2_closed(a,r,d):.3e}  "
              f"I3 {I3_closed(a,r,d):.3e}")

    print(f"\nTEST 3  J and C: both expansions in the overlap range")
    beta = np.sqrt(2.0 * MU0 * 50.0 ** 2)       # sphere 2 S/m, b = 50 m
    alpha = np.sqrt(0.01 / 2.0)
    print(f"  beta = {beta:.6e}  beta^2 = {beta**2:.6e} s   alpha = {alpha:.4f}")
    print(f"  {'t [ms]':>8s} {'beta^2/t':>10s} {'J series':>14s} {'J small':>14s} {'rel':>9s}"
          f" {'C(14)':>14s} {'C small':>14s} {'rel':>9s}")
    for tms in (0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
        t = tms * 1e-3
        js, jk = J_series(beta, t), J_small(beta, t)
        cs, ck = C_from_J(alpha, beta, t, js), C_small(alpha, beta, t)
        rj = abs(js - jk) / max(abs(js), 1e-300)
        rc = abs(cs - ck) / max(abs(cs), 1e-300)
        print(f"  {tms:8.2f} {beta**2/t:10.3f} {js: 14.5e} {jk: 14.5e} {rj:9.2e}"
              f" {cs: 14.5e} {ck: 14.5e} {rc:9.2e}")

    print(f"\nTEST 4  Lee's Fig. 4: a=25 m, b=50 m, sphere 2 S/m, ground 0.01 S/m")
    a, b, I0, s2, s3 = 25.0, 50.0, 1.0, 0.01, 2.0
    for d in (75.0, 100.0, 125.0, 150.0):
        print(f"  d = {d:5.1f} m   t_min(rho=0) = {t_min(s2,d)*1e3:5.2f} ms")
        for tms in (1.0, 2.0, 3.0):
            t = tms * 1e-3
            r = V_s(a, b, d, 0.0, I0, s2, s3, t)
            vp = V_p(a, I0, s2, t)
            print(f"     t={tms:4.1f} ms  V_p {vp: .4e}  V_s {r['total']: .4e}  "
                  f"(t1 {r['t1']: .2e} t3 {r['t3']: .2e} t4 {r['t4']: .2e})")
    print(f"\n  Note: Lee's Fig. 4 shows volts per ampere versus rho; t1 is the "
          f"resonance series of the sphere.")
    print("\nSelf check 1 passed" if ok else "\nSelf check 1 FAILED")


if __name__ == '__main__':
    _tests()
