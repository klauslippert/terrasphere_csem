# SPDX-License-Identifier: MIT
"""Natural time constants (poles) of the sphere.
Free space (k_h -> 0): the TE Mie denominator reduces to  z j_{n-1}(z) + (mu_r - 1) n j_n(z) = 0,  z = k_s b
(for mu_r = 1 the zeros of j_{n-1}: Nabighian 1970; with permeability: Wait 1951/1953).  With omega = -i/tau:
tau_{n,s} = mu0 mu_r sigma_s b^2 / z_s^2.
Conductive host: the exterior wave h_n(k b) with k = sqrt(i w mu sigma_h) has a branch cut along the negative
imaginary w axis; the response is no longer a sum of exponentials.  pole_fullspace() follows the zero of the TE
denominator  mu_r qr(x) - m D_n(mx) = 0  into the complex w-plane by secant iteration (continuation in sigma_h)."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn
from . import MU0
from .mie import mie_scaled


def poles_freespace(n, sigma_s, b, mu_r=1.0, count=5):
    """First `count` time constants tau (s) of degree n in free space."""
    f = lambda z: z * spherical_jn(n - 1, z) + (mu_r - 1) * n * spherical_jn(n, z)
    zs = []; z = 0.5; step = 0.05
    while len(zs) < count:
        if f(z) * f(z + step) < 0:
            zs.append(brentq(f, z, z + step, xtol=1e-14))
        z += step
    zs = np.array(zs)
    return MU0 * mu_r * sigma_s * b ** 2 / zs ** 2, zs


def te_denominator(w, n, sigma_h, sigma_s, b, mu_r=1.0):
    """Scaled TE Mie denominator  mu_r qr(x) - m D_n(m x)  at complex w (zero <-> pole of b_n)."""
    k = np.sqrt(1j * w * MU0 * sigma_h + 0j); k = k if np.imag(k) >= 0 else -k
    ks = np.sqrt(1j * w * MU0 * mu_r * sigma_s + 0j); ks = ks if np.imag(ks) >= 0 else -ks
    a, bb, QS, QR, D, J = mie_scaled(n, k, ks, b, mu_r, intern=True)
    return mu_r * QR[n] - (ks / k) * D[n]


def secant(f, w0, w1, tol=1e-10, maxit=60):
    """Secant iteration for a zero of the complex function f, started from w0 and w1.

    Stops when the step is below tol relative to the iterate or after maxit steps; returns the last iterate."""
    f0, f1 = f(w0), f(w1)
    for _ in range(maxit):
        w2 = w1 - f1 * (w1 - w0) / (f1 - f0)
        if abs(w2 - w1) < tol * abs(w2):
            return w2
        w0, f0, w1, f1 = w1, f1, w2, f(w2)
    return w1


def pole_fullspace(n, s_index, sigma_s, b, sigma_h_list, mu_r=1.0):
    """Continuation of the s-th TE pole of degree n from free space into hosts sigma_h_list (increasing)."""
    tau, zs = poles_freespace(n, sigma_s, b, mu_r, s_index)
    w = -1j / tau[s_index - 1]
    out = []
    for sh in sigma_h_list:
        f = lambda ww: te_denominator(ww, n, sh, sigma_s, b, mu_r)
        w = secant(f, w, w * (1 + 1e-3))
        out.append(w)
    return np.array(out)
