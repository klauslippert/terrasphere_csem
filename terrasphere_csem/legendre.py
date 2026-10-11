# SPDX-License-Identifier: MIT
"""Associated Legendre functions for real and complex arguments, in a scaled form that stays finite
for the complex 'angles' of the plane-wave spectrum.

For a plane wave with horizontal wavenumber lambda in a medium with wavenumber k the propagation
direction has cos(alpha) = k_z/k = +-i u/k and sin(alpha) = lambda/k (branch fixed by this choice).
|cos(alpha)| can be huge (low frequency), so P_n^m(cos alpha) ~ (u/k)^n over/underflows.  We therefore
compute S_n^m := k^n P_n^m(y/k) with y = k cos(alpha) = +-i u, which is a polynomial in (y, lambda, k^2):

  S_m^m = (-1)^m (2m-1)!! lambda^m,  S_{m+1}^m = (2m+1) y S_m^m,
  (n-m+1) S_{n+1}^m = (2n+1) y S_n^m - (n+m) k^2 S_{n-1}^m,

(the ordinary three-term recurrence multiplied by k^{n+1}).  Derivatives with respect to theta:
  dP_n^m/dtheta = [n cos(a) P_n^m - (n+m) P_{n-1}^m] / sin(a)   ->   T_n^m := k^n dP_n^m/dtheta
  T_n^m = [n y S_n^m - (n+m) k^2 S_{n-1}^m] / lambda,
and the term (m/sin a) P_n^m -> k^n (m/sin a) P_n^m = m (k/lambda) S_n^m.
Negative orders: P_n^{-m} = (-1)^m (n-m)!/(n+m)! P_n^m (Condon-Shortley phase, as scipy.special.lpmv).
"""
import numpy as np
from scipy.special import lpmv, gammaln


def norm_nm(n, m):
    """N_nm = sqrt((2n+1)/(4 pi) (n-m)!/(n+m)!) such that Y_nm = N_nm P_n^m e^{im phi} is orthonormal."""
    n = np.asarray(n, float); m = np.asarray(m, float)
    ok = n >= np.abs(m)
    return np.where(ok, np.sqrt((2 * n + 1) / (4 * np.pi) * np.exp(gammaln(np.where(ok, n - m, 0) + 1) - gammaln(np.where(ok, n + m, 0) + 1))), 0.0)


def neg_m_factor(n, m):
    """P_n^{-m} = neg_m_factor * P_n^m for m >= 0."""
    n = np.asarray(n, float); m = np.asarray(m, float)
    return (-1.0) ** m * np.exp(gammaln(n - m + 1) - gammaln(n + m + 1))


def scaled_pnm(nmax, m, y, k, lam):
    """S_n^m = k^n P_n^m(y/k) and T_n^m = k^n dP_n^m/dtheta for n = 0..nmax (rows n < |m| are zero).
    y, lam: arrays (same shape); k: scalar. Returns (S, T) of shape (nmax+1, ...). Any integer m."""
    y = np.asarray(y, complex); lam = np.asarray(lam, complex)
    ma = abs(int(m))
    S = np.zeros((nmax + 1,) + y.shape, complex)
    k2 = k * k
    if ma <= nmax:
        dfact = np.prod(np.arange(1, 2 * ma, 2)) if ma > 0 else 1.0
        S[ma] = (-1) ** ma * dfact * lam ** ma
        if ma + 1 <= nmax:
            S[ma + 1] = (2 * ma + 1) * y * S[ma]
        for n in range(ma + 1, nmax):
            S[n + 1] = ((2 * n + 1) * y * S[n] - (n + ma) * k2 * S[n - 1]) / (n - ma + 1)
    T = np.zeros_like(S)
    for n in range(ma, nmax + 1):
        T[n] = (n * y * S[n] - (n + ma) * k2 * (S[n - 1] if n >= 1 else 0)) / lam
    if m < 0:
        nn = np.arange(nmax + 1)
        f = np.where(nn >= ma, neg_m_factor(np.maximum(nn, ma), ma), 0.0)
        S = S * f[(slice(None),) + (None,) * y.ndim]
        T = T * f[(slice(None),) + (None,) * y.ndim]
    return S, T


def pnm_real(nmax, m, ct):
    """P_n^m(ct) and dP_n^m/dtheta for real ct = cos(theta), n = 0..nmax, any integer m (rows n<|m| zero).
    Uses scipy.lpmv (Condon-Shortley). Handles sin(theta) -> 0 by the limit formula? No: callers avoid poles."""
    ct = np.asarray(ct, float)
    n = np.arange(nmax + 1)
    P = np.array([lpmv(m, nn, ct) if nn >= abs(m) else np.zeros_like(ct) for nn in n])
    st = np.sqrt(1 - ct * ct)
    dP = np.zeros_like(P)
    for nn in range(abs(m), nmax + 1):
        Pm1 = P[nn - 1] if nn >= 1 else 0.0
        dP[nn] = (nn * ct * P[nn] - (nn + m) * Pm1) / st
    return P, dP
