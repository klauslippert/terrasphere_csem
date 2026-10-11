"""Stable ratios of spherical Bessel/Hankel functions and the logarithmic derivative of the Riccati-Bessel
function psi_n(z) = z j_n(z), as used for the Mie coefficients (Bohren & Huffman 1983, ch. 4):
  r_k = h_k^(1)(z)/h_{k-1}^(1)(z)  (upward, stable), s_k = j_k(z)/j_{k-1}(z)  (downward continued fraction),
  D_n(z) = psi_n'(z)/psi_n(z)      (downward recurrence, BH eq. 4.89).
"""
import numpy as np


def ratios_h(z, nmax):
    """Ratios r_k = h_k^(1)(z)/h_{k-1}^(1)(z), k = 1..nmax, by upward recurrence (stable).

    z: complex argument(s). Returns an array (nmax + 1, *z.shape); row 0 is unused (zero)."""
    z = np.atleast_1d(np.asarray(z, complex))
    r = np.zeros((nmax + 1,) + z.shape, complex)
    r[1] = 1.0 / z - 1j                                   # h_1/h_0 in closed form
    for k in range(2, nmax + 1):
        r[k] = (2 * k - 1) / z - 1.0 / r[k - 1]
    return r


def ratios_j(z, nmax, extra=40):
    """Ratios s_k = j_k(z)/j_{k-1}(z), k = 1..nmax, by the downward continued fraction.

    The recurrence starts `extra` + |z| terms above nmax. Returns an array (nmax + 1, *z.shape); row 0 is unused."""
    z = np.atleast_1d(np.asarray(z, complex))
    K = nmax + extra + int(np.max(np.abs(z)))
    sk = np.zeros(z.shape, complex)
    out = np.zeros((nmax + 1,) + z.shape, complex)
    for k in range(K, 0, -1):
        sk = 1.0 / ((2 * k + 1) / z - sk)
        if k <= nmax:
            out[k] = sk
    return out


def logderiv_psi(z, nmax, extra=40):
    """Logarithmic derivative D_n(z) = psi_n'(z)/psi_n(z) of psi_n(z) = z j_n(z), n = 0..nmax, by downward recurrence.

    Returns an array (nmax + 1, *z.shape); the recurrence starts `extra` + |z| degrees above nmax."""
    z = np.atleast_1d(np.asarray(z, complex))
    K = nmax + extra + int(np.max(np.abs(z)))
    D = np.zeros(z.shape, complex)
    out = np.zeros((nmax + 1,) + z.shape, complex)
    for n in range(K, 0, -1):
        if n <= nmax:
            out[n] = D
        D = n / z - 1.0 / (D + n / z)
    out[0] = D
    return out
