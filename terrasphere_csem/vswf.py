"""Closed-form vector spherical wave functions (for tests and for direct field evaluation near the sphere).
M_nm = curl(r psi), N_nm = curl(M_nm)/k, psi = z_n(kr) Y_nm, Y_nm = N_nm P_n^m(cos th) e^{im phi}.
Spherical components (e.g. Bohren & Huffman 1983, eqs. 4.17-4.18, written for e^{im phi}):
  M = z_n(kr) [ th^ (im/sin th) Y  -  ph^ dY/dth ]
  N = r^ n(n+1) z_n(kr)/(kr) Y + (kr z_n(kr))'/(kr) [ th^ dY/dth + ph^ (im/sin th) Y ]
kind=1: z_n = j_n (regular), kind=3: z_n = h_n^(1) (outgoing)."""
import numpy as np
from scipy.special import spherical_jn, spherical_yn, lpmv
from .legendre import norm_nm, pnm_real


def zn(n, x, kind):
    if kind == 1:
        return spherical_jn(n, x)
    return spherical_jn(n, x) + 1j * spherical_yn(n, x)


def dzn(n, x, kind):
    if kind == 1:
        return spherical_jn(n, x, derivative=True)
    return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)


def sph_to_cart(vr, vt, vp, th, ph):
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    vx = vr * st * cp + vt * ct * cp - vp * sp
    vy = vr * st * sp + vt * ct * sp + vp * cp
    vz = vr * ct - vt * st
    return np.array([vx, vy, vz])


def MN_point(n, m, k, r, th, ph, kind):
    """Cartesian components of (M_nm, N_nm) at a point (r, th, ph). Returns (M, N), each shape (3,)."""
    # simpler: use lpmv with signed m directly (scipy supports negative m with CS phase)
    ct = np.cos(th); st = np.sin(th)
    P = lpmv(m, n, ct)
    Pm1 = lpmv(m, n - 1, ct) if n - 1 >= abs(m) else 0.0
    dP = (n * ct * P - (n + m) * Pm1) / st
    Nn = norm_nm(n, m)      # normalisation for signed m (gammaln handles both signs)
    E = np.exp(1j * m * ph)
    x = k * r
    z = zn(n, x, kind); dz = dzn(n, x, kind)
    Mt = z * (1j * m / st) * P * Nn * E
    Mp = -z * dP * Nn * E
    M = sph_to_cart(0.0, Mt, Mp, th, ph)
    Nr = n * (n + 1) * z / x * P * Nn * E
    f = (z + x * dz) / x
    Nt = f * dP * Nn * E
    Np = f * (1j * m / st) * P * Nn * E
    N = sph_to_cart(Nr, Nt, Np, th, ph)
    return M, N


def Y_nm(n, m, th, ph):
    return norm_nm(n, m) * lpmv(m, n, np.cos(th)) * np.exp(1j * m * ph)
