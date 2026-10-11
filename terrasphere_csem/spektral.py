"""Plane-wave (Sommerfeld/Weyl) spectra of spherical waves and the spherical-wave content of plane waves,
for all azimuthal orders m.  These four identities are the bridge between the sphere (spherical modes)
and the layered earth (plane-wave modes).  Every identity is verified numerically in tests/test_spektral.py.

Notation.  k: wavenumber of the homogeneous layer containing the sphere.  For horizontal wavenumber
lambda: u = sqrt(lambda^2 - k^2), Re u > 0; an up-going wave has k_z = +i u (e^{i k_z z} = e^{-u z}, z > 0),
a down-going one k_z = -i u.  The complex 'angle' alpha of the plane wave: cos(alpha) = k_z/k = +-i u/k,
sin(alpha) = lambda/k.  theta^ = (cos a cos b, cos a sin b, -sin a), phi^ = (-sin b, cos b, 0).

Cylindrical vector plane waves of order m (TE_z resp. TM_z polarised):
   W^TE_m(lambda, +-) = (1/(2 pi i^m)) int dbeta e^{i m beta} phi^(beta)        e^{i k.r}
   W^TM_m(lambda, +-) = (1/(2 pi i^m)) int dbeta e^{i m beta} theta^(alpha,beta) e^{i k.r}
with the closed forms (J_m = J_m(lambda rho), ' = derivative w.r.t. the argument, E = e^{i m phi} e^{i k_z z})
   W^TE_m = [ -(m/(lambda rho)) J_m rho^  -  i J_m' phi^ ] E
   W^TM_m = [ -i cos(a) J_m' rho^  +  cos(a) (m/(lambda rho)) J_m phi^  -  sin(a) J_m z^ ] E .
Their z-components: H_z[W^TE] = (lambda/(w mu)) J_m E, E_z[W^TM] = -(lambda/k) J_m E  (used to couple to the
layered-earth scalars H_z (TE_z) and J_z = sigma E_z (TM_z)).

(I)  Scalar Weyl identity:
     h_n(kr) P_n^m(cos th) e^{im phi} = (i^{m-n-1}/k) int_0^inf (lambda/u) P_n^m(cos a_+-) J_m(lambda rho)
                                         e^{im phi} e^{-u|z|} dlambda          (cos a_+- = +-i u/k for z >< 0)
(II) Scalar plane wave in regular spherical waves:
     J_m(lambda rho) e^{im phi} e^{i k_z z} = sum_n (2n+1) i^{n-m} (n-m)!/(n+m)! P_n^m(cos a) j_n(kr) P_n^m(cos th) e^{im phi}
(III) Vector plane waves in regular VSWFs:  W^q_m = sum_n [A^q_nm M_nm + B^q_nm N_nm], q = TE, TM,
     A^TE = -4 pi i^{n-m} N_nm (dP_n^m/dth)(a) / (n(n+1)),      B^TE = -4 pi i^{n-m}   m N_nm P_n^m(cos a)/(sin a n(n+1)),
     A^TM = -4 pi i^{n+1-m} m N_nm P_n^m(cos a)/(sin a n(n+1)), B^TM = -4 pi i^{n+1-m} N_nm (dP_n^m/dth)(a) / (n(n+1)).
(IV) Outgoing VSWFs as spectra of the W's (k_z = +-i u for z >< 0):
     M^(3)_nm = int dlambda [ aM_TE W^TE_m + aM_TM W^TM_m ],  N^(3)_nm = int dlambda [ aN_TE W^TE_m + aN_TM W^TM_m ],
     aM_TE = -i^{m-n} N_nm (lambda/(i k u)) (dP_n^m/dth)(a),    aM_TM = +i^{m-n} N_nm (lambda/(i k u)) (i m/sin a) P_n^m(cos a),
     aN_TE = i^{1-n+m} N_nm (lambda/(i k u)) (i m/sin a) P_n^m,  aN_TM = i^{1-n+m} N_nm (lambda/(i k u)) (dP_n^m/dth)(a),
     (the measure lambda/(i k u) is the same for both half-spaces; only cos a = k_z/k changes sign).
Scaled forms (no over/underflow): all functions below return the coefficients multiplied by k^n, expressed
through S_n^m = k^n P_n^m and T_n^m = k^n dP_n^m/dth of legendre.scaled_pnm.
"""
import numpy as np
from scipy.special import jv, jvp
from .legendre import scaled_pnm, norm_nm


def lam_grid(L, npan, nq=24):
    """Composite Gauss-Legendre grid on [0, L] with npan panels: (lam, weights)."""
    x, w = np.polynomial.legendre.leggauss(nq)
    e = np.linspace(0.0, L, npan + 1)
    lam = np.concatenate([0.5 * (e[i + 1] - e[i]) * x + 0.5 * (e[i + 1] + e[i]) for i in range(npan)])
    wl = np.concatenate([0.5 * (e[i + 1] - e[i]) * w for i in range(npan)])
    return lam, wl


def u_of(lam, k):
    u = np.sqrt(np.asarray(lam, complex) ** 2 - k * k)
    return np.where(np.real(u) < 0, -u, u)


def W_closed(m, lam, k, kz, rho, phi, z):
    """Closed forms of W^TE_m and W^TM_m at cylindrical points (rho, phi, z) for scalar lam, kz.
    Returns (W_TE, W_TM), each of shape (3, npts) in CARTESIAN components."""
    rho = np.atleast_1d(rho).astype(float); phi = np.atleast_1d(phi).astype(float); z = np.atleast_1d(z).astype(float)
    ca, sa = kz / k, lam / k
    J = jv(m, lam * rho); dJ = jvp(m, lam * rho)
    E = np.exp(1j * m * phi) * np.exp(1j * kz * z)
    mJ = (m / (lam * rho)) * J
    te_r, te_p, te_z = -mJ * E, -1j * dJ * E, np.zeros_like(E)
    tm_r, tm_p, tm_z = -1j * ca * dJ * E, ca * mJ * E, -sa * J * E
    cp, sp = np.cos(phi), np.sin(phi)
    def cart(vr, vp, vz):
        return np.array([vr * cp - vp * sp, vr * sp + vp * cp, vz])
    return cart(te_r, te_p, te_z), cart(tm_r, tm_p, tm_z)


def AB_regular_scaled(nmax, m, y, k, lam):
    """k^n * (A^TE, B^TE, A^TM, B^TM)_nm for n = 0..nmax at plane-wave 'angle' y = k cos(alpha) (= +-i u),
    sin(alpha) = lam/k.  y, lam arrays of equal shape. Rows n < max(|m|,1) are zero."""
    S, T = scaled_pnm(nmax, m, y, k, lam)
    n = np.arange(nmax + 1)
    Nn = norm_nm(n, m)
    pref = np.zeros(nmax + 1); pref[1:] = 4 * np.pi * Nn[1:] / (n[1:] * (n[1:] + 1))
    pref = pref[(slice(None),) + (None,) * np.ndim(y)]
    ph = ((1j) ** (n - m))[(slice(None),) + (None,) * np.ndim(y)]
    kol = (k / lam)[None]                                 # 1/sin(alpha) with the k^n scaling absorbed in S
    A_TE = -pref * ph * T
    B_TE = -pref * ph * m * kol * S
    A_TM = -pref * ph * 1j * m * kol * S
    B_TM = -pref * ph * 1j * T
    return A_TE, B_TE, A_TM, B_TM


def outgoing_spectrum_scaled(nmax, m, kz, k, lam):
    """k^n * (aM_TE, aM_TM, aN_TE, aN_TM)_nm for n = 0..nmax; kz = +-i u is the z-wavenumber (= k cos alpha)."""
    y = np.asarray(kz, complex)
    u = np.where(np.imag(y) >= 0, y / 1j, -y / 1j)        # |k_z| = u with Re u > 0
    S, T = scaled_pnm(nmax, m, y, k, lam)
    n = np.arange(nmax + 1)
    Nn = norm_nm(n, m)[(slice(None),) + (None,) * np.ndim(y)]
    ph = ((1j) ** (m - n))[(slice(None),) + (None,) * np.ndim(y)]
    g = (lam / (1j * k * u))[None]
    kol = (k / lam)[None]
    aM_TE = -ph * Nn * g * T
    aM_TM = ph * Nn * g * 1j * m * kol * S
    aN_TE = 1j * ph * Nn * g * 1j * m * kol * S
    aN_TM = 1j * ph * Nn * g * T
    return aM_TE, aM_TM, aN_TE, aN_TM
