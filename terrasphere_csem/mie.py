"""Mie (T-matrix) coefficients of a homogeneous sphere (radius b, wavenumber k_s, permeability mu_r mu0) in a
homogeneous host (wavenumber k, mu0).  For an incident regular field  E = sum a_nm M_nm + b_nm N_nm  the
scattered field is  E_sc = sum (-b_n a_nm) M^(3)_nm + (-a_n b_nm) N^(3)_nm,  with (Bohren & Huffman 1983,
eqs. 4.56-4.57 written with Riccati-Bessel functions psi_n(z) = z j_n(z), xi_n(z) = z h_n^(1)(z);
m = k_s/k, x = k b):
  a_n = [m psi_n(mx) psi_n'(x) - mu_r psi_n(x) psi_n'(mx)] / [m psi_n(mx) xi_n'(x) - mu_r xi_n(x) psi_n'(mx)]  (TM, N)
  b_n = [mu_r psi_n(mx) psi_n'(x) - m psi_n(x) psi_n'(mx)] / [mu_r psi_n(mx) xi_n'(x) - m xi_n(x) psi_n'(mx)]  (TE, M)
Evaluated without over/underflow through D_n(mx) = psi_n'/psi_n, the ratios j_n/j_{n-1}, h_n/h_{n-1}, and
j_n(x)/(h_n(x) x^{2n+1}); returned SCALED as a_n/k^{2n+1}, b_n/k^{2n+1} (finite as w -> 0).
Both coefficients vanish identically for k_s = k and mu_r = 1 (null-contrast test)."""
import numpy as np
from .bessel import ratios_h, ratios_j, logderiv_psi


def mie_scaled(nmax, k, ks, b, mu_r=1.0, intern=False):
    """(a_hat, b_hat) with a_hat[n] = a_n/k^(2n+1), b_hat[n] = b_n/k^(2n+1), n = 0..nmax (index 0 unused = 0).
    intern=True also returns the ingredients (qs, qr, D, J) used by the analytic derivatives."""
    x, mx, m = k * b, ks * b, ks / k
    rx = ratios_h(x, nmax)[:, 0]; sx = ratios_j(x, nmax)[:, 0]; Dm = logderiv_psi(mx, nmax)[:, 0]
    jh = (1 - np.exp(-2j * x)) / 2 / x                   # (j0/h0)/x  (= j_0/(h_0 x^1))
    a = np.zeros(nmax + 1, complex); bb = np.zeros(nmax + 1, complex)
    QS = np.zeros(nmax + 1, complex); QR = np.zeros(nmax + 1, complex); JJ = np.zeros(nmax + 1, complex)
    for n in range(1, nmax + 1):
        jh = jh * sx[n] / (rx[n] * x * x)                 # j_n/(h_n x^(2n+1))
        qs = 1.0 / sx[n] - n / x                          # psi_n'(x)/psi_n(x)
        qr = 1.0 / rx[n] - n / x                          # xi_n'(x)/xi_n(x)
        Ga = (m * qs - mu_r * Dm[n]) / (m * qr - mu_r * Dm[n])
        Gb = (mu_r * qs - m * Dm[n]) / (mu_r * qr - m * Dm[n])
        a[n] = Ga * jh * b ** (2 * n + 1)
        bb[n] = Gb * jh * b ** (2 * n + 1)
        QS[n], QR[n], JJ[n] = qs, qr, jh
    if intern:
        return a, bb, QS, QR, Dm, JJ
    return a, bb


def d_mie_scaled(nmax, k, ks, b, mu_r, param):
    """Analytic derivatives (da_hat, db_hat)/d param, param in {'sigma_s', 'b', 'mu_r'}, from the closed forms
    qs' = n(n+1)/x^2 - 1 - qs^2 (Riccati-Bessel equation), likewise qr' and D'(z), (ln J)' = qs - qr - (2n+1)/x,
    with x = k b, z = k_s b, m = k_s/k:  d/db: dx = x/b, dz = z/b;  d/dsigma_s: dz = z/(2 sigma_s), dm = m/(2 sigma_s);
    d/dmu_r: dz = z/(2 mu_r), dm = m/(2 mu_r) plus the explicit mu_r.  (sigma_s enters only through k_s.)"""
    a, bb, QS, QR, Dm, JJ = mie_scaled(nmax, k, ks, b, mu_r, intern=True)
    x, z, m = k * b, ks * b, ks / k
    da = np.zeros(nmax + 1, complex); db = np.zeros(nmax + 1, complex)
    for n in range(1, nmax + 1):
        qs, qr, D, J = QS[n], QR[n], Dm[n], JJ[n]
        dqs_dx = n * (n + 1) / x ** 2 - 1 - qs ** 2
        dqr_dx = n * (n + 1) / x ** 2 - 1 - qr ** 2
        dD_dz = n * (n + 1) / z ** 2 - 1 - D ** 2
        dlnJ_dx = qs - qr - (2 * n + 1) / x
        if param == 'b':
            dx, dz, dm, dmu, dlnb = x / b, z / b, 0.0, 0.0, (2 * n + 1) / b
        elif param == 'sigma_s':
            sig_s = None
            dx, dz, dm, dmu, dlnb = 0.0, z / 2, m / 2, 0.0, 0.0       # per unit of ln(sigma_s); rescaled below
        elif param == 'mu_r':
            dx, dz, dm, dmu, dlnb = 0.0, z / (2 * mu_r), m / (2 * mu_r), 1.0, 0.0
        else:
            raise ValueError(param)
        dqs, dqr, dD = dqs_dx * dx, dqr_dx * dx, dD_dz * dz
        Na, Da = m * qs - mu_r * D, m * qr - mu_r * D
        dNa = dm * qs + m * dqs - dmu * D - mu_r * dD
        dDa = dm * qr + m * dqr - dmu * D - mu_r * dD
        Ga = Na / Da; dGa = (dNa * Da - Na * dDa) / Da ** 2
        Nb, Db = mu_r * qs - m * D, mu_r * qr - m * D
        dNb = dmu * qs + mu_r * dqs - dm * D - m * dD
        dDb = dmu * qr + mu_r * dqr - dm * D - m * dD
        Gb = Nb / Db; dGb = (dNb * Db - Nb * dDb) / Db ** 2
        bp = b ** (2 * n + 1)
        da[n] = (dGa * J + Ga * J * (dlnJ_dx * dx + dlnb)) * bp
        db[n] = (dGb * J + Gb * J * (dlnJ_dx * dx + dlnb)) * bp
    return da, db


def mie_direct(n, k, ks, b, mu_r=1.0):
    """Unscaled a_n, b_n straight from the Riccati-Bessel formulas (for tests, moderate arguments only)."""
    from scipy.special import spherical_jn, spherical_yn
    x, mx, m = k * b, ks * b, ks / k
    psi = lambda z: z * spherical_jn(n, z)
    dpsi = lambda z: spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    xi = lambda z: z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    dxi = lambda z: (spherical_jn(n, z) + 1j * spherical_yn(n, z)) + z * (spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True))
    a = (m * psi(mx) * dpsi(x) - mu_r * psi(x) * dpsi(mx)) / (m * psi(mx) * dxi(x) - mu_r * xi(x) * dpsi(mx))
    bn = (mu_r * psi(mx) * dpsi(x) - m * psi(x) * dpsi(mx)) / (mu_r * psi(mx) * dxi(x) - m * xi(x) * dpsi(mx))
    return a, bn
