"""Numerical proof of the four spectral identities (spectral.py) for several (n, m), including
negative m, at points above and below the equatorial plane, with complex k."""
import os, sys
import numpy as np
from scipy.special import jv, lpmv, spherical_jn, spherical_yn, gammaln
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrasphere_csem.spectral import lam_grid, u_of, W_closed, AB_regular_scaled, outgoing_spectrum_scaled
from terrasphere_csem.legendre import scaled_pnm, norm_nm
from terrasphere_csem.vswf import MN_point

K = 0.9 + 0.6j
PTS = [(0.7, 0.4, 1.2), (1.3, 2.9, 0.8), (0.3, -1.0, -1.5), (1.1, 5.5, -0.9)]   # (rho, phi, z)
NM = [(1, 0), (2, 1), (3, -1), (4, 2), (5, -3), (6, 6)]


def h1(n, x):
    """Spherical Hankel function h_n^(1)(x)."""
    return spherical_jn(n, x) + 1j * spherical_yn(n, x)


def sph(rho, phi, z):
    """Spherical coordinates (r, theta, phi) of the cylindrical point (rho, phi, z)."""
    r = np.hypot(rho, z); th = np.arccos(z / r)
    return r, th, phi


def test_I_weyl_scalar():
    """Identity (I): scalar Weyl expansion of an outgoing spherical wave."""
    worst = 0.0
    for (n, m) in NM:
        for (rho, phi, z) in PTS:
            r, th, _ = sph(rho, phi, z)
            lhs = h1(n, K * r) * lpmv(m, n, np.cos(th)) * np.exp(1j * m * phi)
            lam, wl = lam_grid(60.0 / abs(z), 300)
            u = u_of(lam, K); y = (1j if z > 0 else -1j) * u
            S, _ = scaled_pnm(n, m, y, K, lam)
            P = S[n] / K ** n
            rhs = (1j) ** (m - n - 1) / K * np.sum(wl * lam / u * P * jv(m, lam * rho) * np.exp(-u * abs(z))) * np.exp(1j * m * phi)
            worst = max(worst, abs(lhs - rhs) / abs(lhs))
    print(f"(I)   scalar Weyl identity: max rel error {worst:.1e}")
    assert worst < 1e-8


def test_II_planewave_scalar():
    """Identity (II): scalar plane wave in regular spherical waves."""
    worst = 0.0
    lam = 0.9; u = u_of(np.array([lam]), K)[0]
    for sgn in (+1, -1):
        kz = sgn * 1j * u
        for m in (0, 1, -2, 3):
            nmax = 40
            S, _ = scaled_pnm(nmax, m, np.array([kz]), K, np.array([lam]))
            n = np.arange(nmax + 1)
            fac = np.where(n >= abs(m), (2 * n + 1) * (1j) ** (n - m) * np.exp(gammaln(np.maximum(n - m, 0) + 1) - gammaln(np.maximum(n + m, 0) + 1)), 0)
            for (rho, phi, z) in [(0.5, 0.3, 0.3), (0.8, 2.0, -0.6), (0.2, 4.0, 0.9)]:
                r, th, _ = sph(rho, phi, z)
                lhs = jv(m, lam * rho) * np.exp(1j * m * phi) * np.exp(1j * kz * z)
                rhs = sum(fac[nn] * S[nn, 0] / K ** nn * spherical_jn(nn, K * r) * lpmv(m, nn, np.cos(th))
                          for nn in range(abs(m), nmax + 1)) * np.exp(1j * m * phi)
                worst = max(worst, abs(lhs - rhs) / abs(lhs))
    print(f"(II)  scalar plane wave -> regular waves: max rel error {worst:.1e}")
    assert worst < 1e-10


def W_beta_quadrature(m, lam, kz, rho, phi, z, nb=720):
    """W^TE_m, W^TM_m by direct beta integration (reference)."""
    ca, sa = kz / K, lam / K
    beta = np.arange(nb) * 2 * np.pi / nb
    x, y = rho * np.cos(phi), rho * np.sin(phi)
    ph = np.exp(1j * m * beta) * np.exp(1j * lam * (x * np.cos(beta) + y * np.sin(beta))) * np.exp(1j * kz * z)
    phat = np.array([-np.sin(beta), np.cos(beta), np.zeros(nb)])
    that = np.array([ca * np.cos(beta), ca * np.sin(beta), -sa * np.ones(nb)])
    WTE = (phat * ph).sum(1) / (nb * (1j) ** m) * 1.0
    WTM = (that * ph).sum(1) / (nb * (1j) ** m)
    return WTE, WTM


def test_III_vector_planewave():
    """Identity (III): vector plane waves in regular vector spherical wave functions."""
    worst_closed = 0.0; worst_series = 0.0
    lam = 0.9; u = u_of(np.array([lam]), K)[0]
    for sgn in (+1, -1):
        kz = sgn * 1j * u
        for m in (0, 1, -1, 2, -3):
            nmax = 40
            A_TE, B_TE, A_TM, B_TM = AB_regular_scaled(nmax, m, np.array([kz]), K, np.array([lam]))
            for (rho, phi, z) in [(0.5, 0.3, 0.3), (0.8, 2.0, -0.6), (0.2, 4.0, 0.9)]:
                r, th, _ = sph(rho, phi, z)
                ref_TE, ref_TM = W_beta_quadrature(m, lam, kz, rho, phi, z)
                cl_TE, cl_TM = W_closed(m, lam, K, kz, rho, phi, z)
                worst_closed = max(worst_closed, np.linalg.norm(cl_TE[:, 0] - ref_TE) / np.linalg.norm(ref_TE),
                                   np.linalg.norm(cl_TM[:, 0] - ref_TM) / np.linalg.norm(ref_TM))
                sTE = np.zeros(3, complex); sTM = np.zeros(3, complex)
                for nn in range(max(abs(m), 1), nmax + 1):
                    M, N = MN_point(nn, m, K, r, th, phi, 1)
                    sTE += A_TE[nn, 0] / K ** nn * M + B_TE[nn, 0] / K ** nn * N
                    sTM += A_TM[nn, 0] / K ** nn * M + B_TM[nn, 0] / K ** nn * N
                worst_series = max(worst_series, np.linalg.norm(sTE - ref_TE) / np.linalg.norm(ref_TE),
                                   np.linalg.norm(sTM - ref_TM) / np.linalg.norm(ref_TM))
    print(f"(III) W closed form vs beta-quadrature: {worst_closed:.1e};  VSWF series vs beta-quadrature: {worst_series:.1e}")
    assert worst_closed < 1e-10 and worst_series < 1e-9


def test_III_equals_han2008():
    """Identity (III) is the relation of Han, Han & Zhang 2008 (J. Opt. A 10, 015006, eqs. 5-8, 15-19, 35), in our
    normalisation: W^TE_m = (i/lambda) m_mlambda, W^TM_m = -(1/lambda) n_mlambda, M_nm = N_nm m_mn, N_nm = N_nm n_mn.
    Part 1: Han's series m_mlambda = sum_n [A_mn m_mn - i B_mn n_mn] (and the TM analogue with m <-> n) with A_mn, B_mn
    exactly as printed (eqs. 16, 17), against Han's closed cylindrical waves (eqs. 5, 6), for a propagating wave
    (real angle, real k) and for evanescent waves (complex angle, quasi-static complex k).
    Part 2: our coefficients equal the mapped Han coefficients: A^TE = i A_mn/(lambda N_nm), B^TE = B_mn/(lambda N_nm),
    A^TM = i B_mn/(lambda N_nm), B^TM = -A_mn/(lambda N_nm)."""
    from scipy.special import jvp, spherical_jn as sjn

    def han_cyl(m, lam, h, k, r, phi, z):              # Han eqs. (5), (6), cylindrical basis (r, phi, z)
        """Cylindrical vector wave functions of Han (2008), eqs. (5), (6)."""
        E = np.exp(1j * h * z) * np.exp(1j * m * phi); J = jv(m, lam * r); dJ = lam * jvp(m, lam * r)
        mm = np.array([1j * m / r * J, -dJ, 0.0 * J]) * E
        nn = np.array([1j * h / k * dJ, -h * m / (k * r) * J, lam ** 2 / k * J]) * E
        return mm, nn

    def han_sph(n, m, k, R, th, phi):                  # Han eqs. (7), (8), spherical basis (R, th, phi)
        """Spherical vector wave functions of Han (2008), eqs. (7), (8)."""
        ct, st = np.cos(th), np.sin(th); P = lpmv(m, n, ct)
        Pm1 = lpmv(m, n - 1, ct) if n - 1 >= abs(m) else 0.0
        dP = (n * ct * P - (n + m) * Pm1) / st
        x = k * R; j = sjn(n, x); dRj = (j + x * sjn(n, x, derivative=True)) / x   # (1/kR) d/dR [R j_n(kR)]
        E = np.exp(1j * m * phi)
        mm = np.array([0.0 * j, 1j * m / st * P * j, -dP * j]) * E
        nn = np.array([n * (n + 1) / x * P * j, dRj * dP, 1j * m / st * dRj * P]) * E
        return mm, nn

    def cyl2cart(v, phi):
        """Cylindrical components to Cartesian components."""
        return np.array([v[0] * np.cos(phi) - v[1] * np.sin(phi), v[0] * np.sin(phi) + v[1] * np.cos(phi), v[2]])

    def sph2cart(v, th, phi):
        """Spherical components to Cartesian components."""
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(phi), np.cos(phi)
        return np.array([v[0] * st * cp + v[1] * ct * cp - v[2] * sp, v[0] * st * sp + v[1] * ct * sp + v[2] * cp,
                         v[0] * ct - v[1] * st])

    def han_AB(nmax, m, k, lam, h):                    # Han eqs. (16), (17); P and dP/dalpha at the (complex) angle
        """Expansion coefficients A, B of Han (2008), eqs. (16), (17)."""
        S, T = scaled_pnm(nmax, m, np.array([h]), k, np.array([lam]))
        n = np.arange(nmax + 1); A = np.zeros(nmax + 1, complex); B = np.zeros(nmax + 1, complex)
        for nn in range(max(abs(m), 1), nmax + 1):
            P = S[nn, 0] / k ** nn; dP = T[nn, 0] / k ** nn; sa = lam / k
            f = (2 * nn + 1) * np.exp(gammaln(nn - m + 1) - gammaln(nn + m + 1)) / (nn * (nn + 1))
            A[nn] = k * 1j ** (nn - m + 1) * f * sa * dP
            B[nn] = -k * 1j ** (nn - m) * m * f * P
        return A, B

    cases = [('propagating, real k', 1.3 + 0j, 0.7, None), ('evanescent up, complex k', K, 0.9, +1),
             ('evanescent down, complex k', K, 1.6, -1)]
    worst1 = 0.0; worst2 = 0.0; nmax = 40
    for name, k, lam, sgn in cases:
        h = np.sqrt(k ** 2 - lam ** 2) if sgn is None else sgn * 1j * u_of(np.array([lam]), k)[0]
        for m in (0, 1, -1, 2, -3):
            A, B = han_AB(nmax, m, k, lam, h)
            for (rho, phi, z) in [(0.5, 0.3, 0.3), (0.8, 2.0, -0.6), (0.2, 4.0, 0.9)]:
                R, th, _ = sph(rho, phi, z)
                mc, nc = han_cyl(m, lam, h, k, rho, phi, z)
                ref_m, ref_n = cyl2cart(mc, phi), cyl2cart(nc, phi)
                sm = np.zeros(3, complex); sn = np.zeros(3, complex)
                for nn in range(max(abs(m), 1), nmax + 1):
                    ms, ns = han_sph(nn, m, k, R, th, phi)
                    ms, ns = sph2cart(ms, th, phi), sph2cart(ns, th, phi)
                    sm += A[nn] * ms - 1j * B[nn] * ns          # Han eq. (35)
                    sn += A[nn] * ns - 1j * B[nn] * ms          # TM analogue, Han eqs. (18), (19)
                worst1 = max(worst1, np.linalg.norm(sm - ref_m) / np.linalg.norm(ref_m),
                             np.linalg.norm(sn - ref_n) / np.linalg.norm(ref_n))
            A_TE, B_TE, A_TM, B_TM = AB_regular_scaled(nmax, m, np.array([h]), k, np.array([lam]))
            for nn in range(max(abs(m), 1), nmax + 1):
                Nn = norm_nm(nn, m); s = k ** nn
                ours = np.array([A_TE[nn, 0], B_TE[nn, 0], A_TM[nn, 0], B_TM[nn, 0]]) / s
                hans = np.array([1j * A[nn], B[nn], 1j * B[nn], -A[nn]]) / (lam * Nn)
                worst2 = max(worst2, np.max(np.abs(ours - hans)) / np.max(np.abs(hans)))
    print(f"(III) = Han et al. 2008: Han series vs Han closed form {worst1:.1e}; our coefficients vs mapped Han {worst2:.1e}")
    assert worst1 < 1e-9 and worst2 < 1e-12


def test_IV_outgoing_spectra():
    """Identity (IV): outgoing vector spherical wave functions as spectra of cylindrical vector plane waves."""
    worst = 0.0
    for (n, m) in [(1, 0), (1, 1), (2, -1), (3, 2), (4, -3), (5, 0)]:
        for (rho, phi, z) in PTS:
            r, th, _ = sph(rho, phi, z)
            M3, N3 = MN_point(n, m, K, r, th, phi, 3)
            lam, wl = lam_grid(60.0 / abs(z), 300)
            u = u_of(lam, K); kz = (1j if z > 0 else -1j) * u
            aM_TE, aM_TM, aN_TE, aN_TM = outgoing_spectrum_scaled(n, m, kz, K, lam)
            Mi = np.zeros(3, complex); Ni = np.zeros(3, complex)
            for i in range(len(lam)):
                WTE, WTM = W_closed(m, lam[i], K, kz[i], rho, phi, z)
                Mi += wl[i] * (aM_TE[n, i] * WTE[:, 0] + aM_TM[n, i] * WTM[:, 0]) / K ** n
                Ni += wl[i] * (aN_TE[n, i] * WTE[:, 0] + aN_TM[n, i] * WTM[:, 0]) / K ** n
            worst = max(worst, np.linalg.norm(Mi - M3) / np.linalg.norm(M3), np.linalg.norm(Ni - N3) / np.linalg.norm(N3))
    print(f"(IV)  outgoing M^(3), N^(3) as spectra of W: max rel error {worst:.1e}")
    assert worst < 1e-7


if __name__ == '__main__':
    test_I_weyl_scalar(); test_II_planewave_scalar(); test_III_vector_planewave(); test_III_equals_han2008(); test_IV_outgoing_spectra()
