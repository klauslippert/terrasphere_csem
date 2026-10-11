"""Mie coefficients: (1) scaled form = direct form; (2) boundary conditions (tangential E and H continuous at
r = b) satisfied by incident + scattered + interior fields for TE and TM modes, mu_r = 1 and 3; (3) equal to the
reference 1 code for mu_r = 1; (4) null contrast -> 0."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrasphere_csem.mie import mie_scaled, mie_direct
from terrasphere_csem.vswf import MN_point
from terrasphere_csem import MU0


def k_of(sig, w, mu_r=1.0):
    """Wavenumber k = sqrt(i w mu0 mu_r sigma) with Im k >= 0."""
    k = np.sqrt(1j * w * MU0 * mu_r * sig + 0j)
    return k if np.imag(k) >= 0 else -k


def test_scaled_vs_direct():
    """Scaled Mie coefficients against the direct Riccati-Bessel formulas."""
    w, sh, ss, b = 3e3, 0.01, 1.0, 180.0
    for mu_r in (1.0, 3.0):
        k, ks = k_of(sh, w), k_of(ss, w, mu_r)
        a, bb = mie_scaled(8, k, ks, b, mu_r)
        worst = 0.0
        for n in range(1, 9):
            ad, bd = mie_direct(n, k, ks, b, mu_r)
            worst = max(worst, abs(a[n] * k ** (2 * n + 1) - ad) / abs(ad), abs(bb[n] * k ** (2 * n + 1) - bd) / abs(bd))
        print(f"scaled vs direct (mu_r {mu_r}): {worst:.1e}")
        assert worst < 1e-12


def test_boundary_conditions():
    """E_tan and H_tan continuous at r=b: incident c*Z_nm(k) + scattered s*Z^(3)_nm(k) = interior d*Z_nm(k_s)."""
    w, sh, ss, b = 3e3, 0.01, 1.0, 180.0
    worst = 0.0
    for mu_r in (1.0, 3.0):
        k, ks = k_of(sh, w), k_of(ss, w, mu_r)
        for (n, m) in ((1, 0), (2, 1), (3, -2)):
            an, bn = mie_direct(n, k, ks, b, mu_r)
            th, ph = 0.9, 0.4
            rhat = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            tan = lambda v: v - np.dot(v, rhat) * rhat
            M1, N1 = MN_point(n, m, k, b, th, ph, 1); M3, N3 = MN_point(n, m, k, b, th, ph, 3)
            Ms, Ns = MN_point(n, m, ks, b, th, ph, 1)
            # TE (M) incidence: E = M1 - bn M3 outside, d*Ms inside; H = (k/(i w mu)) [N1 - bn N3], (ks/(i w mu mu_r)) d Ns
            Eo = M1 - bn * M3; Ho = k * (N1 - bn * N3)
            d = np.linalg.norm(tan(Eo)) / np.linalg.norm(tan(Ms)) * np.exp(1j * (np.angle(tan(Eo)[0]) - np.angle(tan(Ms)[0])))
            d = tan(Eo)[np.argmax(np.abs(tan(Ms)))] / tan(Ms)[np.argmax(np.abs(tan(Ms)))]
            eE = np.linalg.norm(tan(Eo) - d * tan(Ms)) / np.linalg.norm(tan(Eo))
            eH = np.linalg.norm(tan(Ho) - (ks / mu_r) * d * tan(Ns)) / np.linalg.norm(tan(Ho))
            # TM (N) incidence: E = N1 - an N3, H = (k/(i w mu)) [M1 - an M3]
            Eo = N1 - an * N3; Ho = k * (M1 - an * M3)
            d = tan(Eo)[np.argmax(np.abs(tan(Ns)))] / tan(Ns)[np.argmax(np.abs(tan(Ns)))]
            eE2 = np.linalg.norm(tan(Eo) - d * tan(Ns)) / np.linalg.norm(tan(Eo))
            eH2 = np.linalg.norm(tan(Ho) - (ks / mu_r) * d * tan(Ms)) / np.linalg.norm(tan(Ho))
            worst = max(worst, eE, eH, eE2, eH2)
    print(f"boundary conditions at r=b (TE and TM, mu_r 1 and 3): {worst:.1e}")
    assert worst < 1e-9


def test_reference_and_null():
    """Mie coefficients against the independent reference implementation; null contrast gives zero."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference'))
    from lee1981_coaxial import Bhat
    w, sh, ss, b = 3e3, 0.01, 1.0, 180.0
    k, ks = k_of(sh, w), k_of(ss, w)
    a, bb = mie_scaled(15, k, ks, b)
    ref = Bhat(15, k, ks, b)
    e = np.max(np.abs(bb[1:] - ref) / np.abs(ref))
    a0, b0 = mie_scaled(15, k, k, b)
    print(f"b_hat vs reference 1 Bhat: {e:.1e};  null contrast |a|,|b| / |a|,|b|(contrast) = {np.max(np.abs(a0[1:])/np.abs(a[1:])):.1e}, {np.max(np.abs(b0[1:])/np.abs(bb[1:])):.1e}")
    assert e < 1e-13 and np.max(np.abs(b0[1:]) / np.abs(bb[1:])) < 1e-13


if __name__ == '__main__':
    test_scaled_vs_direct(); test_boundary_conditions(); test_reference_and_null()
