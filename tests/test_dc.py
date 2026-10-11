"""Direct-current limit (galvanic, TM) of the grounded dipole: conductive sphere in a full space, point-source
series phi_s = (I/(4 pi sig_h)) sum_n B_n r^{-n-1} P_n(cos gamma)/r0^{n+1},  B_n = -b^{2n+1} n (sig_s - sig_h)/(n sig_s + (n+1) sig_h)
(sphere in the field of a point current source, separation of variables), the HED as two point electrodes."""
import os, sys
import numpy as np
from scipy.special import eval_legendre
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver


def phi_s_point(r, r0, b, sig_s, sig_h, I=1.0, nmax=60):
    """Scattered potential at r of a point current source at r0 for a sphere in a full space (Legendre series)."""
    r = np.asarray(r, float); r0 = np.asarray(r0, float)
    R, R0 = np.linalg.norm(r), np.linalg.norm(r0); cg = np.dot(r, r0) / (R * R0)
    tot = 0.0
    for n in range(1, nmax + 1):
        Bn = -b ** (2 * n + 1) * n * (sig_s - sig_h) / (n * sig_s + (n + 1) * sig_h)
        tot += Bn * R ** (-n - 1) * eval_legendre(n, cg) / R0 ** (n + 1)
    return I / (4 * np.pi * sig_h) * tot


def E_s_dc(r, src_pos, psi, b, sig_s, sig_h, dl=0.2, h=1e-2):
    """Secondary E of an HED (moment 1 A m) as two electrodes +-I at +-dl/2 along psi, by central differences."""
    d = np.array([np.cos(psi), np.sin(psi), 0.0]); I = 1.0 / dl
    rp = np.asarray(src_pos) + 0.5 * dl * d; rm = np.asarray(src_pos) - 0.5 * dl * d
    phi = lambda x: phi_s_point(x, rp, b, sig_s, sig_h, I) - phi_s_point(x, rm, b, sig_s, sig_h, I)
    r = np.asarray(r, float)
    return -np.array([(phi(r + h * e) - phi(r - h * e)) / (2 * h) for e in np.eye(3)])


def test_dc_limit():
    """Low-frequency limit of the HED response against the direct-current series (two point electrodes)."""
    sig_h, sig_s, b = 0.01, 0.5, 50.0
    st = LayerStack([0.0], [sig_h], sigma_air=sig_h)
    zc = 300.0; mod = SphereModel(st, Sphere(b, sig_s), (0.0, 0.0, zc), nmax=12)
    src = (120.0, -60.0, 100.0); psi = 0.8
    pts = [(200.0, 150.0, 420.0), (-90.0, 40.0, 180.0), (30.0, -250.0, 300.0)]
    w = 1e-3
    worst = 0.0
    for (x, y, zt) in pts:
        ours = np.array([mod.scattered_field(w, [Dipole('HED', src, 1.0, psi)], [Receiver('E', (x, y, zt), e)])[0] for e in np.eye(3)])
        # DC reference in the sphere frame (z up): r = (x, y, zc - zt), source (xs, ys, zc - zs); E_z sign flips back
        ref = E_s_dc((x, y, zc - zt), (src[0], src[1], zc - src[2]), psi, b, sig_s, sig_h)
        err = np.max(np.abs(ours - ref)) / np.max(np.abs(ref))
        print(f"DC limit, HED, point {(x, y, zt)}: ours {np.real(ours)}  ref {ref}  rel {err:.1e}  (Im/Re {np.max(np.abs(np.imag(ours)))/np.max(np.abs(ref)):.1e})")
        worst = max(worst, err)
    assert worst < 1e-4


if __name__ == '__main__':
    test_dc_limit()
