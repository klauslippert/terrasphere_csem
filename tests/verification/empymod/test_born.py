"""First-order null test against empymod.  At null contrast (sigma_s = sigma of its layer) the sphere response
vanishes identically; its derivative with respect to sigma_s is the Born (Frechet) integral
   dV/dsigma_s = int_sphere E_tx(r) . E_adj(r) dV,
with E_tx the 1-D field of the transmitter and E_adj the 1-D field of the 'adjoint' source at the receiver
(Lorentz reciprocity, e^{-iwt}):  E receiver along d  -> unit HED along d at the receiver;
H_z receiver -> E_adj = E of a unit VMD at the receiver divided by i w mu0  (E_b(A).p = i w mu0 m.H_a(B)),
dB_z/dt = -i w mu0 H_z -> E_adj = -E(VMD).
Both 1-D fields come from empymod (independent code; displacement currents switched off, eperm = 0, because
empymod includes them by default and this package is quasi-static - with them the deviation is ~ w eps0/sigma); the left side from terrasphere_csem's analytic derivative, which
exercises the source map (Graf decomposition, all orders m, TE and TM), the Mie derivative and the receiver map."""
import os, sys
import numpy as np
import empymod
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from terrasphere_csem import MU0
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver
AIR = 2e14
DEPTHS = [0.0, 100.0, 300.0]; SIG = [0.02, 0.005, 0.03]
ZC = 200.0; XC, YC = 60.0, -40.0; B = 40.0


def emp_E(kind, pos, psi, pts, f):
    """E (npts, 3) in OUR convention from empymod (conversion as in tests/verification/empymod/test_onedim.py)."""
    pts = np.asarray(pts, float); w = 2 * np.pi * f
    src = [pos[0], pos[1], pos[2], 0.0, 90.0] if kind == 'VMD' else [pos[0], pos[1], pos[2], np.degrees(psi), 0.0]
    E = np.zeros((len(pts), 3), complex)
    for ic, (az, dip) in enumerate(((0.0, 0.0), (90.0, 0.0), (0.0, 90.0))):
        E[:, ic] = np.asarray(empymod.bipole(src=src, rec=[pts[:, 0], pts[:, 1], pts[:, 2], az, dip], depth=DEPTHS,
                                             res=[AIR] + [1 / s for s in SIG], freqtime=f, msrc=(kind != 'HED'),
                                             strength=0.0, verb=0, htarg={'dlf': 'key_401_2009'},
                                             epermH=[0.0] * 4, epermV=[0.0] * 4)).reshape(-1)   # quasi-static like this package
    fac = {'HED': 1.0, 'HMD': 1j * w * MU0, 'VMD': -1j * w * MU0}[kind]
    return fac * np.conj(E) * np.array([1.0, 1.0, -1.0])


def sphere_quadrature(nr=8, nt=12, nphi=16):
    """Points (global x, y, zeta) and weights of a product rule on the sphere (r^2 dr dcos dphi)."""
    xr, wr = np.polynomial.legendre.leggauss(nr); r = 0.5 * B * (xr + 1); wr = 0.5 * B * wr * r ** 2
    xc, wc = np.polynomial.legendre.leggauss(nt); ph = 2 * np.pi * np.arange(nphi) / nphi; wp = 2 * np.pi / nphi
    R, C, P = np.meshgrid(r, xc, ph, indexing='ij'); W = wr[:, None, None] * wc[None, :, None] * wp * np.ones((1, 1, nphi))
    S = np.sqrt(1 - C ** 2)
    x = XC + R * S * np.cos(P); y = YC + R * S * np.sin(P); z_up = R * C
    return np.column_stack([x.ravel(), y.ravel(), (ZC - z_up).ravel()]), W.ravel()


def born(tx, rx_kind, rx_pos, rx_vec, f, quad):
    """Born integral of E_tx . E_adj over the sphere (quadrature `quad`) from empymod fields: d(response)/d(sigma_s) at null contrast."""
    pts, wq = quad; w = 2 * np.pi * f
    Etx = emp_E(tx.kind, tx.pos, tx.psi, pts, f)
    if rx_kind == 'E':
        psi = np.arctan2(rx_vec[1], rx_vec[0]); Eadj = emp_E('HED', rx_pos, psi, pts, f)
    else:                                                    # dB_z/dt
        Eadj = -emp_E('VMD', rx_pos, 0.0, pts, f)
    return np.sum(wq * np.sum(Etx * Eadj, axis=1))


def test_born_null():
    """Analytic derivative with respect to sigma_s at null contrast against the Born integral with empymod fields."""
    st = LayerStack(DEPTHS, SIG); mod = SphereModel(st, Sphere(B, SIG[1]), (XC, YC, ZC), nmax=8)
    txs = [Dipole('HED', (300.0, 100.0, 0.0), 1.0, 0.5), Dipole('VMD', (-150.0, 80.0, 0.0), 1.0)]
    rxs = [('E', (-250.0, 150.0, 0.0), (1.0, 0.0, 0.0)), ('dBdt', (100.0, 250.0, 0.0), (0.0, 0.0, 1.0))]
    q1, q2 = sphere_quadrature(), sphere_quadrature(10, 16, 24)
    worst = 0.0
    for f in (10.0, 100.0, 1000.0, 1e4):
        w = 2 * np.pi * f
        for tx in txs:
            ours = mod.derivative(w, [tx], [Receiver(t, p, v) for (t, p, v) in rxs], 'sigma_s')
            for ir, (t, p, v) in enumerate(rxs):
                b1 = born(tx, t, p, v, f, q1); b2 = born(tx, t, p, v, f, q2)
                err = abs(ours[ir] - b2) / abs(b2)
                print(f"f={f:6.0f} Hz  {tx.kind} -> {t:4s}: analytic dV/dsigma_s {ours[ir]: .6e}   Born(empymod) {b2: .6e}   rel {err:.1e}   (quadrature {abs(b1-b2)/abs(b2):.0e})")
                worst = max(worst, err)
    assert worst < 1e-10


if __name__ == '__main__':
    test_born_null()
