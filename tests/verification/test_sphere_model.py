"""Sphere model: (1) coaxial loop, half-space -> equals reference 2 (m = 0, TE); (2) reciprocity Tx <-> Rx
for electric dipoles at arbitrary positions (all m, TE+TM); (3) null contrast -> scattered field vanishes."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference'))
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver


def test_coaxial_vs_lee1981():
    """Coaxial loop over a sphere in a half-space against the independent reference implementation."""
    import lee1981_coaxial as L2
    A, H, B, sh, ss = 200.0, 250.0, 180.0, 0.01, 1.0
    mod = SphereModel(LayerStack([0.0], [sh]), Sphere(B, ss), (0.0, 0.0, H), nmax=15)
    tx = [Dipole('LOOP', (0.0, 0.0, 0.0), 1.0, radius=A)]
    rx = [Receiver('LOOP', (0.0, 0.0, 0.0), radius=A)]
    worst = 0.0
    for w in (100.0, 3e3, 1e5):
        ours = mod.scattered_field(w, tx, rx)[0]
        ref = L2.Vs_freq(w, A, H, B, sh, ss, nmax=15)[0]
        print(f"coaxial loop w={w:7.0f}: ours {ours: .6e}  reference 2 {ref: .6e}  rel {abs(ours-ref)/abs(ref):.1e}")
        worst = max(worst, abs(ours - ref) / abs(ref))
    assert worst < 1e-5


def test_fullspace_vs_singh():
    """Coaxial loop and sphere in a full space against the independent reference implementation."""
    import singh_multipole as S1
    A, H, B, sh, ss = 200.0, 250.0, 180.0, 0.01, 1.0
    mod = SphereModel(LayerStack([0.0], [sh], sigma_air=sh), Sphere(B, ss), (0.0, 0.0, H), nmax=15)
    tx = [Dipole('LOOP', (0.0, 0.0, 0.0), 1.0, radius=A)]; rx = [Receiver('LOOP', (0.0, 0.0, 0.0), radius=A)]
    worst = 0.0
    for w in (100.0, 3e3, 1e5):
        ours = mod.scattered_field(w, tx, rx)[0]; ref = S1.Vs_freq(w, A, H, B, sh, ss, nmax=15)[0]
        print(f"full space loop w={w:7.0f}: ours {ours: .6e}  reference 1 {ref: .6e}  rel {abs(ours-ref)/abs(ref):.1e}")
        worst = max(worst, abs(ours - ref) / abs(ref))
    assert worst < 1e-5


def test_reciprocity():
    """Reciprocity of transmitter and receiver for electric dipoles at arbitrary positions."""
    st = LayerStack([0.0, 120.0], [0.02, 0.005]); mod = SphereModel(st, Sphere(60.0, 2.0), (0.0, 0.0, 200.0), nmax=10)
    pA = (150.0, 80.0, 0.0); pB = (-220.0, 130.0, 0.0); dA = (np.cos(0.3), np.sin(0.3), 0.0); dB = (np.cos(2.0), np.sin(2.0), 0.0)
    w = 2e3
    vAB = mod.scattered_field(w, [Dipole('HED', pA, 1.0, 0.3)], [Receiver('E', pB, dB)])[0]
    vBA = mod.scattered_field(w, [Dipole('HED', pB, 1.0, 2.0)], [Receiver('E', pA, dA)])[0]
    print(f"reciprocity HED/HED: {vAB: .6e} vs {vBA: .6e}  rel {abs(vAB-vBA)/abs(vAB):.1e}")
    # magnetic: H.n at B due to HED at A  <->  E.d at A due to HMD at B (moment 1 A m^2): E_A = -i w mu0 * H_B? check sign numerically
    hB = mod.scattered_field(w, [Dipole('HED', pA, 1.0, 0.3)], [Receiver('H', pB, (0, 0, 1.0))])[0]
    eA = mod.scattered_field(w, [Dipole('VMD', pB, 1.0)], [Receiver('E', pA, dA)])[0]
    from terrasphere_csem import MU0
    print(f"reciprocity HED/VMD: -i w mu0 H_B = {-1j*w*MU0*hB: .6e}   E_A = {eA: .6e}   ratio {eA/(-1j*w*MU0*hB):.6f}")
    assert abs(vAB - vBA) / abs(vAB) < 1e-6


def test_null_contrast():
    """The scattered field vanishes when the sphere has the conductivity of its layer."""
    st = LayerStack([0.0, 120.0], [0.02, 0.005]); mod = SphereModel(st, Sphere(60.0, 0.005), (0.0, 0.0, 200.0), nmax=10)
    mod2 = SphereModel(st, Sphere(60.0, 0.5), (0.0, 0.0, 200.0), nmax=10)
    w = 2e3
    tx = [Dipole('HED', (150.0, 80.0, 0.0), 1.0, 0.3)]; rx = [Receiver('E', (-220.0, 130.0, 0.0), (1, 0, 0))]
    s0 = mod.scattered_field(w, tx, rx)[0]; s1 = mod2.scattered_field(w, tx, rx)[0]
    print(f"null contrast: |V_s(null)|/|V_s(contrast)| = {abs(s0)/abs(s1):.1e}")
    assert abs(s0) / abs(s1) < 1e-12


if __name__ == '__main__':
    test_coaxial_vs_lee1981(); test_fullspace_vs_singh(); test_reciprocity(); test_null_contrast()
