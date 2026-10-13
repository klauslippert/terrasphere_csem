"""Laplace-domain route to the time domain (fixed Talbot): scattered response of the coaxial loop (1:100 design)
against the sine-filter route.  The primary half-space voltage against empymod is in
tests/verification/empymod/test_talbot_empymod.py."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference'))
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver
from terrasphere_csem.timedomain import switch_off_talbot, model_transient, talbot as talbot_fn


def test_talbot_scattered_vs_filter():
    """Talbot inversion of the sphere response against the sine-filter route."""
    A, H, B, sh, ss = 200.0, 250.0, 180.0, 0.01, 1.0
    mod = SphereModel(LayerStack([0.0], [sh]), Sphere(B, ss), (0.0, 0.0, H), nmax=15)
    tx = [Dipole('LOOP', (0.0, 0.0, 0.0), 1.0, radius=A)]; rx = [Receiver('LOOP', (0.0, 0.0, 0.0), radius=A)]
    T = np.array([1e-3, 1e-2, 3.162e-2, 1e-1])
    t0 = time.time()
    vt = switch_off_talbot(lambda w: mod.scattered_field(w, tx, rx), T, M=24)[:, 0]
    dt = time.time() - t0
    vt16 = switch_off_talbot(lambda w: mod.scattered_field(w, tx, rx), T, M=16)[:, 0]
    vt32 = switch_off_talbot(lambda w: mod.scattered_field(w, tx, rx), T, M=32)[:, 0]
    sp = model_transient(mod, tx, rx, T); vf = sp.switch_off(T)[:, 0]
    for i, t in enumerate(T):
        print(f"t = {t*1e3:6.1f} ms: Talbot M16/24/32 {vt16[i]: .6e} {vt[i]: .6e} {vt32[i]: .6e}   sine filter {vf[i]: .6e}   ratio24 {vt[i]/vf[i]:.6f}")
    print(f"(Talbot: {dt:.0f} s for {len(T)} times)")
    assert np.max(np.abs(vt / vf - 1)) < 1e-6           # exact filter evaluation


if __name__ == '__main__':
    test_talbot_scattered_vs_filter()
