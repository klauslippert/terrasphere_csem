"""Time domain: coaxial loop over the sphere in a half-space (1:100 design).  the exact filter evaluation of this package
against (i) the independent reference implementation (reference 2; independent assembly, but spline-interpolated: valid to ~1e-5 up to 10 ms) and
(ii) the fixed Talbot inversion (independent time transform, exact) at all times."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference'))
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver
from terrasphere_csem.timedomain import model_transient, time_axis, switch_off_talbot


def test_time_coaxial():
    """Exact filter evaluation of the coaxial-loop transient against the reference implementation and Talbot inversion."""
    import lee1981_coaxial as L2
    A, H, B, sh, ss = 200.0, 250.0, 180.0, 0.01, 1.0
    mod = SphereModel(LayerStack([0.0], [sh]), Sphere(B, ss), (0.0, 0.0, H), nmax=15)
    tx = [Dipole('LOOP', (0.0, 0.0, 0.0), 1.0, radius=A)]; rx = [Receiver('LOOP', (0.0, 0.0, 0.0), radius=A)]
    T = time_axis(3e-4, 1e-1)
    t0 = time.time(); sp = model_transient(mod, tx, rx, T); dt = time.time() - t0
    ours = sp.switch_off()[:, 0]; sel = np.arange(0, len(T), 8)
    ref = L2.Vs_time(T[sel], A, H, B, sh, ss, nmax=15)
    tal = switch_off_talbot(lambda w: mod.scattered_field(w, tx, rx), T[sel], M=24)[:, 0]
    w1 = 0.0; w2 = 0.0
    for j, i in enumerate(sel):
        print(f"t = {T[i]*1e3:8.3f} ms: ours {ours[i]: .9e}  reference 2 {ref[j]/ours[i]-1: .1e}  Talbot {tal[j]/ours[i]-1: .1e}")
        if T[i] <= 1.01e-2:
            w1 = max(w1, abs(ref[j] / ours[i] - 1))
        w2 = max(w2, abs(tal[j] / ours[i] - 1))
    print(f"({len(sp.w)} frequencies for {len(T)} times: {dt:.0f} s)")
    assert w1 < 3e-5 and w2 < 1e-6          # reference 2 is spline-interpolated (~1e-5 up to 10 ms); Talbot is exact


if __name__ == '__main__':
    test_time_coaxial()
