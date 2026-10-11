"""Analytic derivatives of the scattered response w.r.t. sigma_s, b, mu_r against central finite differences."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrasphere_csem.layers import LayerStack
from terrasphere_csem.sources import Dipole
from terrasphere_csem.sphere import SphereModel, Sphere, Receiver


def test_derivatives():
    """Analytic derivatives with respect to sigma_s, b and mu_r against central finite differences."""
    st = LayerStack([0.0, 120.0], [0.02, 0.005]); zc = 200.0
    tx = [Dipole('HED', (150.0, 80.0, 0.0), 1.0, 0.3), Dipole('LOOP', (-100.0, 50.0, 0.0), 1.0, radius=80.0)]
    rx = [Receiver('E', (-220.0, 130.0, 0.0), (1, 0, 0)), Receiver('dBdt', (60.0, -90.0, 0.0), (0, 0, 1)), Receiver('LOOP', (0.0, 0.0, 0.0), radius=100.0)]
    w = 2e3; b0, s0, mu0 = 60.0, 2.0, 1.5
    worst = 0.0
    for param, h in (('sigma_s', 1e-4), ('b', 1e-4), ('mu_r', 1e-4)):
        mod = SphereModel(st, Sphere(b0, s0, mu0), (0.0, 0.0, zc), nmax=10)
        dV = mod.derivative(w, tx, rx, param)
        def val(f):
            """Scattered response with the parameter scaled by (1 + f)."""
            kw = dict(b=b0, sigma_s=s0, mu_r=mu0); kw[param] = kw[param] * (1 + f)
            return SphereModel(st, Sphere(kw['b'], kw['sigma_s'], kw['mu_r']), (0.0, 0.0, zc), nmax=10).scattered_field(w, tx, rx)
        x0 = dict(b=b0, sigma_s=s0, mu_r=mu0)[param]
        fd = (val(h) - val(-h)) / (2 * h * x0)
        err = np.max(np.abs(dV - fd) / np.abs(fd))
        print(f"d/d{param}: analytic {dV}  FD {fd}  max rel diff {err:.1e}")
        worst = max(worst, err)
    assert worst < 1e-6


if __name__ == '__main__':
    test_derivatives()


def test_position_derivatives():
    """Analytic derivatives with respect to the sphere-centre coordinates x_c and y_c against finite differences."""
    st = LayerStack([0.0, 120.0], [0.02, 0.005]); zc = 200.0
    tx = [Dipole('HED', (150.0, 80.0, 0.0), 1.0, 0.3), Dipole('LOOP', (-100.0, 50.0, 0.0), 1.0, radius=80.0)]
    rx = [Receiver('E', (-220.0, 130.0, 0.0), (1, 0, 0)), Receiver('dBdt', (60.0, -90.0, 0.0), (0, 0, 1)), Receiver('LOOP', (10.0, 20.0, 0.0), radius=100.0)]
    w = 2e3; kug = Sphere(60.0, 2.0, 1.5); xc, yc = 15.0, -25.0
    worst = 0.0
    for axis, ax in (('x', 0), ('y', 1)):
        mod = SphereModel(st, kug, (xc, yc, zc), nmax=10)
        dV = mod.derivative_position(w, tx, rx, axis)
        h = 0.05
        def val(d):
            """Scattered response with the sphere centre moved by d along the chosen axis."""
            c = [xc, yc, zc]; c[ax] += d
            return SphereModel(st, kug, tuple(c), nmax=10).scattered_field(w, tx, rx)
        fd = (val(h) - val(-h)) / (2 * h); fd2 = (val(2 * h) - val(-2 * h)) / (4 * h); fd = (4 * fd - fd2) / 3
        err = np.max(np.abs(dV - fd) / np.abs(fd))
        print(f"d/d{axis}_c: analytic {dV}  FD {fd}  max rel diff {err:.1e}")
        worst = max(worst, err)
    assert worst < 1e-6


if __name__ == '__main__':
    test_position_derivatives()
