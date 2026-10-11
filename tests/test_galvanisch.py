"""Galvanic (direct-current carrying) responses of a grounded wire.
(1) DC limit: E_x of the wire in a homogeneous half-space at w -> 0 equals the analytic field of two point electrodes
    +-I at the wire ends (U = I/(2 pi sigma r)).
(2) Transient: our switch-off response (sine filter with the V(0) term) against empymod's switch-off in a
    three-layer earth, empymod quasi-static (eperm = 0) with the wire 1 um below the surface.  With empymod's
    defaults (displacement currents, wire in the air layer) its galvanic field is wrong below 0.1 Hz and its
    switch-off unusable (KSA-019/KSA-024)."""
import os, sys
import numpy as np
import empymod
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrasphere_csem.schichten import Schichten
from terrasphere_csem.quellen import draht
from terrasphere_csem.eindim import primaerfeld
from terrasphere_csem.zeit import Transient
AIR = 2e14
wire = draht((-250.0, 0.0), (250.0, 0.0), 1.0, n_gauss=7); pts = [(0.0, 600.0, 0.0), (400.0, 300.0, 0.0)]


def test_dc_two_electrodes():
    sig = 0.02; st = Schichten([0.0], [sig])
    wire15 = draht((-250.0, 0.0), (250.0, 0.0), 1.0, n_gauss=15)
    # source and receivers in the same plane: the spectral kernel does not decay and the 401-point filter limits the
    # accuracy near an electrode to ~1e-3 (empymod, with the same filter, agrees with layersphere to 5e-6 there)
    worst = 0.0
    for (x, y, _) in pts:
        E, _ = primaerfeld(st, wire15, [(x, y, 0.0)], 2 * np.pi * 1e-3)
        U = lambda xx: 1 / (2 * np.pi * sig) * (1 / np.hypot(xx - 250.0, y) - 1 / np.hypot(xx + 250.0, y))
        Ex = -(U(x + 0.5) - U(x - 0.5))
        print(f"half-space wire, DC limit at {(x, y)}: ours {np.real(E[0, 0]): .5e}   two electrodes {Ex: .5e}   rel {abs(np.real(E[0,0])/Ex-1):.1e}")
        worst = max(worst, abs(np.real(E[0, 0]) / Ex - 1))
    assert worst < 2e-3


def test_wire_Ex_switch_off():
    tiefen = [0.0, 100.0, 300.0]; sig = [0.02, 0.005, 0.03]; st = Schichten(tiefen, sig)
    T = np.geomspace(1e-4, 1e-1, 7)
    sp = Transient(lambda wi: primaerfeld(st, wire, pts, wi)[0][:, 0], T); ours = sp.ausschalten()
    src = [-250.0, 250.0, 0.0, 0.0, 1e-6, 1e-6]                      # wire 1 um into the ground (Appendix B)
    worst = 0.0
    for ip, (x, y, z) in enumerate(pts):
        emp = np.real(np.asarray(empymod.bipole(src=src, rec=[x, y, 1e-6, 0.0, 0.0], depth=tiefen, res=[AIR] + [1 / s for s in sig],
                                                freqtime=T, signal=-1, srcpts=7, strength=1.0, verb=0, htarg={'dlf': 'key_401_2009'},
                                                ftarg={'pts_per_dec': -1}, epermH=[0.0] * 4, epermV=[0.0] * 4)))
        d = np.abs(ours[:, ip] - emp) / np.max(np.abs(emp))
        print(f"rx {(x, y)}: switch-off |ours - empymod| / max|empymod| =", np.round(d, 6), f"; V(0) ours {sp.V0()[ip]: .4e}")
        worst = max(worst, np.max(d))
    assert worst < 1e-3


if __name__ == '__main__':
    test_dc_two_electrodes(); test_wire_Ex_switch_off()
