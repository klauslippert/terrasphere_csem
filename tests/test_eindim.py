"""1-D fields (no sphere) of HED, VMD, HMD in full space, half-space and a 3-layer earth against empymod 2.6.0,
both quasi-static (empymod eperm = 0) and with the conventions of Appendix B of the paper.
Expected agreement 1e-8..1e-10, except where source and receiver lie in the same horizontal plane (full space on
the source plane; grounded dipole surface-to-surface): there the spectral kernel does not decay and the digital
filter limits the accuracy (full space on the source plane: ours 1e-3 against empymod's analytic direct field;
surface-to-surface grounded dipole: 1e-5)."""
import os, sys
import numpy as np
import empymod
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrasphere_csem import MU0
from terrasphere_csem.schichten import Schichten
from terrasphere_csem.quellen import Dipol
from terrasphere_csem.eindim import primaerfeld

AIR = 2e14
HT = {'htarg': {'dlf': 'key_401_2009'}, 'verb': 0}
# empymod includes displacement currents by default (eperm = 1); layersphere is quasi-static -> eperm = 0 everywhere.
# empymod puts a source lying exactly on an interface into the layer ABOVE (for z = 0: the air); layersphere into the
# layer below (the ground).  With eperm = 0 the empymod 'air' is a near-perfect insulator: E in the ground from a
# source in it is fine (1e-12, tests/test_born.py), H is not (1e-3), and fields evaluated in it are ill-conditioned.
# Sources and receivers lying on an interface are therefore moved 1 micrometre into the layer below (this
# displacement itself changes the fields by ~1e-8, which sets the level of agreement here).
def QS(res):
    return {'epermH': [0.0] * len(res), 'epermV': [0.0] * len(res)}


ZSHIFT = 1e-6


def emp_fields(src_typ, src_pos, pts, depth, res, f, psi=0.0):
    """E, H (npts, 3) from empymod for a unit dipole, converted to OUR frame and convention:
    empymod: e^{+iwt}, depth frame (z down, left-handed w.r.t. ours), H with opposite sign, magnetic sources in
    units of magnetic current moment (factor i w mu0 per A m^2), VMD 'dip=90' pointing DOWN (ours: up).
    Hence  E_ours = fac * conj(E_emp) * (1, 1, -1),  H_ours = -fac * conj(H_emp) * (1, 1, -1),
    fac = 1 (HED), i w mu0 (HMD), -i w mu0 (VMD).  (Established against closed-form full-space dipoles.)"""
    w = 2 * np.pi * f
    E = np.zeros((len(pts), 3), complex); H = np.zeros((len(pts), 3), complex)
    msrc = src_typ != 'HED'
    zs = src_pos[2] + (ZSHIFT if (len(depth) > 0 and np.any(np.isclose(src_pos[2], depth))) else 0.0)
    if src_typ == 'VMD':
        src = [src_pos[0], src_pos[1], zs, 0.0, 90.0]
    else:
        src = [src_pos[0], src_pos[1], zs, np.degrees(psi), 0.0]
    for ip, (x, y, z) in enumerate(pts):
        z = z + (ZSHIFT if (len(depth) > 0 and np.any(np.isclose(z, depth))) else 0.0)   # receivers on interfaces: also into the layer below
        for ic, (az, dip) in enumerate(((0.0, 0.0), (90.0, 0.0), (0.0, 90.0))):
            rec = [x, y, z, az, dip]
            xd = (len(depth) == 0)          # full space: empymod's analytical direct field
            E[ip, ic] = empymod.bipole(src=src, rec=rec, depth=depth, res=res, freqtime=f, msrc=msrc, mrec=False, strength=0.0, xdirect=xd, **HT, **QS(res))
            H[ip, ic] = empymod.bipole(src=src, rec=rec, depth=depth, res=res, freqtime=f, msrc=msrc, mrec=True, strength=0.0, xdirect=xd, **HT, **QS(res))
    fac = {'HED': 1.0, 'HMD': 1j * w * MU0, 'VMD': -1j * w * MU0}[src_typ]
    flip = np.array([1.0, 1.0, -1.0])
    return fac * np.conj(E) * flip, -fac * np.conj(H) * flip


def compare(name, stack, dip, pts, depth, res, f, comps_E=(0, 1, 2), comps_H=(0, 1, 2)):
    w = 2 * np.pi * f
    E, H = primaerfeld(stack, [dip], pts, w)
    Ee, He = emp_fields(dip.typ, dip.pos, pts, depth, res, f, dip.psi)
    errE = np.max(np.abs(E[:, comps_E] - Ee[:, comps_E]) / np.max(np.abs(Ee[:, comps_E])))
    errH = np.max(np.abs(H[:, comps_H] - He[:, comps_H]) / np.max(np.abs(He[:, comps_H])))
    print(f"{name:52s}: max |dE|/max|E| = {errE:.1e}   max |dH|/max|H| = {errH:.1e}")
    return errE, errH


def test_fullspace():
    sig = 0.01; f = 300.0
    st = Schichten([0.0], [sig], sigma_luft=sig)
    pts = [(120.0, 40.0, 180.0), (-60.0, 90.0, 60.0), (30.0, -200.0, 140.0)]
    plane = [(250.0, 0.0, 100.0), (-80.0, 170.0, 100.0)]                 # on the source plane
    worst = 0; worst_plane = 0
    for typ, psi in (('HED', 0.0), ('HED', 0.7), ('VMD', 0.0), ('HMD', 0.0), ('HMD', 2.1)):
        d = Dipol(typ, (0.0, 0.0, 100.0), 1.0, psi)
        eE, eH = compare(f"full space {typ} psi={psi}", st, d, pts, [], [1 / sig], f)
        pE, pH = compare(f"full space {typ} psi={psi}, source plane", st, d, plane, [], [1 / sig], f)
        worst = max(worst, eE, eH); worst_plane = max(worst_plane, pE, pH)
    assert worst < 1e-8 and worst_plane < 2e-3


def test_halfspace_surface_source():
    sig = 0.01; f = 300.0
    st = Schichten([0.0], [sig])
    pts_d = [(120.0, 40.0, 50.0), (-300.0, 90.0, 20.0), (30.0, -200.0, 140.0)]
    pts_s = [(120.0, 40.0, 0.0), (-300.0, 90.0, 0.0), (500.0, 100.0, 0.0)]
    worst = 0; worst_s = 0
    for typ, psi in (('HED', 0.0), ('HED', 1.1), ('VMD', 0.0), ('HMD', 0.4)):
        d = Dipol(typ, (0.0, 0.0, 0.0), 1.0, psi)
        eE, eH = compare(f"half space, {typ} at surface, rx at depth", st, d, pts_d, [0], [AIR, 1 / sig], f)
        eE2, eH2 = compare(f"half space, {typ} at surface, rx at surface", st, d, pts_s, [0], [AIR, 1 / sig], f, comps_E=(0, 1))
        worst = max(worst, eE, eH, max(eE2, eH2) if typ != 'HED' else eH2); worst_s = max(worst_s, eE2)
    assert worst < 1e-7 and worst_s < 5e-5


def test_three_layers():
    f = 100.0
    tiefen = [0.0, 80.0, 250.0]; sig = [0.02, 0.002, 0.05]
    st = Schichten(tiefen, sig)
    pts = [(150.0, 60.0, 40.0), (-200.0, 30.0, 150.0), (90.0, -120.0, 400.0), (300.0, 0.0, 0.0)]
    worst = 0
    for (typ, psi, zs) in (('HED', 0.3, 30.0), ('VMD', 0.0, 30.0), ('HMD', 1.0, 30.0), ('HED', 0.0, 0.0), ('VMD', 0.0, 120.0), ('HED', 0.5, 300.0)):
        d = Dipol(typ, (0.0, 0.0, zs), 1.0, psi)
        eE, eH = compare(f"3 layers, {typ} at {zs} m", st, d, pts, tiefen, [AIR] + [1 / s for s in sig], f, comps_E=(0, 1))
        worst = max(worst, eE, eH)
    assert worst < 3e-6


if __name__ == '__main__':
    test_fullspace(); test_halfspace_surface_source(); test_three_layers()
