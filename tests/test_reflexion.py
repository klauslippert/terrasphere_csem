"""External check of the reflection operator R (sphere -> layers -> sphere) for m = 0 and m = +-1 including the
TE/TM mixing: a point dipole placed at the sphere centre radiates a pure degree-1 outgoing VSWF; its field
reflected by the layered earth, re-expanded in regular VSWFs with our R, must equal empymod(layered) -
empymod(full space) at points around the centre.  VMD -> M_{1,0}; HMD -> M_{1,+-1}; HED -> N_{1,+-1}."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terrasphere_csem import MU0
from terrasphere_csem.schichten import Schichten
from terrasphere_csem.kugel import Modell, Kugel
from terrasphere_csem.vswf import MN_point
from terrasphere_csem.quellen import Dipol
from test_eindim import emp_fields

AIR = 2e14


def closed_dipole_E(typ, psi, k, sig, w, pts):
    """E of a unit dipole at the origin in a full space (our conventions; z up), from G = e^{ikR}/(4 pi R)."""
    out = []
    for x, y, z in pts:
        r = np.array([x, y, z]); R = np.linalg.norm(r)
        G = np.exp(1j * k * R) / (4 * np.pi * R); g1 = (1j * k - 1 / R) * G / R
        g2 = G * (-k * k - 3j * k / R + 3 / R ** 2) / R ** 2
        if typ == 'VMD':
            d = np.array([0, 0, 1.0])
        else:
            d = np.array([np.cos(psi), np.sin(psi), 0.0])
        if typ == 'HED':
            E = (g1 * d + g2 * np.dot(d, r) * r + k * k * G * d) / sig
        else:
            E = 1j * w * MU0 * g1 * np.cross(r, d)
        out.append(E)
    return np.array(out)


def test_reflection_all_m():
    tiefen = [0.0, 150.0]; sig = [0.02, 0.004]; zc = 260.0; sig_h = sig[1]
    st = Schichten(tiefen, sig); mod = Modell(st, Kugel(30.0, 1.0), (0.0, 0.0, zc), nmax=20)
    f = 400.0; w = 2 * np.pi * f
    k = np.sqrt(1j * w * MU0 * sig_h + 0j); k = k if k.imag >= 0 else -k
    # evaluation points (sphere frame, z up) at R = 0.4 h_up
    Rr = 0.4 * mod.h_up
    pts = [(Rr * np.sin(t) * np.cos(p), Rr * np.sin(t) * np.sin(p), Rr * np.cos(t)) for t, p in ((0.5, 0.3), (1.3, 2.0), (2.2, 4.0), (2.8, 5.5))]
    lam, wl = mod.grid([], [])
    kk, ks, u, R, T = mod._prep(w, lam)
    worst = 0.0
    for typ, psi, ms, kind in (('VMD', 0.0, (0,), 'M'), ('HMD', 0.4, (-1, 1), 'M'), ('HED', 0.4, (-1, 1), 'N')):
        # fit the dipole field to the degree-1 outgoing VSWFs of the orders ms (exact representation)
        fitpts = [(60.0, 20.0, 50.0), (-30.0, 70.0, -40.0), (10.0, -50.0, 80.0), (-70.0, -20.0, -30.0)]
        Ec = closed_dipole_E(typ, psi, k, sig_h, w, fitpts)
        cols = []
        for m in ms:
            col = []
            for (x, y, z) in fitpts:
                r = np.linalg.norm([x, y, z]); th = np.arccos(z / r); ph = np.arctan2(y, x)
                M3, N3 = MN_point(1, m, k, r, th, ph, 3)
                col.append(M3 if kind == 'M' else N3)
            cols.append(np.concatenate(col))
        coef, res, _, _ = np.linalg.lstsq(np.array(cols).T, Ec.reshape(-1), rcond=None)
        fit_err = np.linalg.norm(np.array(cols).T @ coef - Ec.reshape(-1)) / np.linalg.norm(Ec)
        # reflected field from our R: for each m, outgoing unit (n'=1, kind) -> regular coefficients R_{n,(1,kind)}
        Erefl = np.zeros((len(pts), 3), complex)
        N = mod.N
        for cm, m in zip(coef, ms):
            ch, Rh, spectra = mod._order(m, w, lam, wl, kk, u, R, [])
            col = 0 if kind == 'M' else N
            for ip, (x, y, z) in enumerate(pts):
                r = np.linalg.norm([x, y, z]); th = np.arccos(z / r); ph = np.arctan2(y, x)
                for n in range(max(1, abs(m)), N + 1):
                    cM = Rh[n - 1, col] / kk ** n / kk ** 2; cN = Rh[N + n - 1, col] / kk ** n / kk ** 2
                    M1, N1 = MN_point(n, m, k, r, th, ph, 1)
                    Erefl[ip] += cm * (cM * M1 + cN * N1)
        # empymod: layered minus full space, converted to our frame (depth zeta = zc - z)
        gpts = [(x, y, zc - z) for (x, y, z) in pts]
        E_lay, _ = emp_fields(typ, (0.0, 0.0, zc), gpts, tiefen, [AIR] + [1 / s_ for s_ in sig], f, psi)
        E_ful, _ = emp_fields(typ, (0.0, 0.0, zc), gpts, [], [1 / sig_h], f, psi)
        Eemp = E_lay - E_ful
        err = np.max(np.abs(Erefl - Eemp)) / np.max(np.abs(Eemp))
        print(f"{typ} (orders {ms}, {kind}-type): VSWF fit of the dipole field {fit_err:.1e}; reflected field vs empymod: {err:.1e}  (|E_refl|/|E_dir| ~ {np.max(np.abs(Eemp))/np.max(np.abs(E_ful)):.1e})")
        worst = max(worst, err)
    assert worst < 1e-8


if __name__ == '__main__':
    test_reflection_all_m()
