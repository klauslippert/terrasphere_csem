"""Sphere below the terrain (kugel_gelaende.py):
(1) order 0: sphere response at surface points from the radial tables against kugel.Modell.streufeld;
(2) the grid route of the higher orders (grid plane waves -> sphere -> grid) for a smooth incident field (vertical
    magnetic dipole 200 m above the ground) against kugel.Modell.streufeld (periodic box: images at 5 km);
(3) null contrast: sphere conductivity = host -> the coupled series equals the layered series;
(4) sphere on the hill axis: rotating the source by 90 degrees rotates the coupled fields (all orders m)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from terrasphere_csem import MU0, huegel
from terrasphere_csem.schichten import Schichten
from terrasphere_csem.quellen import geerdeter_dipol, Dipol, dipol_moden
from terrasphere_csem.kugel import Modell, Kugel, Empfaenger
from terrasphere_csem.kugel_gelaende import KugelGelaende

SIG = 0.01
ST = Schichten([0.0, 300.0], [SIG, 0.003])


def _streu(km, w, quellen, pts):
    out = {}
    for c, typ, v in [('Ex', 'E', (1, 0, 0)), ('Ey', 'E', (0, 1, 0)), ('Hx', 'H', (1, 0, 0)), ('Hy', 'H', (0, 1, 0)),
                      ('Hz', 'H', (0, 0, 1))]:
        out[c] = km.streufeld(w, quellen, [Empfaenger(typ, (x, y, 0.0), vec=v) for x, y in pts])
    return out


def test_ordnung0_tabelle():
    km = Modell(ST, Kugel(60.0, 1.0), (40.0, -30.0, 140.0), nmax=10)
    q = geerdeter_dipol((0.0, -1500.0), psi=0.2, laenge=1.0)
    g = huegel.Gitter(q, n=256, dx=10.0, s=150.0)
    kg = KugelGelaende(km, g)
    pts = np.array([[0.0, 0.0], [200.0, -100.0], [-400.0, 300.0], [600.0, 500.0]])
    worst = 0.0
    for f in [1.0, 100.0, 3e3]:
        w = 2 * np.pi * f
        a = kg.oberflaeche(w, q, pts[:, 0], pts[:, 1]); b = _streu(km, w, q, pts)
        for c in a:
            e = np.max(np.abs(a[c] - b[c])) / np.max(np.abs(b[c])); worst = max(worst, e)
            print(f"f = {f:6.0f} Hz {c}: radial table vs kugel.Modell {e:.1e}")
    assert worst < 1e-5, worst


def test_gitterweg_vmd():
    km = Modell(ST, Kugel(60.0, 1.0), (0.0, 0.0, 140.0), nmax=10)
    src = Dipol('VMD', (150.0, 80.0, -200.0), moment=1.0)
    g = huegel.Gitter(geerdeter_dipol((0.0, -1500.0)), n=512, dx=10.0, s=150.0)
    kg = KugelGelaende(km, g)
    pts = np.array([[0.0, 0.0], [200.0, -100.0], [-300.0, 250.0], [450.0, 400.0]])
    worst = 0.0
    for f in [3.0, 300.0]:
        w = 2 * np.pi * f
        kg.frequenz(w)
        lam = kg.lam
        x0 = (g.n // 2) * g.dx
        A = dipol_moden(src, src.pos[:2], lam, 0, 0.0, lam)                 # TE, about its own axis (m = 0)
        Rup, Rdn = ST.reflections('TE', lam, w)
        Pu, Pd = ST.propagate('TE', lam, w, src.pos[2], A[0], A[1], 0.0, Rup, Rdn)
        ph = np.exp(-1j * (g.KX * (x0 + src.pos[0]) + g.KY * (x0 + src.pos[1])))[kg.msk]
        Hz_d = Pd / (2 * np.pi * lam) * g.n ** 2 * kg.dxi ** 2 * ph
        gam = np.sqrt(g.KAP ** 2 - 1j * w * MU0 * SIG)
        hd = np.zeros((3, g.n, g.n), complex)
        a_par = 1j * Hz_d / lam
        hd[0][kg.msk] = gam[kg.msk] * g.EX[kg.msk] * a_par; hd[1][kg.msk] = gam[kg.msk] * g.EY[kg.msk] * a_par
        hd[2][kg.msk] = Hz_d
        dhd, dhu, dB = kg.antwort(w, hd, np.zeros_like(hd), gam)
        H = np.fft.ifft2(dhd + dhu, axes=(-2, -1))
        iy, ix = g.punkte_index(pts[:, 0], pts[:, 1])
        b = _streu(km, w, [src], pts)
        for k, c in enumerate(['Hx', 'Hy', 'Hz']):
            e = np.max(np.abs(H[k][iy, ix] - b[c])) / np.max(np.abs(b[c])); worst = max(worst, e)
            print(f"f = {f:5.0f} Hz {c}: grid route vs kugel.Modell {e:.1e}")
    assert worst < 3e-3, worst


def test_nullkontrast():
    q = geerdeter_dipol((0.0, -1500.0), laenge=1.0)
    g = huegel.Gitter(q, n=256, dx=10.0, s=150.0)
    km = Modell(ST, Kugel(60.0, SIG), (0.0, 0.0, 140.0), nmax=8)
    kg = KugelGelaende(km, g)
    a = huegel.koeffizienten([10.0], g, ST, nord=6, kugel=kg)
    b = huegel.koeffizienten([10.0], g, ST, nord=6)
    e = max(np.max(np.abs(x - y)) / np.max(np.abs(y)) for x, y in zip(a, b))
    print(f"null contrast: coupled vs layered series {e:.1e}")
    assert e < 1e-12, e


def test_drehung_mit_kugel():
    res = []
    for q in (geerdeter_dipol((0.0, -1500.0), psi=0.0), geerdeter_dipol((1500.0, 0.0), psi=np.pi / 2)):
        g = huegel.Gitter(q, n=256, dx=10.0, s=150.0)
        km = Modell(ST, Kugel(60.0, 1.0), (0.0, 0.0, 140.0), nmax=8)
        res.append(huegel.koeffizienten([30.0], g, ST, nord=6, kugel=KugelGelaende(km, g)))
    ref = huegel.koeffizienten([30.0], g, ST, nord=6)
    n = g.n; i = np.arange(1, n)
    worst = 0.0
    for C_a, C_b in zip(res[0], res[1]):
        A = C_a[:, :, :, 0, i]; B = C_b[:, :, :, 1, n - i]
        rot = np.stack([-A[:, :, 1], A[:, :, 0], A[:, :, 2]], axis=2)
        worst = max(worst, np.max(np.abs(B - rot)) / np.max(np.abs(A)))
    eff = np.max(np.abs(res[1][1][0, 1:] - ref[1][0, 1:])) / np.max(np.abs(ref[1][0, 1:]))
    print(f"rotation with sphere: {worst:.1e}; sphere changes the terrain orders by {eff:.2f} (relative)")
    assert worst < 1e-9 and eff > 1e-3, (worst, eff)
