"""Hill (huegel.py) in the unified package:
(1) closed-form surface fields of rotated HEDs (halbraum.py) against the layered-earth code (eindim.py, one layer):
    the two primary-field codes of the package agree, same conventions (e^{-iwt}, z up, psi);
(2) regression: the converted hill code with the reference wire equals the complex conjugate of the original
    gausshill code (e^{+iwt}) -- the time-convention change touches only gamma;
(3) grounded point dipole: direct-current limit of the H-series against the independent potential expansion;
(4) grounded point dipole: rotating source and receivers by 90 degrees about the hill axis rotates the fields."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from terrasphere_csem import halbraum, huegel, huegel_gleichstrom
from terrasphere_csem.quellen import draht, geerdeter_dipol
from terrasphere_csem.schichten import Schichten
from terrasphere_csem.eindim import primaerfeld

GAUSSHILL = 'path/to/gausshill/code'
SIGMA, S = 0.01, 150.0


def test_halbraum_vs_eindim():
    st = Schichten([0.0], [SIGMA])
    q = geerdeter_dipol((120.0, -40.0), psi=0.7, laenge=1.0)
    pts = np.array([[700.0, 300.0], [-500.0, 900.0], [300.0, -1100.0], [-900.0, -600.0]])
    worst = 0.0
    for f in [0.1, 10.0, 1e3]:          # at 30 kHz the 1-mm offset itself changes the fields by 5e-5
        a = halbraum.oberflaeche(pts[:, 0], pts[:, 1], f, SIGMA, q)
        # receivers 1 mm below the surface: with source and receiver exactly in the same plane the Hankel filter of
        # the layered code loses E to 5e-4 (KH-006); tangential E and H change by O(gamma * 1 mm) ~ 1e-6 over 1 mm
        E, H = primaerfeld(st, q, [(x, y, 1e-3) for x, y in pts], 2 * np.pi * f)
        for c, ref in [('Ex', E[:, 0]), ('Ey', E[:, 1]), ('Hx', H[:, 0]), ('Hy', H[:, 1]), ('Hz', H[:, 2])]:
            e = np.max(np.abs(a[c][0] - ref)) / np.max(np.abs(ref))
            worst = max(worst, e)
            print(f"f = {f:8.3g} Hz {c}: closed form vs layered code {e:.1e}")
    assert worst < 2e-5, worst


def test_regression_gausshill():
    if not os.path.isdir(GAUSSHILL):
        print("gausshill code not found, skipped"); return
    sys.path.insert(0, GAUSSHILL)
    from gauss import lauf as L_alt, reihe as R_alt, konfig as K_alt
    freqs = [0.5, 300.0]; nord = 8
    g_alt = R_alt.Gitter(n=256, dx=10.0)
    CEa, CHa, _ = L_alt.koeffizienten(freqs, nord=nord, g=g_alt)
    w = K_alt.WIRE
    q = draht(w['a'], w['b'], w['current'], n_gauss=96)
    g = huegel.Gitter(q, n=256, dx=10.0, s=K_alt.S_HILL, r_win=K_alt.R_WIN, w_win=K_alt.W_WIN)
    assert np.array_equal(g.maske, g_alt.maske)
    CE, CH = huegel.koeffizienten(freqs, g, K_alt.SIGMA, nord=nord, kc=K_alt.KC)
    for nm, a, b in [('E', CE, CEa), ('H', CH, CHa)]:
        e = np.max(np.abs(a - np.conj(b))) / np.max(np.abs(b))
        print(f"{nm}: |new - conj(old)| / max = {e:.1e}")
        assert e < 1e-10, e


def test_dipol_gleichstrom():
    q = geerdeter_dipol((0.0, -1500.0), psi=0.0, laenge=100.0)
    g = huegel.Gitter(q, n=256, dx=10.0, s=S)
    nord = 12
    CE, CH = huegel.koeffizienten([1e-6], g, SIGMA, nord=nord)
    CEd = huegel_gleichstrom.koeffizienten(g, SIGMA, nord)
    sel = np.abs(g.x) <= 600.0
    for Hh in [30.0, -30.0]:
        a = huegel.summe(CE[0], Hh); b = huegel.summe(CEd, Hh); b0 = CEd[0]
        for lin in (0, 1):
            A, B, B0 = (v[:2, lin][:, sel] for v in (a, b, b0))
            eff = np.abs(B - B0).max() / np.abs(B0).max()
            dev = np.abs(A - B).max() / np.abs(B).max()
            print(f"H = {Hh:+.0f} m, line {lin}: terrain effect {eff:.3f}, |H-series - potential| / max = {dev:.1e}")
            assert eff > 0.01 and dev < 1e-6, (eff, dev)


def test_dipol_drehung():
    """Source A: x-directed dipole at (0, -1500); source B: A rotated by +90 deg about z: at (1500, 0), y-directed.
    Point (0, y) on the y-profile of A maps to (-y, 0) on the x-profile of B; fields rotate: B_x = -A_y, B_y = A_x."""
    qa = geerdeter_dipol((0.0, -1500.0), psi=0.0, laenge=50.0)
    qb = geerdeter_dipol((1500.0, 0.0), psi=np.pi / 2, laenge=50.0)
    res = []
    for q in (qa, qb):
        g = huegel.Gitter(q, n=256, dx=10.0, s=S)
        res.append(huegel.koeffizienten([3.0, 2e3], g, SIGMA, nord=10))
    n = g.n; i = np.arange(1, n)                       # A: y index i (y = (i - n/2) dx)  ->  B: x = -y, index n - i
    worst = 0.0
    for C_a, C_b in zip(res[0], res[1]):
        A = C_a[:, :, :, 0, i]; B = C_b[:, :, :, 1, n - i]
        rot = np.stack([-A[:, :, 1], A[:, :, 0], A[:, :, 2]], axis=2)
        e = np.max(np.abs(B - rot)) / np.max(np.abs(A)); worst = max(worst, e)
    print(f"rotation by 90 deg: |B - R A| / max = {worst:.1e}")
    assert worst < 1e-10, worst


def test_punkte_wie_linien():
    """Receivers at arbitrary grid points (full inverse FFT) give the same coefficients as the line extraction."""
    q = geerdeter_dipol((300.0, -1500.0), psi=0.4, laenge=10.0)
    g = huegel.Gitter(q, n=256, dx=10.0, s=S)
    xs = np.array([-300.0, -100.0, 0.0, 150.0, 500.0])
    iy, ix = g.punkte_index(np.concatenate([np.zeros(5), xs]), np.concatenate([xs, np.zeros(5)]))
    CE_l, CH_l = huegel.koeffizienten([30.0], g, SIGMA, nord=8)
    CE_p, CH_p = huegel.koeffizienten([30.0], g, SIGMA, nord=8, punkte=(iy, ix))
    il = np.rint(xs / g.dx).astype(int) + g.n // 2
    for a, b in [(CE_l, CE_p), (CH_l, CH_p)]:
        ref = np.concatenate([a[..., 0, il], a[..., 1, il]], axis=-1)
        e = np.max(np.abs(ref - b)) / np.max(np.abs(ref))
        print(f"points vs lines: {e:.1e}")
        assert e < 1e-12, e


def test_schleifen_primaer():
    """Rectangular loop (4 wires) and circular loop (256-gon) on the surface: closed-form half-space sum against the
    layered code (rectangle: the same HEDs; circle: a 1024-gon), receivers inside and outside the loop.  The layered
    code gives H exactly at the surface and E 1 mm below it (KH-006); at 1 mm the small horizontal H inside a loop
    would already change by ~1e-3 of itself through the large d_z H_x = d_x H_z."""
    from terrasphere_csem.quellen import polygon_draht, kreisschleife, Dipol
    st = Schichten([0.0], [SIGMA])
    pts = np.array([[0.0, 0.0], [120.0, -60.0], [900.0, 300.0], [-700.0, -800.0]])
    rect = polygon_draht([(-400.0, -300.0), (400.0, -300.0), (400.0, 300.0), (-400.0, 300.0)], 1.0, n_gauss=12,
                         geschlossen=True)
    kreis = kreisschleife((0.0, 0.0), 350.0, 1.0, n_seg=256, n_gauss=3)
    kreis_ref = Dipol('LOOP', (0.0, 0.0, 0.0), moment=1.0, radius=350.0).als_polygon(n_seg=1024, n_gauss=3)
    worst = {}
    for name, a_src, b_src in [('rectangle', rect, rect), ('circle', kreis, kreis_ref)]:
        for f in [1.0, 100.0]:
            a = halbraum.oberflaeche(pts[:, 0], pts[:, 1], f, SIGMA, a_src)
            E, _ = primaerfeld(st, b_src, [(x, y, 1e-3) for x, y in pts], 2 * np.pi * f)
            _, H = primaerfeld(st, b_src, [(x, y, 0.0) for x, y in pts], 2 * np.pi * f)
            for c, ref in [('Ex', E[:, 0]), ('Ey', E[:, 1]), ('Hx', H[:, 0]), ('Hy', H[:, 1]), ('Hz', H[:, 2])]:
                e = np.max(np.abs(a[c][0] - ref)) / np.max(np.abs(ref))
                worst[name] = max(worst.get(name, 0.0), e)
                print(f"{name:9s} f = {f:5.0f} Hz {c}: {e:.1e}")
    assert worst['rectangle'] < 2e-4 and worst['circle'] < 2e-4, worst


def test_schleife_um_huegel():
    """Square loop (side 2.6 km) centred on the hill, receivers on the summit lines: the configuration is invariant
    under a 90-degree rotation, so the x-line fields are the rotated y-line fields; E vanishes at DC (no galvanic
    part) and the terrain still changes H_z."""
    from terrasphere_csem.quellen import polygon_draht
    a = 1300.0
    q = polygon_draht([(-a, -a), (a, -a), (a, a), (-a, a)], 1.0, n_gauss=48, geschlossen=True)
    g = huegel.Gitter(q, n=256, dx=10.0, s=S)
    CE, CH = huegel.koeffizienten([10.0], g, SIGMA, nord=10)
    n = g.n; i = np.arange(1, n)
    worst = 0.0
    for C in (CE[0], CH[0]):
        A = C[:, :, 0, i]; B = C[:, :, 1, n - i]
        rot = np.stack([-A[:, 1], A[:, 0], A[:, 2]], axis=1)
        worst = max(worst, np.max(np.abs(B - rot)) / np.max(np.abs(A)))
    eff = np.abs(huegel.summe(CH[0], 30.0)[2] - CH[0, 0, 2]).max() / np.abs(CH[0, 0, 2]).max()
    E0 = halbraum.oberflaeche(np.array([0.0, 200.0]), np.array([0.0, 100.0]), 0.0, SIGMA, q)
    e_dc = max(np.abs(E0['Ex'][0]).max(), np.abs(E0['Ey'][0]).max())
    e_ac = np.abs(CE[0, 0, :2]).max()
    print(f"loop around the hill: rotation symmetry {worst:.1e}, H_z terrain effect (30 m) {eff:.3f}, "
          f"|E| at DC / |E| at 10 Hz = {e_dc / e_ac:.1e}")
    assert worst < 1e-10 and eff > 1e-3 and e_dc / e_ac < 1e-8


def test_schichten_primaer():
    """Layered surface fields (closed-form top layer + tabulated reflection correction) against the layered code:
    the layer correction alone (layered - half-space) is compared, E 1 mm deep, H at the surface."""
    st = Schichten([0.0, 80.0, 300.0], [SIGMA, 0.1, 0.002]); hs = Schichten([0.0], [SIGMA])
    q = geerdeter_dipol((50.0, -30.0), psi=0.6, laenge=1.0)
    pts = np.array([[700.0, 300.0], [-500.0, 900.0], [300.0, -1100.0], [-900.0, -600.0]])
    worst = 0.0
    for f in [0.1, 10.0, 1e3, 3e4]:
        a = halbraum.oberflaeche_schichten(pts[:, 0], pts[:, 1], f, st, q)
        b = halbraum.oberflaeche(pts[:, 0], pts[:, 1], f, SIGMA, q)
        w = 2 * np.pi * f
        E1, _ = primaerfeld(st, q, [(x, y, 1e-3) for x, y in pts], w); E0, _ = primaerfeld(hs, q, [(x, y, 1e-3) for x, y in pts], w)
        _, H1 = primaerfeld(st, q, [(x, y, 0.0) for x, y in pts], w); _, H0 = primaerfeld(hs, q, [(x, y, 0.0) for x, y in pts], w)
        for c, ref, tot in [('Ex', E1[:, 0] - E0[:, 0], E1[:, 0]), ('Ey', E1[:, 1] - E0[:, 1], E1[:, 1]),
                            ('Hx', H1[:, 0] - H0[:, 0], H1[:, 0]), ('Hy', H1[:, 1] - H0[:, 1], H1[:, 1]),
                            ('Hz', H1[:, 2] - H0[:, 2], H1[:, 2])]:
            e = np.max(np.abs((a[c][0] - b[c][0]) - ref)) / np.max(np.abs(tot))
            worst = max(worst, e)
            print(f"f = {f:8.3g} Hz {c}: layer correction vs layered code {e:.1e} (of the total field)")
    assert worst < 1e-5, worst


def test_schichten_gleichstrom():
    """Terrain over layered ground, direct-current limit: H-series (f = 1e-6 Hz, layered primary, TE/TM reflections)
    against the potential expansion (potential reflections), hill and valley, conductive and resistive basement."""
    q = geerdeter_dipol((0.0, -1500.0), psi=0.0, laenge=100.0)
    g = huegel.Gitter(q, n=256, dx=10.0, s=S)
    sel = np.abs(g.x) <= 600.0
    for st in [Schichten([0.0, 100.0], [SIGMA, 0.1]), Schichten([0.0, 60.0, 200.0], [SIGMA, 0.001, 0.05])]:
        CE, CH = huegel.koeffizienten([1e-6], g, st, nord=12)
        CEd = huegel_gleichstrom.koeffizienten(g, st, 12)
        CEh = huegel_gleichstrom.koeffizienten(g, SIGMA, 12)
        for Hh in [30.0, -30.0]:
            a = huegel.summe(CE[0], Hh); b = huegel.summe(CEd, Hh); bh = huegel.summe(CEh, Hh)
            for lin in (0, 1):
                A, B, B0, BH = (v[:2, lin][:, sel] for v in (a, b, CEd[0], bh))
                eff = np.abs(B - B0).max() / np.abs(B0).max()
                lay = np.abs(B - BH).max() / np.abs(B).max()
                dev = np.abs(A - B).max() / np.abs(B).max()
                print(f"sigma {st.sigma_g}, H = {Hh:+.0f}, line {lin}: terrain {eff:.3f}, layers {lay:.3f}, "
                      f"|H-series - potential| / max = {dev:.1e}")
                assert eff > 0.01 and lay > 0.01 and dev < 1e-5, (eff, lay, dev)
