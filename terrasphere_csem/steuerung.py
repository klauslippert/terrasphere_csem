"""Control file of a computation (KH-026).  Every computation lives in its own folder with the control file
`steuer.toml`; the series results are written to `<folder>/ergebnis/`, the elfe3D input to `<folder>/elfe3d/`.

Control file (TOML, SI units, z up, positions on the surface (x, y); depth positive down):

    name = "inline_drei"
    [gelaende]                     # Gaussian hill/valley h = H exp(-r^2/2s^2), centre (0, 0)
    breite = 150.0                 # s (m)
    hoehen = [0.0, 30.0, -30.0]    # H (m): 0 flat, > 0 hill, < 0 valley; evaluated from the same coefficients
    [schichten]                    # top layer carries the terrain; all interfaces below the valley floor
    tiefen = [0.0, 100.0, 350.0]   # depths of the layer tops (first = 0)
    leitfaehigkeit = [0.01, 0.005, 0.03]   # S/m
    [kugel]                        # optional; omit the block for no sphere
    radius = 60.0; leitfaehigkeit = 1.0; mu_r = 1.0; mitte = [0.0, 0.0, 220.0]; grad = 10
    [[sender]]                     # one or more blocks, summed
    typ = "draht"                  # draht | dipol | rechteck | polygon | kreis
    von = [-1750.0, 0.0]; nach = [-1250.0, 0.0]; strom = 1.0; gauss = 96
    # dipol: mitte, richtung (deg from x towards y), laenge, strom | rechteck: ecke1, ecke2, strom |
    # polygon: ecken [[x, y], ...], strom | kreis: mitte, radius, strom, segmente
    [empfaenger]
    profile = ["x", "y"]           # profiles through the summit along the x-axis (y = 0) and the y-axis (x = 0)
    halblaenge = 600.0; abstand = 20.0
    [frequenzen]                   # either fixed frequencies ...
    liste = [0.1, 1.0, 1000.0, 100000.0]
    # [zeit]                       # ... or a time window (frequencies follow from the filter, lagged grid)
    # tmin = 1e-5; tmax = 10.0; signal = "aus"      # aus = switch-off step, ein = switch-on step
    [ausgabe]
    formate = ["npz", "csv"]       # any of npz, csv; each file holds all components
    [numerik]                      # defaults in NUMERIK
    verfahren = "hankel"           # hankel (default, no periodic box) | fft (2-D FFT, box correction)
    m_max = 40; n_phi = 128; npan_r = 32; npan_l = 32; lam_max = 0.25     # hankel
    n = 512; dx = 10.0; ordnungen = 20; tiefpass = 0.2; fenster_radius = 860.0; fenster_breite = 40.0
    boxkorrektur = true; box_fmax = 100.0
"""
import os
import tomllib
import numpy as np

from .quellen import draht, geerdeter_dipol, polygon_draht, kreisschleife
from .schichten import Schichten

NUMERIK = dict(verfahren='hankel', m_max=40, n_phi=128, npan_r=32, npan_l=32, lam_max=0.25,
               n=512, dx=10.0, ordnungen=20, tiefpass=0.2, fenster_radius=860.0, fenster_breite=40.0, fmax_reihe=2e6,
               toleranz=1e-3,
               boxkorrektur=True, box_fmax=100.0, box_dx=20.0, box_n=(256, 512, 1024), box_tiefpass=0.12,
               box_ordnungen=12)


class Steuerung:
    def __init__(self, ordner):
        self.ordner = os.path.abspath(ordner)
        with open(os.path.join(self.ordner, 'steuer.toml'), 'rb') as fh:
            self.d = tomllib.load(fh)
        d = self.d
        self.name = d.get('name', os.path.basename(self.ordner))
        g = d['gelaende']
        self.s = float(g['breite']); self.hoehen = [float(v) for v in g.get('hoehen', [0.0])]
        sch = d['schichten']
        self.schichten = Schichten(sch['tiefen'], sch['leitfaehigkeit'])
        if self.schichten.J > 1:
            assert self.schichten.tiefen[1] > -min(0.0, min(self.hoehen)), \
                "all layer interfaces must lie below the valley floor"
        self.kugel = d.get('kugel')
        if ('frequenzen' in d) == ('zeit' in d):
            raise ValueError("give either [frequenzen] or [zeit]")
        self.bereich = 'frequenz' if 'frequenzen' in d else 'zeit'
        if self.bereich == 'frequenz':
            self.freqs = np.array([float(v) for v in d['frequenzen']['liste']])
        else:
            from .zeit import zeitachse, filter_frequenzen
            z = d['zeit']
            self.t = zeitachse(float(z['tmin']), float(z['tmax']))
            self.signal = z.get('signal', 'aus'); assert self.signal in ('aus', 'ein')
            self.freqs = filter_frequenzen(self.t)                   # all filter abscissae (the series up to fmax)
        a = d.get('ausgabe', {})
        self.formate = [f.lower() for f in a.get('formate', ['npz'])]
        assert set(self.formate) <= {'npz', 'csv'} and self.formate, "ausgabe.formate: npz and/or csv"
        e = d['empfaenger']
        self.profile = list(e.get('profile', ['x', 'y']))
        self.halb = float(e.get('halblaenge', 600.0)); self.abstand = float(e.get('abstand', 20.0))
        self.num = dict(NUMERIK); self.num.update(d.get('numerik', {}))
        self.sender = d['sender']

    # ------------------------------------------------------------------ objects
    def quellen(self):
        out = []
        for q in self.sender:
            t = q['typ']; I = float(q.get('strom', 1.0))
            if t == 'draht':
                out += draht(q['von'], q['nach'], I, int(q.get('gauss', 96)))
            elif t == 'dipol':
                out += geerdeter_dipol(q['mitte'], np.radians(float(q.get('richtung', 0.0))), float(q['laenge']), I,
                                       n_gauss=q.get('gauss'))
            elif t == 'rechteck':
                (x1, y1), (x2, y2) = q['ecke1'], q['ecke2']
                out += polygon_draht([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], I, int(q.get('gauss', 24)),
                                     geschlossen=True)
            elif t == 'polygon':
                out += polygon_draht([tuple(p) for p in q['ecken']], I, int(q.get('gauss', 24)), geschlossen=True)
            elif t == 'kreis':
                out += kreisschleife(q['mitte'], float(q['radius']), I, int(q.get('segmente', 256)),
                                     int(q.get('gauss', 3)))
            else:
                raise ValueError(f"unknown source type {t}")
        return out

    def empfaenger(self):
        """Receiver positions (px, py) and the profile index of each: x-profile (y = 0) and/or y-profile (x = 0)."""
        a = np.arange(-self.halb, self.halb + 1e-6, self.abstand)
        px, py, prof = [], [], []
        for p in self.profile:
            if p == 'x':
                px += list(a); py += [0.0] * len(a); prof += ['x'] * len(a)
            elif p == 'y':
                px += [0.0] * len(a); py += list(a); prof += ['y'] * len(a)
        return np.array(px), np.array(py), prof

    def hankelgitter(self):
        from .huegel_hankel import Hankelgitter
        px, py, _ = self.empfaenger()
        nu = self.num
        return Hankelgitter(self.quellen(), (px, py), s=self.s, M=int(nu['m_max']), n_phi=int(nu['n_phi']),
                            r_win=float(nu['fenster_radius']), w_win=float(nu['fenster_breite']),
                            lam_max=float(nu['lam_max']), kc=float(nu['tiefpass']), npan_r=int(nu['npan_r']),
                            npan_l=int(nu['npan_l']))

    def gitter(self, n=None, dx=None):
        from .huegel import Gitter
        return Gitter(self.quellen(), n=n or int(self.num['n']), dx=dx or float(self.num['dx']), s=self.s,
                      r_win=float(self.num['fenster_radius']), w_win=float(self.num['fenster_breite']))

    def kugel_kopplung(self, g):
        if not self.kugel:
            return None
        from .kugel import Modell, Kugel
        from .kugel_gelaende import KugelGelaende, KugelGelaendeHankel
        from .huegel_hankel import Hankelgitter
        k = self.kugel
        km = Modell(self.schichten, Kugel(float(k['radius']), float(k['leitfaehigkeit']), float(k.get('mu_r', 1.0))),
                    tuple(k['mitte']), nmax=int(k.get('grad', 10)))
        return KugelGelaendeHankel(km, g) if isinstance(g, Hankelgitter) else KugelGelaende(km, g)

    def h(self, x, y, H):
        return H * np.exp(-(np.asarray(x) ** 2 + np.asarray(y) ** 2) / (2 * self.s ** 2))
