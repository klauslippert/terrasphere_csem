"""Named model cases for the scripts.  Common geometry (gausshill reference, GH-012): Gaussian hill/valley of width
s = 150 m centred at the origin, profiles +-600 m through the summit, times 1e-5 .. 10 s; top layer 0.01 S/m.
Each case: sources (on the flat ground off the hill), ground (sigma or Schichten), optional sphere.
  'draht'          grounded wire (-500, -1500) .. (500, -1500) m, 1 A, homogeneous ground (gausshill reference)
  'dipol'          grounded point dipole at (0, -1500) m, x-directed, 1 A m, homogeneous ground
  'rechteck'       rectangular loop 2.6 km x 2.6 km centred on the hill, 1 A, homogeneous ground
  'kreis'          circular loop, radius 1.3 km, centred on the hill, 1 A, homogeneous ground
  'draht_schichten' wire over two layers: 0.01 S/m to 300 m, 0.003 S/m below
  'draht_kugel'    as 'draht_schichten' plus a sphere (radius 60 m, 1 S/m) centred 140 m below the summit
  'inline_*'       500 m wire on the x-axis left of the hill, (-1750, 0) .. (-1250, 0) m (in line with the x-profile)
  'broadside_*'    500 m wire parallel to y left of the hill, (-1500, -250) .. (-1500, 250) m
                   (* = 'schichten' or 'kugel', ground and sphere as above; KH-020/021).  Centre 1500 m from the
                   summit; the nearest electrode (inline) at 1250 m, where the window 0.5 erfc((r - 860)/40) is < 1e-40.
  'inline_drei', 'broadside_drei'            three layers: 0.01 S/m to 100 m, 0.005 S/m to 350 m, 0.03 S/m below
  'inline_drei_kugel', 'broadside_drei_kugel' plus a sphere in the MIDDLE layer: radius 60 m, 1 S/m, centre 220 m
                   below the summit (160-280 m deep; 60 m below / 70 m above the interfaces) (KH-022)
  'schleife_offset_drei(_kugel)'  square loop 400 m x 400 m, centre (-1500, 0), axis-parallel, 1 A, counter-clockwise
                   seen from above; nearest side 1300 m from the summit; three layers (+ sphere) (KH-023)
  'schleife_gross_drei(_kugel)'   square loop 2.6 km x 2.6 km centred on the hill (in-loop receivers), 1 A
  Receivers in all cases: the two profiles through the summit along the x-axis (y = 0) and the y-axis (x = 0), +-600 m.
"""
from terrasphere_csem.quellen import draht, geerdeter_dipol, polygon_draht, kreisschleife
from terrasphere_csem.schichten import Schichten

SIGMA = 0.01
S_HILL = 150.0
PROFILE_HALF = 600.0
T_MIN, T_MAX = 1e-5, 10.0

_DRAHT = lambda: draht((-500.0, -1500.0), (500.0, -1500.0), 1.0, n_gauss=96)
_A = 1300.0
_ZWEI = lambda: Schichten([0.0, 300.0], [SIGMA, 0.003])

FAELLE = {
    'draht': dict(quellen=_DRAHT, boden=lambda: SIGMA),
    'dipol': dict(quellen=lambda: geerdeter_dipol((0.0, -1500.0), psi=0.0, laenge=1.0, strom=1.0), boden=lambda: SIGMA),
    'rechteck': dict(quellen=lambda: polygon_draht([(-_A, -_A), (_A, -_A), (_A, _A), (-_A, _A)], 1.0, n_gauss=48,
                                                   geschlossen=True), boden=lambda: SIGMA),
    'kreis': dict(quellen=lambda: kreisschleife((0.0, 0.0), _A, 1.0, n_seg=256, n_gauss=3), boden=lambda: SIGMA),
    'draht_schichten': dict(quellen=_DRAHT, boden=_ZWEI),
    'draht_kugel': dict(quellen=_DRAHT, boden=_ZWEI, kugel=dict(b=60.0, sigma_s=1.0, mu_r=1.0,
                                                                 zentrum=(0.0, 0.0, 140.0), nmax=10)),
}
_KUGEL = dict(b=60.0, sigma_s=1.0, mu_r=1.0, zentrum=(0.0, 0.0, 140.0), nmax=10)
_INLINE = lambda: draht((-1750.0, 0.0), (-1250.0, 0.0), 1.0, n_gauss=96)
_BROAD = lambda: draht((-1500.0, -250.0), (-1500.0, 250.0), 1.0, n_gauss=96)
for _n, _q in (('inline', _INLINE), ('broadside', _BROAD)):
    FAELLE[f'{_n}_schichten'] = dict(quellen=_q, boden=_ZWEI)
    FAELLE[f'{_n}_kugel'] = dict(quellen=_q, boden=_ZWEI, kugel=_KUGEL)
_DREI = lambda: Schichten([0.0, 100.0, 350.0], [SIGMA, 0.005, 0.03])
_KUGEL3 = dict(b=60.0, sigma_s=1.0, mu_r=1.0, zentrum=(0.0, 0.0, 220.0), nmax=10)
for _n, _q in (('inline', _INLINE), ('broadside', _BROAD)):
    FAELLE[f'{_n}_drei'] = dict(quellen=_q, boden=_DREI)
    FAELLE[f'{_n}_drei_kugel'] = dict(quellen=_q, boden=_DREI, kugel=_KUGEL3)
_LOFF = lambda: polygon_draht([(-1700.0, -200.0), (-1300.0, -200.0), (-1300.0, 200.0), (-1700.0, 200.0)], 1.0,
                              n_gauss=24, geschlossen=True)
_LGROSS = lambda: polygon_draht([(-_A, -_A), (_A, -_A), (_A, _A), (-_A, _A)], 1.0, n_gauss=48, geschlossen=True)
for _n, _q in (('schleife_offset', _LOFF), ('schleife_gross', _LGROSS)):
    FAELLE[f'{_n}_drei'] = dict(quellen=_q, boden=_DREI)
    FAELLE[f'{_n}_drei_kugel'] = dict(quellen=_q, boden=_DREI, kugel=_KUGEL3)


def quellen(name):
    return FAELLE[name]['quellen']()


def boden(name):
    return FAELLE[name]['boden']()


def kugel(name, g):
    """KugelGelaende for the case (or None), for the Gitter g."""
    k = FAELLE[name].get('kugel')
    if k is None:
        return None
    from terrasphere_csem.kugel import Modell, Kugel
    from terrasphere_csem.kugel_gelaende import KugelGelaende
    from terrasphere_csem.huegel import boden_schichten
    km = Modell(boden_schichten(boden(name)), Kugel(k['b'], k['sigma_s'], k['mu_r']), k['zentrum'], nmax=k['nmax'])
    return KugelGelaende(km, g)
