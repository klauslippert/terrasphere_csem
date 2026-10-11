"""Named model cases for the scripts.  Common geometry: Gaussian hill/valley of width
s = 150 m centred at the origin, profiles +-600 m through the summit, times 1e-5 .. 10 s; top layer 0.01 S/m.
Each case: sources (on the flat ground off the hill), ground (sigma or LayerStack), optional sphere.
  'wire'            grounded wire (-500, -1500) .. (500, -1500) m, 1 A, homogeneous ground
  'dipole'          grounded point dipole at (0, -1500) m, x-directed, 1 A m, homogeneous ground
  'rectangle'       rectangular loop 2.6 km x 2.6 km centred on the hill, 1 A, homogeneous ground
  'circle'          circular loop, radius 1.3 km, centred on the hill, 1 A, homogeneous ground
  'wire_layered' wire over two layers: 0.01 S/m to 300 m, 0.003 S/m below
  'wire_sphere'    as 'wire_layered' plus a sphere (radius 60 m, 1 S/m) centred 140 m below the summit
  'inline_*'       500 m wire on the x-axis left of the hill, (-1750, 0) .. (-1250, 0) m (in line with the x-profile)
  'broadside_*'    500 m wire parallel to y left of the hill, (-1500, -250) .. (-1500, 250) m
                   (* = 'layers' or 'sphere', ground and sphere as above).  Centre 1500 m from the
                   summit; the nearest electrode (inline) at 1250 m, where the window 0.5 erfc((r - 860)/40) is < 1e-40.
  'inline_three', 'broadside_three'            three layers: 0.01 S/m to 100 m, 0.005 S/m to 350 m, 0.03 S/m below
  'inline_three_sphere', 'broadside_three_sphere' plus a sphere in the MIDDLE layer: radius 60 m, 1 S/m, centre 220 m
                   below the summit (160-280 m deep; 60 m below / 70 m above the interfaces)
  'loop_offset_three(_sphere)'  square loop 400 m x 400 m, centre (-1500, 0), axis-parallel, 1 A, counter-clockwise
                   seen from above; nearest side 1300 m from the summit; three layers (+ sphere)
  'loop_large_three(_sphere)'   square loop 2.6 km x 2.6 km centred on the hill (in-loop receivers), 1 A
  Receivers in all cases: the two profiles through the summit along the x-axis (y = 0) and the y-axis (x = 0), +-600 m.
"""
from terrasphere_csem.sources import grounded_wire, grounded_dipole, polygon_wire, circular_loop
from terrasphere_csem.layers import LayerStack

SIGMA = 0.01
S_HILL = 150.0
PROFILE_HALF = 600.0
T_MIN, T_MAX = 1e-5, 10.0

_WIRE = lambda: grounded_wire((-500.0, -1500.0), (500.0, -1500.0), 1.0, n_gauss=96)
_A = 1300.0
_TWO = lambda: LayerStack([0.0, 300.0], [SIGMA, 0.003])

CASES = {
    'wire': dict(sources=_WIRE, ground=lambda: SIGMA),
    'dipole': dict(sources=lambda: grounded_dipole((0.0, -1500.0), psi=0.0, length=1.0, current=1.0), ground=lambda: SIGMA),
    'rectangle': dict(sources=lambda: polygon_wire([(-_A, -_A), (_A, -_A), (_A, _A), (-_A, _A)], 1.0, n_gauss=48,
                                                   closed=True), ground=lambda: SIGMA),
    'circle': dict(sources=lambda: circular_loop((0.0, 0.0), _A, 1.0, n_seg=256, n_gauss=3), ground=lambda: SIGMA),
    'wire_layered': dict(sources=_WIRE, ground=_TWO),
    'wire_sphere': dict(sources=_WIRE, ground=_TWO, sphere=dict(b=60.0, sigma_s=1.0, mu_r=1.0,
                                                                 center=(0.0, 0.0, 140.0), nmax=10)),
}
_SPHERE = dict(b=60.0, sigma_s=1.0, mu_r=1.0, center=(0.0, 0.0, 140.0), nmax=10)
_INLINE = lambda: grounded_wire((-1750.0, 0.0), (-1250.0, 0.0), 1.0, n_gauss=96)
_BROAD = lambda: grounded_wire((-1500.0, -250.0), (-1500.0, 250.0), 1.0, n_gauss=96)
for _n, _q in (('inline', _INLINE), ('broadside', _BROAD)):
    CASES[f'{_n}_layered'] = dict(sources=_q, ground=_TWO)
    CASES[f'{_n}_sphere'] = dict(sources=_q, ground=_TWO, sphere=_SPHERE)
_THREE = lambda: LayerStack([0.0, 100.0, 350.0], [SIGMA, 0.005, 0.03])
_SPHERE3 = dict(b=60.0, sigma_s=1.0, mu_r=1.0, center=(0.0, 0.0, 220.0), nmax=10)
for _n, _q in (('inline', _INLINE), ('broadside', _BROAD)):
    CASES[f'{_n}_three'] = dict(sources=_q, ground=_THREE)
    CASES[f'{_n}_three_sphere'] = dict(sources=_q, ground=_THREE, sphere=_SPHERE3)
_LOOP_OFFSET = lambda: polygon_wire([(-1700.0, -200.0), (-1300.0, -200.0), (-1300.0, 200.0), (-1700.0, 200.0)], 1.0,
                                    n_gauss=24, closed=True)
_LOOP_LARGE = lambda: polygon_wire([(-_A, -_A), (_A, -_A), (_A, _A), (-_A, _A)], 1.0, n_gauss=48, closed=True)
for _n, _q in (('loop_offset', _LOOP_OFFSET), ('loop_large', _LOOP_LARGE)):
    CASES[f'{_n}_three'] = dict(sources=_q, ground=_THREE)
    CASES[f'{_n}_three_sphere'] = dict(sources=_q, ground=_THREE, sphere=_SPHERE3)


def sources(name):
    """Source list (Dipole objects) of the named case."""
    return CASES[name]['sources']()


def ground(name):
    """Ground of the named case: a conductivity [S/m] or a LayerStack."""
    return CASES[name]['ground']()


def sphere(name, g):
    """SphereTerrain for the case (or None), for the Grid g."""
    k = CASES[name].get('sphere')
    if k is None:
        return None
    from terrasphere_csem.sphere import SphereModel, Sphere
    from terrasphere_csem.sphere_terrain import SphereTerrain
    from terrasphere_csem.terrain import ground_layers
    km = SphereModel(ground_layers(ground(name)), Sphere(k['b'], k['sigma_s'], k['mu_r']), k['center'], nmax=k['nmax'])
    return SphereTerrain(km, g)
