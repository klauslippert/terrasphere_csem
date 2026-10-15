"""Named model cases of the paper scripts (coupling.py, fig_coupling.py, the terrain figures).  Gaussian hill of width s = 150 m centred
at the origin, receivers on the two profiles through the summit (x- and y-axis), top layer 0.01 S/m.
  'wire_halfspace' grounded wire (-500, -1500) .. (500, -1500) m, 1 A, over the half-space 0.01 S/m
  'wire_layered'   grounded wire (-500, -1500) .. (500, -1500) m, 1 A, over two layers: 0.01 S/m to 300 m, 0.003 S/m below
  'wire_sphere'    as 'wire_layered' plus a sphere (radius 60 m, 1 S/m) centred 140 m below the flat ground (z = -140 m)
"""
from terrasphere_csem.sources import grounded_wire
from terrasphere_csem.layers import LayerStack

SIGMA = 0.01
S_HILL = 150.0

_WIRE = lambda: grounded_wire((-500.0, -1500.0), (500.0, -1500.0), 1.0, n_gauss=96)
_TWO = lambda: LayerStack([0.0, 300.0], [SIGMA, 0.003])

CASES = {
    'wire_halfspace': dict(sources=_WIRE, ground=lambda: SIGMA),
    'wire_layered': dict(sources=_WIRE, ground=_TWO),
    'wire_sphere': dict(sources=_WIRE, ground=_TWO, sphere=dict(b=60.0, sigma_s=1.0, mu_r=1.0,
                                                                 center=(0.0, 0.0, 140.0), nmax=10)),
}


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
