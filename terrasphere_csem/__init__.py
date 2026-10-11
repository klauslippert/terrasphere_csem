"""terrasphere_csem (project terrasphere-csem) -- semi-analytic frequency- and time-domain EM responses for two 3-D targets with common sources,
conventions and time transform:

  * kugel.py   : conductive (permeable) sphere in a layered half-space (all multipoles, TE and TM; formerly the
                 package `layersphere` of kugel-semianalytik)
  * huegel.py  : Gaussian hill or valley h = H exp(-r^2/2s^2) on a homogeneous half-space, field expansion in H
                 (formerly gausshill/code/gauss), sources on the flat ground off the hill

Conventions (used throughout, all modules):
  time dependence exp(-i w t); quasi-static (no displacement currents); mu = mu0 except in the sphere;
  k^2 = i w mu0 sigma with Im k > 0; u = sqrt(lambda^2 - k^2) with Re u > 0.
  Positions: (x, y, zeta) with zeta = depth, positive DOWN (surface zeta = 0).
  Field components: (x, y, z) with z pointing UP (right-handed); dB/dt = -i w mu0 H.
  Terrain: surface at z = h(x, y), i.e. zeta = -h.
Sources (quellen.py): Dipol('HED' | 'VMD' | 'HMD' | 'LOOP'), grounded wires and polygon loops as HED sums, the
grounded point dipole `geerdeter_dipol`.  The hill accepts HED-type sources on the surface (HED, wires, polygons).
Time domain (zeit.py): Key's 201-point sine/cosine filter on a lagged time grid; Talbot inversion.
"""
MU0 = 4e-7 * 3.141592653589793
__version__ = "0.1"
