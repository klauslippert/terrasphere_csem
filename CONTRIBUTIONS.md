# Contributions and third-party material

## Third-party data shipped in this repository

The digital linear filter coefficients in `terrasphere_csem/filters/*.txt` were designed by Kerry Key and are
**not** covered by the MIT licence of this package. They are used under the Creative Commons Attribution 4.0
licence (CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>).

| File | Content | Reference | Copyright |
|------|---------|-----------|-----------|
| `key_201_2009.txt` | 201-point Hankel filter, J0 and J1 | Key, K., 2009, 1D inversion of multicomponent, multifrequency marine CSEM data: Methodology and synthetic studies for resolving thin resistive layers: Geophysics, 74(2), F9-F20, doi:10.1190/1.3058434 | 2009 Kerry Key |
| `key_401_2009.txt` | 401-point Hankel filter, J0 and J1 | same as above | 2009 Kerry Key |
| `key_201_2012.txt` | 201-point Fourier filter, sine and cosine | Key, K., 2012, Is the fast Hankel transform faster than quadrature?: Geophysics, 77(3), F21-F30, doi:10.1190/geo2011-0237.1 | 2012 Kerry Key |

Changes: none to the numbers (full double precision, checked to be bit-identical to the source); the tables were only
converted to plain text with a header naming source and licence.

Where we took them from: the Python library **libdlf** (BSD-3-Clause, <https://github.com/emsig/libdlf>, version
0.3.0, maintained by the emsig community), as distributed with **empymod** 2.6.0 (Apache-2.0,
<https://github.com/emsig/empymod>). No code of empymod or libdlf is copied; the package does not import empymod.

## Software used

- numpy, scipy (BSD-3-Clause): required at run time.
- empymod (Apache-2.0): used only by the comparison tests in `tests/verification/empymod/` and the flat-model comparison in `tests/validation/1D/empymod` as an independent reference; optional
  (`tests/verification/empymod/requirements.txt`, `tests/validation/requirements.txt`).
- elfe3D (Apache-2.0): external finite-element code used for the validation runs in `tests/validation/3D/elfe3d_topo` and `tests/validation/3D/elfe3d_sphere`;
  not distributed here.

## Published methods

The analytic building blocks (Mie/T-matrix coupling, plane- and spherical-wave identities, Talbot inversion,
Hankel/Fourier filter method) are cited in the accompanying paper (repository `terrasphere_csem_paper`).
