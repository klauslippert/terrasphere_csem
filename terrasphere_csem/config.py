# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klaus Lippert
"""Control file of a computation.

Every computation lives in its own folder with the control file `control.toml`; the series results are written to
`<folder>/results/`, the elfe3D input to `<folder>/elfe3d/`.  `Config` reads the control file and builds the
sources, the receiver positions, the grids and the sphere coupling from it.

Control file (TOML, SI units; z up, positions on the surface (x, y); depth positive down):

    name = "inline_three"
    [terrain]                      # Gaussian hill/depression h = H exp(-r^2/2s^2), centre (0, 0)
    width = 150.0                  # s (m)
    heights = [0.0, 30.0, -30.0]   # H (m): 0 flat, > 0 hill, < 0 depression; evaluated from the same coefficients
    [layers]                       # the top layer carries the terrain; all interfaces lie below the depression
    depths = [0.0, 100.0, 350.0]   # depths of the layer tops (first = 0)
    conductivity = [0.01, 0.005, 0.03]   # S/m
    [sphere]                       # optional; omit the block for no sphere
    radius = 60.0; conductivity = 1.0; mu_r = 1.0; center = [0.0, 0.0, 220.0]; degree = 10
    [[source]]                     # one or more blocks, summed
    type = "wire"                  # wire | dipole | rectangle | polygon | circle
    start = [-1750.0, 0.0]; end = [-1250.0, 0.0]; current = 1.0; gauss = 96
    # dipole: center, direction (deg from x towards y), length, current | rectangle: corner1, corner2, current |
    # polygon: corners [[x, y], ...], current | circle: center, radius, current, segments
    [receivers]
    profiles = ["x", "y"]          # profiles through the summit along the x-axis (y = 0) and the y-axis (x = 0)
    half_length = 600.0; spacing = 20.0
    [frequencies]                  # either fixed frequencies ...
    list = [0.1, 1.0, 1000.0, 100000.0]
    # [time]                       # ... or a time window (frequencies follow from the filter, lagged grid)
    # tmin = 1e-5; tmax = 10.0; signal = "off"      # off = switch-off step, on = switch-on step
    [output]
    formats = ["npz", "csv"]       # npz and/or csv (each file holds all components); add "coefficients" for
                                   # results/coefficients.npz (series coefficients, see __main__.py)
    [numerics]                     # defaults in NUMERICS
    method = "hankel"              # hankel (default, no periodic box) | fft (2-D FFT, box correction)
    m_max = 40; n_phi = 128; npan_r = 32; npan_l = 32; lam_max = 0.25     # hankel
    n = 512; dx = 10.0; orders = 20; lowpass = 0.2; window_radius = 860.0; window_width = 40.0     # fft
    box_correction = true; box_fmax = 100.0
"""
import os
import tomllib
import numpy as np

from .sources import grounded_wire, grounded_dipole, polygon_wire, circular_loop
from .layers import LayerStack

NUMERICS = dict(method='hankel', m_max=40, n_phi=128, npan_r=32, npan_l=32, lam_max=0.25,
                n=512, dx=10.0, orders=20, lowpass=0.2, window_radius=860.0, window_width=40.0, fmax_series=2e6,
                tolerance=1e-3,
                box_correction=True, box_fmax=100.0, box_dx=20.0, box_n=(256, 512, 1024), box_lowpass=0.12,
                box_orders=12)


class Config:
    """Parsed control file of one computation folder (terrain, layers, optional sphere, sources, receivers,
    frequencies or time window, output and numerical settings; see the module docstring).

    The objects needed by the series computation (sources, grids, sphere coupling) are built on demand."""
    def __init__(self, path):
        """Read and check the control file `path`, or `<path>/control.toml` if `path` is a folder.  `folder` is the
        folder of the control file (the outputs go to `<folder>/results/`), `file` the control file itself.

        Raises ValueError unless exactly one of the blocks [frequencies] and [time] is present."""
        path = os.path.abspath(path)
        self.file = path if os.path.isfile(path) else os.path.join(path, 'control.toml')
        self.folder = os.path.dirname(self.file)
        with open(self.file, 'rb') as fh:
            self.d = tomllib.load(fh)
        d = self.d
        self.name = d.get('name', os.path.basename(self.folder))
        g = d['terrain']
        self.s = float(g['width']); self.heights = [float(v) for v in g.get('heights', [0.0])]
        sch = d['layers']
        self.layers = LayerStack(sch['depths'], sch['conductivity'])
        if self.layers.J > 1:
            assert self.layers.depths[1] > -min(0.0, min(self.heights)), \
                "all layer interfaces must lie below the bottom of the depression"
        self.sphere = d.get('sphere')
        if ('frequencies' in d) == ('time' in d):
            raise ValueError("give either [frequencies] or [time]")
        self.domain = 'frequency' if 'frequencies' in d else 'time'
        if self.domain == 'frequency':
            self.freqs = np.array([float(v) for v in d['frequencies']['list']])
        else:
            from .timedomain import time_axis, filter_frequencies
            z = d['time']
            self.t = time_axis(float(z['tmin']), float(z['tmax']))
            self.signal = z.get('signal', 'off'); assert self.signal in ('off', 'on')
            self.freqs = filter_frequencies(self.t)                   # all filter abscissae (the series up to fmax)
        a = d.get('output', {})
        self.formats = [f.lower() for f in a.get('formats', ['npz'])]
        assert set(self.formats) <= {'npz', 'csv', 'coefficients'} and set(self.formats) & {'npz', 'csv'}, \
            "output.formats: npz and/or csv, optionally coefficients"
        e = d['receivers']
        self.profiles = list(e.get('profiles', ['x', 'y']))
        self.half_length = float(e.get('half_length', 600.0)); self.spacing = float(e.get('spacing', 20.0))
        self.numerics = dict(NUMERICS); self.numerics.update(d.get('numerics', {}))
        self.source_specs = d['source']

    # ------------------------------------------------------------------ objects
    def sources(self):
        """List of HED Dipole objects built from the [[source]] blocks (wires, grounded dipoles, rectangular, polygonal
        and circular loops, all as HED sums on the surface)."""
        out = []
        for q in self.source_specs:
            t = q['type']; I = float(q.get('current', 1.0))
            if t == 'wire':
                out += grounded_wire(q['start'], q['end'], I, int(q.get('gauss', 96)))
            elif t == 'dipole':
                out += grounded_dipole(q['center'], np.radians(float(q.get('direction', 0.0))), float(q['length']), I,
                                       n_gauss=q.get('gauss'))
            elif t == 'rectangle':
                (x1, y1), (x2, y2) = q['corner1'], q['corner2']
                out += polygon_wire([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], I, int(q.get('gauss', 24)),
                                    closed=True)
            elif t == 'polygon':
                out += polygon_wire([tuple(p) for p in q['corners']], I, int(q.get('gauss', 24)), closed=True)
            elif t == 'circle':
                out += circular_loop(q['center'], float(q['radius']), I, int(q.get('segments', 256)),
                                     int(q.get('gauss', 3)))
            else:
                raise ValueError(f"unknown source type {t}")
        return out

    def receivers(self):
        """Receiver positions (px, py) and the profile index of each: x-profile (y = 0) and/or y-profile (x = 0)."""
        a = np.arange(-self.half_length, self.half_length + 1e-6, self.spacing)
        px, py, prof = [], [], []
        for p in self.profiles:
            if p == 'x':
                px += list(a); py += [0.0] * len(a); prof += ['x'] * len(a)
            elif p == 'y':
                px += [0.0] * len(a); py += list(a); prof += ['y'] * len(a)
        return np.array(px), np.array(py), prof

    def hankel_grid(self):
        """HankelGrid for the receivers and the numerical settings of the control file."""
        from .terrain_hankel import HankelGrid
        px, py, _ = self.receivers()
        nu = self.numerics
        return HankelGrid(self.sources(), (px, py), s=self.s, M=int(nu['m_max']), n_phi=int(nu['n_phi']),
                          r_win=float(nu['window_radius']), w_win=float(nu['window_width']),
                          lam_max=float(nu['lam_max']), kc=float(nu['lowpass']), npan_r=int(nu['npan_r']),
                          npan_l=int(nu['npan_l']))

    def grid(self, n=None, dx=None):
        """FFT Grid for the control file; n and dx (grid points per side, spacing in m) override the [numerics] values,
        e.g. for the coarse runs of the box correction."""
        from .terrain import Grid
        return Grid(self.sources(), n=n or int(self.numerics['n']), dx=dx or float(self.numerics['dx']), s=self.s,
                    r_win=float(self.numerics['window_radius']), w_win=float(self.numerics['window_width']))

    def sphere_coupling(self, g):
        """Coupling of the sphere to the terrain series for the grid g (HankelGrid: SphereTerrainHankel, Grid:
        SphereTerrain), or None if the control file has no [sphere] block."""
        if not self.sphere:
            return None
        from .sphere import SphereModel, Sphere
        from .sphere_terrain import SphereTerrain, SphereTerrainHankel
        from .terrain_hankel import HankelGrid
        k = self.sphere
        km = SphereModel(self.layers, Sphere(float(k['radius']), float(k['conductivity']), float(k.get('mu_r', 1.0))),
                         tuple(k['center']), nmax=int(k.get('degree', 10)))
        return SphereTerrainHankel(km, g) if isinstance(g, HankelGrid) else SphereTerrain(km, g)

    def height(self, x, y, H):
        """Terrain height H exp(-(x^2 + y^2)/(2 s^2)) [m] (z up) of the Gaussian hill (H < 0: depression) at (x, y)."""
        return H * np.exp(-(np.asarray(x) ** 2 + np.asarray(y) ** 2) / (2 * self.s ** 2))
