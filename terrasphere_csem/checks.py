# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klaus Lippert
"""Sanity checks of a control file before a computation.

`check(path)` reads the control file (or `<path>/control.toml`) and returns a list of `Issue`s, each an error
(the model is outside the validity of the method or the file is wrong) or a warning (the model is valid but
close to a limit; the result may be less accurate than the series suggests).  The main program prints them all
before it starts and stops on errors unless it is called with `-f`.  The thresholds come from the paper:

  terrain slope           H/s > 0.6 (20 deg): warning; H/s > 1.65 (beyond the Laplace convergence radius): error
  sphere and terrain      |H| > 0.5 x depth of the sphere top below z = 0: warning (series converges poorly)
  sphere degree           q^(2N) > 1e-6 with q = max(b/c_source, b/(2 h_up), b/(2 h_down)): warning (estimate of
                          the truncation of the sphere part; c_source the distance to the nearest source point,
                          h the distance from the centre to the interfaces resp. to z = 0)
  terrain frequency       f > 1/(pi mu0 sigma_1 H^2) (skin depth not large against H): warning
  window                  with terrain (a height != 0) or a sphere (its coupling uses the same grid, also on flat
                          ground): fft: sources where the window exceeds 1e-15, receivers where it differs from 1 by
                          more than 1e-8; hankel: sources inside r_max + 50 m (radial grid to r_max = R + 8 w),
                          receivers beyond R - 4.5 w.  With the defaults: sources at least 1084 m (fft) / 1230 m
                          (hankel) from the summit, receivers within 698 m / 680 m: error
  interfaces near sphere  gap below 10 m: warning
"""
import os
import tomllib
from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

MU0 = 4e-7 * np.pi

KEYS = {
    None: {'name', 'terrain', 'layers', 'sphere', 'source', 'receivers', 'frequencies', 'time', 'output', 'numerics'},
    'terrain': {'width', 'heights'},
    'layers': {'depths', 'conductivity'},
    'sphere': {'radius', 'conductivity', 'mu_r', 'center', 'degree'},
    'receivers': {'profiles', 'half_length', 'spacing', 'points'},
    'frequencies': {'list'},
    'time': {'tmin', 'tmax', 'signal'},
    'output': {'formats'},
}
SOURCE_KEYS = {
    'wire': {'start', 'end', 'gauss'},
    'dipole': {'center', 'direction', 'length', 'gauss'},
    'rectangle': {'corner1', 'corner2', 'gauss'},
    'polygon': {'corners', 'gauss'},
    'circle': {'center', 'radius', 'segments', 'gauss'},
}
SOURCE_REQUIRED = {'wire': ('start', 'end'), 'dipole': ('center', 'length'), 'rectangle': ('corner1', 'corner2'),
                   'polygon': ('corners',), 'circle': ('center', 'radius')}


@dataclass
class Issue:
    """One finding of the checks: level 'error' or 'warning', and the message."""
    level: str
    message: str

    def __str__(self):
        """'ERROR: ...' or 'warning: ...'."""
        return ('ERROR: ' if self.level == 'error' else 'warning: ') + self.message


def _num(v):
    """True if v is a real number (not a bool)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check(path):
    """Issues of the control file `path` (file, or folder containing control.toml)."""
    from .config import NUMERICS
    path = os.path.abspath(path)
    file = path if os.path.isfile(path) else os.path.join(path, 'control.toml')
    try:
        with open(file, 'rb') as fh:
            d = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        return [Issue('error', f"cannot read {file}: {e}")]
    out = []
    E = lambda m: out.append(Issue('error', m))
    W = lambda m: out.append(Issue('warning', m))

    # 1. unknown keys (typing errors)
    for blk, allowed in KEYS.items():
        sec = d if blk is None else d.get(blk, {})
        if isinstance(sec, dict):
            for k in set(sec) - allowed:
                E(f"unknown key '{k}'" + (f" in [{blk}]" if blk else " at the top level"))
    for k in set(d.get('numerics', {})) - set(NUMERICS):
        E(f"unknown key '{k}' in [numerics]")
    srcs = d.get('source', [])
    if isinstance(srcs, dict):
        srcs = [srcs]
    for i, q in enumerate(srcs):
        t = q.get('type')
        if t not in SOURCE_KEYS:
            E(f"source {i + 1}: unknown type {t!r} (wire, dipole, rectangle, polygon, circle)")
            continue
        for k in set(q) - SOURCE_KEYS[t] - {'type', 'current'}:
            E(f"source {i + 1} ({t}): unknown key '{k}'")
        for k in SOURCE_REQUIRED[t]:
            if k not in q:
                E(f"source {i + 1} ({t}): '{k}' missing")

    # 2. required entries and ranges
    for blk, key in (('terrain', 'width'), ('layers', 'depths'), ('layers', 'conductivity')):
        if key not in d.get(blk, {}):
            E(f"[{blk}] {key} missing")
    if not srcs:
        E("no [[source]] block")
    if ('frequencies' in d) == ('time' in d):
        E("give exactly one of [frequencies] and [time]")
    if any(e.level == 'error' for e in out):
        return out                                              # the rest needs a readable model
    g, la = d['terrain'], d['layers']
    s = g['width']; heights = g.get('heights', [0.0])
    if not (_num(s) and s > 0):
        E("[terrain] width must be > 0")
    if not all(_num(h) for h in heights):
        E("[terrain] heights must be numbers")
    dep, con = la['depths'], la['conductivity']
    if not (all(_num(v) for v in dep) and all(_num(v) and v > 0 for v in con)):
        E("[layers]: depths must be numbers and conductivities > 0")
    sp = d.get('sphere')
    if sp is not None:
        for k in ('radius', 'conductivity', 'center'):
            if k not in sp:
                E(f"[sphere] {k} missing")
        if 'radius' in sp and not (_num(sp['radius']) and sp['radius'] > 0):
            E("[sphere] radius must be > 0")
        if 'conductivity' in sp and not (_num(sp['conductivity']) and sp['conductivity'] > 0):
            E("[sphere] conductivity must be > 0")
        if not (_num(sp.get('mu_r', 1.0)) and sp.get('mu_r', 1.0) > 0):
            E("[sphere] mu_r must be > 0")
        if not (isinstance(sp.get('degree', 10), int) and sp.get('degree', 10) >= 1):
            E("[sphere] degree must be an integer >= 1")
        if 'center' in sp and not (len(sp['center']) == 3 and all(_num(v) for v in sp['center'])):
            E("[sphere] center must be [x, y, depth]")
    if 'frequencies' in d:
        fl = d['frequencies'].get('list', [])
        if not fl or not all(_num(f) and f > 0 for f in fl):
            E("[frequencies] list must contain frequencies > 0")
    else:
        tm = d['time']
        if not (_num(tm.get('tmin')) and _num(tm.get('tmax')) and 0 < tm['tmin'] < tm['tmax']):
            E("[time]: 0 < tmin < tmax required")
        if tm.get('signal', 'off') not in ('off', 'on'):
            E("[time] signal must be 'off' or 'on'")
    rc = d.get('receivers', {})
    pts = rc.get('points', [])
    if not (isinstance(pts, list) and all(isinstance(p, list) and len(p) == 2 and all(_num(v) for v in p)
                                          for p in pts)):
        E("[receivers] points must be a list of [x, y] pairs")
        pts = []
    profiles = rc.get('profiles', [] if pts else ['x', 'y'])
    if not (isinstance(profiles, list) and set(profiles) <= {'x', 'y'}):
        E("[receivers] profiles: 'x' and/or 'y'")
    elif not profiles and not pts:
        E("[receivers]: no receivers (give profiles and/or points)")
    if not (_num(rc.get('half_length', 600.0)) and rc.get('half_length', 600.0) > 0
            and _num(rc.get('spacing', 20.0)) and rc.get('spacing', 20.0) > 0):
        E("[receivers] half_length and spacing must be > 0")
    fm = [f.lower() for f in d.get('output', {}).get('formats', ['npz'])]
    if not (set(fm) <= {'npz', 'csv', 'coefficients'} and set(fm) & {'npz', 'csv'}):
        E("[output] formats: npz and/or csv, optionally coefficients")
    nu = dict(NUMERICS); nu.update(d.get('numerics', {}))
    if nu['method'] not in ('hankel', 'fft'):
        E("[numerics] method must be 'hankel' or 'fft'")
    for k in ('orders', 'm_max', 'n_phi', 'npan_r', 'npan_l', 'n', 'box_orders'):
        if not (isinstance(nu[k], int) and nu[k] >= 1):
            E(f"[numerics] {k} must be an integer >= 1")
    for k in ('lowpass', 'lam_max', 'window_radius', 'window_width', 'dx', 'box_dx', 'box_lowpass', 'box_fmax',
              'fmax_series', 'tolerance'):
        if not (_num(nu[k]) and nu[k] > 0):
            E(f"[numerics] {k} must be > 0")
    if any(e.level == 'error' for e in out):
        return out

    # 3. layers
    dep = np.asarray(dep, float)
    if dep[0] != 0.0 or np.any(np.diff(dep) <= 0) or len(dep) != len(con):
        E("[layers]: depths must start at 0 and increase, one conductivity per layer")
        return out
    hmin = min(0.0, min(heights))
    if len(dep) > 1 and not dep[1] > -hmin:
        E(f"layer interface at {dep[1]:g} m lies above the bottom of the depression ({-hmin:g} m): the terrain must "
          "lie in the top layer")

    # sources (Gauss points) and receivers from the parsed configuration
    from .config import Config
    try:
        st = Config(file)
        q = st.sources()
        px, py, _ = st.receivers()
    except Exception as e:                                       # reported, not raised
        E(f"the model cannot be built: {type(e).__name__}: {e}")
        return out
    sxy = np.array([[p.pos[0], p.pos[1]] for p in q])
    Hmax = max(abs(h) for h in heights)
    s = float(s); s1 = float(con[0])
    R, w = float(nu['window_radius']), float(nu['window_width'])

    # 4.-8. sphere
    if sp is not None:
        b = float(sp['radius']); xc, yc, dc = (float(v) for v in sp['center']); N = int(sp.get('degree', 10))
        if dc - b <= 0:
            E(f"sphere top at depth {dc - b:g} m: the sphere must lie entirely below the flat ground (z = 0)")
        else:
            l = int(np.searchsorted(dep, dc, side='right')) - 1
            top = dep[l]; bot = dep[l + 1] if l + 1 < len(dep) else np.inf
            gap_up = dc - b - top if l > 0 else np.inf; gap_dn = bot - (dc + b)
            if min(gap_up, gap_dn) <= 0:
                E(f"sphere (depth {dc - b:g} to {dc + b:g} m) crosses or touches a layer interface "
                  f"({top:g} / {bot:g} m): it must lie entirely inside one layer")
            elif min(gap_up, gap_dn) < 10.0:
                W(f"sphere comes within {min(gap_up, gap_dn):.1f} m of a layer interface: high degrees needed")
            # 5. depression cutting into the sphere, also beside the axis
            if hmin < 0:
                xs = np.linspace(xc - b - 1, xc + b + 1, 161); ys = np.linspace(yc - b - 1, yc + b + 1, 161)
                X, Y = np.meshgrid(xs, ys)
                Z = hmin * np.exp(-(X ** 2 + Y ** 2) / (2 * s ** 2))
                dist = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2 + (Z + dc) ** 2).min()
                if dist <= b:
                    E(f"the depression of {hmin:g} m cuts into the sphere")
            # 6. convergence of the terrain series near the sphere
            if Hmax > 0.5 * (dc - b):
                W(f"|H| = {Hmax:g} m exceeds half the depth of the sphere top ({dc - b:g} m): the terrain series "
                  "converges poorly near the sphere (paper: 30 m for a top at 80 m converges, 60 and 90 m do not)")
            # 7. degree of the sphere part (estimate)
            h_up = dc - top if l > 0 else dc
            h_dn = bot - dc
            c_src = np.min(np.sqrt((sxy[:, 0] - xc) ** 2 + (sxy[:, 1] - yc) ** 2 + dc ** 2))
            qq = max(b / c_src, b / (2 * h_up), b / (2 * h_dn) if np.isfinite(h_dn) else 0.0)
            if qq ** (2 * N) > 1e-6:
                Nreq = int(np.ceil(-6.0 / (2 * np.log10(qq)))) if qq < 1 else None
                W(f"sphere degree {N} may be too low (estimated truncation {qq ** (2 * N):.0e}" +
                  (f"; about {Nreq} for 1e-6)" if Nreq else ")"))
        # 8. Hankel method: sphere on the hill axis
        if nu['method'] == 'hankel' and (abs(xc) > 1e-9 or abs(yc) > 1e-9):
            E("method 'hankel' needs the sphere on the hill axis (x = y = 0); use method = 'fft'")

    # 9. sources off the hill and outside the window
    if Hmax > 0:
        hs = Hmax * np.exp(-(sxy[:, 0] ** 2 + sxy[:, 1] ** 2) / (2 * s ** 2))
        if hs.max() > 1e-6:
            E(f"terrain height {hs.max():.1e} m at a source: sources must lie on the flat ground off the hill")
    # The source field is singular at the source: the window 0.5 erfc((r - R)/w) must cut it off completely before
    # the spectral derivatives (fft), and the radial grid of the Hankel route ends at r_max = R + 8 w, 50 m short of
    # the nearest source (terrain_hankel.HankelGrid).  The grid is built for the terrain series and for the sphere
    # coupling, so the limits also hold on flat ground when there is a sphere.  The limits are those of the code.
    rs = np.hypot(sxy[:, 0], sxy[:, 1]).min()
    rr = np.hypot(px, py).max()
    if Hmax > 0 or sp is not None:
        why = "the terrain" if Hmax > 0 else "the sphere coupling"
        if nu['method'] == 'hankel':
            rs_min, rr_max = R + 8 * w + 50.0, R - 4.5 * w
            src_bad, rec_bad = rs <= rs_min, rr >= rr_max
        else:
            rs_min, rr_max = R + 5.6 * w, R - 4.05 * w
            src_bad, rec_bad = 0.5 * erfc((rs - R) / w) > 1e-15, 0.5 * erfc((R - rr) / w) > 1e-8
        if src_bad:
            E(f"a source lies {rs:.0f} m from the summit: too close to the hill; with {why} sources must lie more "
              f"than {rs_min:.0f} m from the summit (method {nu['method']}: outside the window of the expansion, "
              f"window_radius {R:g}, window_width {w:g})")

        # 10. receivers inside the window, on the grid
        if rec_bad:
            E(f"receivers reach {rr:.0f} m from the summit, beyond the exact part of the window (with {why} and "
              f"method {nu['method']}: < {rr_max:.0f} m)")
    if nu['method'] == 'fft':
        # the series is evaluated on the FFT grid (dx) and, for the box correction, on the coarse grids (box_dx)
        box = (nu['box_correction'] and Hmax > 0 and st.freqs.min() <= float(nu['box_fmax']))
        for dx, key in [(float(nu['dx']), 'dx')] + ([(float(nu['box_dx']), 'box_dx')] if box else []):
            on = lambda v: abs(v / dx - round(v / dx)) <= 1e-9
            if st.profiles:
                for v, nm in ((st.spacing, 'spacing'), (st.half_length, 'half_length')):
                    if not on(v):
                        E(f"[receivers] {nm} = {v:g} m is not a multiple of {key} = {dx:g} m: receivers must lie on "
                          "the FFT grid")
            off = [(x, y) for x, y in st.points if not (on(x) and on(y))]
            if off:
                E(f"[receivers] {len(off)} point(s) not on the FFT grid ({key} = {dx:g} m), e.g. {off[0]}: with "
                  f"method fft x and y must be multiples of {key}")

    # 11. slope
    if Hmax / s > 1.65:
        E(f"H/s = {Hmax / s:.2f} > 1.65: beyond the convergence radius of the height series")
    elif Hmax / s > 0.6:
        W(f"H/s = {Hmax / s:.2f} > 0.6 (slope above 20 deg): the series leaves larger truncation errors")

    # 12. frequency limit of the terrain series (frequency domain)
    if Hmax > 0 and st.domain == 'frequency':
        flim = 1.0 / (np.pi * MU0 * s1 * Hmax ** 2)
        if st.freqs.max() > flim:
            W(f"frequencies above {flim:.3g} Hz: the skin depth is not large against H = {Hmax:g} m")

    # 13. FFT box and low-pass
    if nu['method'] == 'fft' and Hmax > 0:
        if nu['n'] * nu['dx'] / 2 < R + 6 * w:
            E(f"FFT box {nu['n'] * nu['dx']:g} m is too small for the window (needs > {2 * (R + 6 * w):g} m)")
        if nu['lowpass'] >= np.pi / nu['dx']:
            E(f"lowpass {nu['lowpass']:g} rad/m is not below the Nyquist wavenumber pi/dx = {np.pi / nu['dx']:.3g}")
        if nu['box_correction']:
            if nu['box_lowpass'] >= np.pi / nu['box_dx']:
                E(f"box_lowpass {nu['box_lowpass']:g} rad/m is not below pi/box_dx = {np.pi / nu['box_dx']:.3g}")
            if min(nu['box_n']) * nu['box_dx'] / 2 < R + 6 * w:
                E("the smallest box of the box correction is too small for the window")
        box = nu['n'] * nu['dx']
        delta = 503.3 / np.sqrt(s1 * st.freqs)
        unc = st.freqs[(delta > 0.1 * box) & ((st.freqs > nu['box_fmax']) | (not nu['box_correction']))]
        if len(unc):
            W(f"{len(unc)} frequencies (from {unc.min():.3g} Hz) have a skin depth above a tenth of the box and no box "
              "correction")

    # 14. time domain
    if st.domain == 'time':
        if st.freqs.max() > nu['fmax_series'] and Hmax > 0:
            W(f"the earliest times need frequencies up to {st.freqs.max():.3g} Hz; the terrain part above fmax_series = "
              f"{nu['fmax_series']:g} Hz and above the cut-off f_c (written to the results) is continued as a constant")

    # 15. output size
    if 'coefficients' in fm:
        W("formats contains 'coefficients': results/coefficients.npz is large (about 13 MB per case with fft, n = 512)")
    return out


def report(issues):
    """Text block of the issues (empty string if none)."""
    if not issues:
        return "checks of the control file: no issues"
    ne = sum(i.level == 'error' for i in issues)
    return (f"checks of the control file: {ne} error(s), {len(issues) - ne} warning(s)\n" +
            '\n'.join('  ' + str(i) for i in issues))
