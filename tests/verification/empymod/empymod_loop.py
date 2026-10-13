#!/usr/bin/env python3
"""Circular loop in empymod as a 1-D reference.

The loop is a polygon of N straight segments (empymod.bipole, `srcpts` Gauss points per segment) whose circumradius
is scaled so that the polygon area equals pi a^2; the midpoint of one chord lies on the +x axis.
Coordinates: empymod z points DOWN (z_emp = depth), ours up. Receivers lie on the +x axis with azimuth 90 degrees
(= +y = phi direction), i.e. they measure E_phi.
Time convention: empymod uses e^{+iwt}, we use e^{-iwt}; the frequency-domain values are therefore returned
COMPLEX CONJUGATED (checked: full-space VMD, ours/conj(empymod) real to 1e-7)."""
import numpy as np
import empymod
# empymod's default Hankel filter (key_201_2012) gives 2-3 % error for the fine loop polygon at receivers in the
# near field; key_401_2009 with exact frequencies (pts_per_dec = -1) brings this down to 1e-4.
STD = dict(htarg={'dlf': 'key_401_2009'}, ftarg={'pts_per_dec': -1})


def polygon(a, N=256, z=0.0, equal_area=True):
    """Polygon source of N segments for empymod.bipole (list [x0, x1, y0, y1, z0, z1]); radius a [m], depth z [m].

    equal_area=True scales the circumradius so that the polygon has the area pi a^2."""
    r = a * np.sqrt(np.pi / ((N / 2) * np.sin(2 * np.pi / N))) if equal_area else a
    ph = (np.arange(N + 1) + 0.5) * 2 * np.pi / N
    x, y = r * np.cos(ph), r * np.sin(ph)
    return [x[:-1], x[1:], y[:-1], y[1:], np.full(N, z), np.full(N, z)]


def E_phi_freq(a, rho, z_emp, f, depth, res, N=256, srcpts=5, **kw):
    """E_phi [V/m] for 1 A at the points (rho_i, 0, z_emp_i) and frequencies f [Hz]; shape (nf, npts); conjugated."""
    rho, z_emp, f = (np.atleast_1d(v).astype(float) for v in (rho, z_emp, f))
    rec = [rho, np.zeros_like(rho), z_emp, 90.0, 0.0]
    out = empymod.bipole(src=polygon(a, N), rec=rec, depth=depth, res=res, freqtime=f,
                         srcpts=srcpts, strength=1.0, verb=0, **{'htarg': STD['htarg'], **kw})
    return np.conj(np.asarray(out).reshape(len(f), len(rho), N).sum(-1))


def V_off_time(a, t, depth, res, a_rx=None, z_rec=0.02, N=256, srcpts=5, **kw):
    """Switch-off voltage [V] of the coaxial receiver loop (radius a_rx, default a) for 1 A:
    V = 2 pi a_rx E_phi(rho = a_rx, z_rec).  CAUTION: for a_rx = a the receiver sits on the wire; the logarithmically
    singular self-inductance (purely imaginary) spoils empymod's time transform (dlf grossly wrong from ~3 ms,
    fftlog completely). Choose a_rx clearly different from a for time-domain comparisons."""
    t = np.atleast_1d(t).astype(float)
    a_rx = a if a_rx is None else a_rx
    rec = [a_rx, 0.0, z_rec, 90.0, 0.0]
    out = empymod.bipole(src=polygon(a, N), rec=rec, depth=depth, res=res, freqtime=t, signal=-1,
                         srcpts=srcpts, strength=1.0, verb=0, **{**STD, **kw})
    return 2 * np.pi * a_rx * np.asarray(out).reshape(len(t), N).sum(-1)


def V_freq(a, f, depth, res, a_rx=None, z_rec=0.02, N=256, srcpts=5, **kw):
    """V(w) [V/A] of the coaxial receiver loop, complex conjugated. Re V is finite and usable even for a_rx = a."""
    a_rx = a if a_rx is None else a_rx
    return 2 * np.pi * a_rx * E_phi_freq(a, a_rx, z_rec, f, depth, res, N=N, srcpts=srcpts, **kw)[:, 0]


def E_vmd(x, z_emp, z_src, f, depth, res):
    """E_phi (= E_y on the x axis) of a vertical magnetic dipole at (0, 0, z_src), empymod normalisation (strength 0);
    receivers at different depths -> bipole with msrc; complex conjugated."""
    x, z_emp = np.atleast_1d(x).astype(float), np.atleast_1d(z_emp).astype(float)
    out = empymod.bipole(src=[0.0, 0.0, z_src, 0.0, 90.0], rec=[x, np.zeros_like(x), z_emp, 90.0, 0.0], depth=depth,
                         res=res, freqtime=f, msrc=True, strength=0.0, verb=0, htarg=STD['htarg'])
    return np.conj(np.asarray(out).reshape(-1))
