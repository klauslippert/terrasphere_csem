# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klaus Lippert
"""Closed-form surface fields of horizontal electric dipoles (HED) on a homogeneous half-space, quasi-static, source
and receiver at the surface (time convention e^{-iwt}).

Dipole p along x at the origin, receiver at (x, y), r = |(x, y)|, gamma = sqrt(-i w mu sigma), Re gamma > 0
(e^{-iwt}; Ward & Hohmann 1988 write the same formulas with e^{+iwt} and gamma = sqrt(i w mu sigma)):
  E_x = p/(2 pi sigma r^3) [3 x^2/r^2 - 2 + (1 + gamma r) e^{-gamma r}]          (Ward & Hohmann 1988, ch. 4)
  E_y = p/(2 pi sigma r^3) 3 x y / r^2                                           (frequency independent)
  H_z = p y/(2 pi gamma^2 r^5) [3 - (3 + 3 gamma r + gamma^2 r^2) e^{-gamma r}]   (-> p y/(4 pi r^3) at DC)
Horizontal H from the air-side potential: H_h = -grad Phi, Phi = p d_y G, G~ = -1/(kappa (u + kappa)),
u = sqrt(kappa^2 + gamma^2), so with G' = dG/dr, G'' = d^2G/dr^2:
  H_x = -p x y/r^2 (G'' - G'/r),     H_y = -p [ y^2/r^2 G'' + x^2/r^3 G' ],
  G'  = 1/(4 pi r) + (1/2pi) int_0^inf (kappa-u)/(2(u+kappa)) J1(kappa r) dkappa,
  G'' = -G'/r + (1/2pi) int_0^inf kappa (kappa-u)/(2(u+kappa)) J0(kappa r) dkappa,
both kernels smooth and decaying (digital linear filter on a radial table, cubic spline in log r).
Frame: x, y, z UP (components); time e^{-iwt}.  Checked against empymod and against onedim.primary_field (tests).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from . import filters as _flt
from . import MU0
from .sources import hed_list

_HT = _flt.key_201_2009()


def gamma0(f, sigma):
    """gamma = sqrt(-i w mu0 sigma), Re gamma >= 0 (e^{-iwt})."""
    return np.sqrt(-1j * 2 * np.pi * f * MU0 * sigma)


def _gtab(r, gam):
    """G'(r), G''(r) for one frequency on radii r (1-D)."""
    lam = _HT.base[None, :] / r[:, None]
    u = np.sqrt(lam ** 2 + gam ** 2)
    k1 = (lam - u) / (2 * (u + lam))
    i1 = (k1 * _HT.j1[None, :]).sum(1) / r
    i0 = (lam * k1 * _HT.j0[None, :]).sum(1) / r
    g1 = 1 / (4 * np.pi * r) + i1 / (2 * np.pi)
    g2 = -g1 / r + i0 / (2 * np.pi)
    return g1, g2


def hed_fields(x, y, f, sigma, p=1.0, gtab=None):
    """Surface fields (Ex, Ey, Hx, Hy, Hz) of an x-directed HED of moment p at the origin; x, y arrays.
    gtab: optional callable r -> (G', G'') (spline table) for speed."""
    gam = gamma0(f, sigma)
    r = np.hypot(x, y); c, s = x / r, y / r
    e = np.exp(-gam * r)
    Ex = p / (2 * np.pi * sigma * r ** 3) * (3 * c * c - 2 + (1 + gam * r) * e)
    Ey = p / (2 * np.pi * sigma * r ** 3) * 3 * s * c
    if f == 0:
        Hz = p * y / (4 * np.pi * r ** 3)
    else:
        Hz = p * y / (2 * np.pi * gam ** 2 * r ** 5) * (3 - (3 + 3 * gam * r + gam ** 2 * r ** 2) * e)
    g1, g2 = gtab(r) if gtab is not None else _gtab(np.atleast_1d(r).ravel(), gam)
    g1 = np.reshape(g1, np.shape(r)); g2 = np.reshape(g2, np.shape(r))
    Hx = -p * c * s * (g2 - g1 / r)
    Hy = -p * (s * s * g2 + c * c * g1 / r)
    return Ex, Ey, Hx, Hy, Hz


def surface_fields(x, y, freqs, sigma, sources, ntab=600):
    """Surface fields of HED-type sources (Dipole 'HED' on the surface, grounded wires, polygon loops) at points
    (x, y).  Returns dict comp -> (nf, npts), comps Ex, Ey, Hx, Hy, Hz (frame x, y, z up; e^{-iwt}).
    Points must not coincide with a dipole."""
    x = np.asarray(x, float); y = np.asarray(y, float); freqs = np.atleast_1d(freqs)
    dips = hed_list(sources)
    out = {c: np.zeros((len(freqs), len(x)), complex) for c in ['Ex', 'Ey', 'Hx', 'Hy', 'Hz']}
    rmin, rmax = np.inf, 0.0
    for d in dips:
        r = np.hypot(x - d.pos[0], y - d.pos[1]); rmin = min(rmin, r.min()); rmax = max(rmax, r.max())
    rtab = np.geomspace(rmin * 0.99, rmax * 1.01, ntab)
    for i, f in enumerate(freqs):
        g1, g2 = _gtab(rtab, gamma0(f, sigma))
        sp = [CubicSpline(np.log(rtab), g1.real), CubicSpline(np.log(rtab), g1.imag),
              CubicSpline(np.log(rtab), g2.real), CubicSpline(np.log(rtab), g2.imag)]
        gt = lambda r: (sp[0](np.log(r)) + 1j * sp[1](np.log(r)), sp[2](np.log(r)) + 1j * sp[3](np.log(r)))
        for d in dips:
            ux, uy = np.cos(d.psi), np.sin(d.psi)
            X, Y = x - d.pos[0], y - d.pos[1]
            xl, yl = ux * X + uy * Y, -uy * X + ux * Y                # dipole frame (dipole along local x')
            ex, ey, hx, hy, hz = hed_fields(xl, yl, f, sigma, d.moment, gt)
            out['Ex'][i] += ux * ex - uy * ey; out['Ey'][i] += uy * ex + ux * ey
            out['Hx'][i] += ux * hx - uy * hy; out['Hy'][i] += uy * hx + ux * hy
            out['Hz'][i] += hz
    return out


def potential_dc(x, y, sigma, sources):
    """Direct-current surface potential of HED-type sources: V = sum p d.(x - s)/(2 pi sigma |x - s|^3)
    (limit of the electrode pair I/(2 pi sigma) (1/|x-B| - 1/|x-A|), current leaving into the ground at B)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    V = np.zeros(np.broadcast(x, y).shape)
    for d in hed_list(sources):
        X, Y = x - d.pos[0], y - d.pos[1]; r = np.hypot(X, Y)
        V += d.moment * (np.cos(d.psi) * X + np.sin(d.psi) * Y) / (2 * np.pi * sigma * r ** 3)
    return V


# ---------------------------------------------------------------------------------------------------------------
# Layered earth below the surface: closed-form half-space of the top layer + reflections from below.
# The layered surface field of an HED is computed as  F = F_halfspace(sigma_1) [closed form]  +  (F_layered - F_hs)
# [digital Hankel filter, same abscissae for both], where the difference contains only the waves reflected at the
# interfaces and decays like exp(-2 u d_1): the non-decaying source terms, which the filter integrates poorly when
# source and receiver lie in the same plane, cancel exactly.  For an x-directed unit HED at the origin the
# surface fields have the form (by symmetry; r, phi polar coordinates of the receiver)
#   E_x = a0 + a2 cos 2phi,  E_y = a2 sin 2phi,  H_x = c2 sin 2phi,  H_y = c0 - c2 cos 2phi,  H_z = d sin phi,
# so the correction is tabulated from receivers on the x- and y-axes and splined in log r.

def _modes_axes(stack, lam, w0, w1, rho, w):
    """E, H of a unit x-directed HED at the origin (zeta = 0) at surface receivers on the x-axis (phi = 0) and
    y-axis (phi = pi/2), radii rho (nr, 1), filter abscissae lam (nr, nf).  Returns dict of (nr,) arrays."""
    from .sources import Dipole, dipole_modes
    from .fields import bessel_weights
    dip = Dipole('HED', (0.0, 0.0, 0.0), moment=1.0, psi=0.0)
    k2s = 1j * w * MU0 * stack.sigma[1]; us = stack.u(1, lam, w)
    out = {k: np.zeros(rho.shape[0], complex) for k in ('Ex0', 'Ex90', 'Hy0', 'Hy90', 'Hz90')}
    for q in ('TE', 'TM'):
        Rup, Rdn = stack.reflections(q, lam, w)
        for m in (-1, 0, 1):
            A = dipole_modes(dip, (0.0, 0.0), lam, m, k2s, us)
            Aup, Adn = (A[0], A[1]) if q == 'TE' else (A[2], A[3])
            if not np.any(Aup) and not np.any(Adn):
                continue
            Pu, Pd = stack.propagate(q, lam, w, 0.0, Aup, Adn, 0.0, Rup, Rdn)
            wJ, wdJ, wmJ = bessel_weights(m, lam, rho, w01=(w0, w1))
            P = Pu + Pd; dP = us * (Pd - Pu); L2 = lam ** 2
            if q == 'TE':
                Er = np.sum((1j * w * MU0 / L2) * 1j * wmJ * P, -1); Ep = -np.sum((1j * w * MU0 / L2) * lam * wdJ * P, -1)
                Hr = np.sum(lam * wdJ * dP / L2, -1); Hp = np.sum(1j * wmJ * dP / L2, -1); Hz = np.sum(wJ * P, -1)
            else:
                s1 = stack.sigma[1]
                Hr = np.sum(1j * wmJ * P / L2, -1); Hp = -np.sum(lam * wdJ * P / L2, -1); Hz = 0.0
                Er = np.sum(lam * wdJ * dP / L2, -1) / s1; Ep = np.sum(1j * wmJ * dP / L2, -1) / s1
            for phi, tag in ((0.0, '0'), (np.pi / 2, '90')):
                e = np.exp(1j * m * phi); cp, sp = np.cos(phi), np.sin(phi)
                out['Ex' + tag] += (Er * cp - Ep * sp) * e
                out['Hy' + tag] += (Hr * sp + Hp * cp) * e
                if tag == '90':
                    out['Hz90'] += Hz * e
    return out


def correction_table(stack, f, rtab):
    """(a0, a2, c0, c2, d) of (layered - half-space(sigma_1)) for a unit x-HED, on radii rtab."""
    from .fields import dlf_grid
    from .layers import LayerStack
    w = 2 * np.pi * f
    hs = LayerStack([0.0], [stack.sigma[1]])
    lam = np.empty((len(rtab), 0)); w0 = w1 = None
    grids = [dlf_grid(r) for r in rtab]
    lam = np.array([g[0] for g in grids]); w0 = np.array([g[1] for g in grids]); w1 = np.array([g[2] for g in grids])
    rho = rtab[:, None]
    a = _modes_axes(stack, lam, w0, w1, rho, w); b = _modes_axes(hs, lam, w0, w1, rho, w)
    d = {k: a[k] - b[k] for k in a}
    return ((d['Ex0'] + d['Ex90']) / 2, (d['Ex0'] - d['Ex90']) / 2,
            (d['Hy0'] + d['Hy90']) / 2, (d['Hy90'] - d['Hy0']) / 2, d['Hz90'])


def surface_fields_layered(x, y, freqs, stack, sources, ntab=300):
    """Surface fields of HED-type sources on a layered earth (LayerStack `stack`, sources and receivers at zeta = 0):
    closed-form half-space of the top layer + tabulated reflection correction.  Same output as surface_fields()."""
    if stack.J == 1:
        return surface_fields(x, y, freqs, stack.sigma[1], sources)
    x = np.asarray(x, float); y = np.asarray(y, float); freqs = np.atleast_1d(freqs)
    out = surface_fields(x, y, freqs, stack.sigma[1], sources)
    dips = hed_list(sources)
    rmin, rmax = np.inf, 0.0
    for dd in dips:
        r = np.hypot(x - dd.pos[0], y - dd.pos[1]); rmin = min(rmin, r.min()); rmax = max(rmax, r.max())
    rtab = np.geomspace(rmin * 0.98, rmax * 1.02, ntab)
    lr = np.log(rtab)
    for i, f in enumerate(freqs):
        T = correction_table(stack, f, rtab)
        sp = [(CubicSpline(lr, t.real), CubicSpline(lr, t.imag)) for t in T]
        ev = lambda k, r: sp[k][0](np.log(r)) + 1j * sp[k][1](np.log(r))
        for dd in dips:
            ux, uy = np.cos(dd.psi), np.sin(dd.psi)
            X, Y = x - dd.pos[0], y - dd.pos[1]
            xl, yl = ux * X + uy * Y, -uy * X + ux * Y
            r = np.hypot(xl, yl); c2, s2 = (xl * xl - yl * yl) / r ** 2, 2 * xl * yl / r ** 2; s1 = yl / r
            a0, a2, c0, cc2, dz = (ev(k, r) for k in range(5))
            ex = dd.moment * (a0 + a2 * c2); ey = dd.moment * a2 * s2
            hx = dd.moment * cc2 * s2; hy = dd.moment * (c0 - cc2 * c2); hz = dd.moment * dz * s1
            out['Ex'][i] += ux * ex - uy * ey; out['Ey'][i] += uy * ex + ux * ey
            out['Hx'][i] += ux * hx - uy * hy; out['Hy'][i] += uy * hx + ux * hy
            out['Hz'][i] += hz
    return out


def potential_dc_layered(x, y, stack, sources):
    """Direct-current surface potential of HED-type sources on a layered earth: V = p cos(theta) (-U'(r)),
    -U'(r) = (1/(2 pi sigma_1)) int kappa J1(kappa r) (1 + rho)/(1 - rho) dkappa,  rho = -R_TM,dn(kappa, 0) e^{-2 kappa d_1}
    (potential reflection, = (sigma_1 - sigma_2)/(sigma_1 + sigma_2) e^{-2 kappa d_1} for two layers); the 1/r^2 part
    in closed form, the decaying rest with the digital filter."""
    from .fields import dlf_grid
    x = np.asarray(x, float); y = np.asarray(y, float)
    s1 = stack.sigma[1]
    V = potential_dc(x, y, s1, sources)
    if stack.J == 1:
        return V
    d1 = stack.depths[1]
    def rest(r):
        """Radial kernel of the layered DC potential without its closed-form half-space part, by the digital filter:
        (1/(2 pi sigma_1)) int kappa J_1(kappa r) 2 rho/(1 - rho) dkappa at the radius r."""
        lam, w0, w1 = dlf_grid(r)
        R = stack.reflections('TM', lam, 0.0)[1][1]
        rho = -R * np.exp(-2 * lam * d1)
        return np.real(np.sum(lam * 2 * rho / (1 - rho) * w1)) / (2 * np.pi * s1)
    for dd in hed_list(sources):
        X, Y = x - dd.pos[0], y - dd.pos[1]; r = np.hypot(X, Y)
        rt = np.geomspace(r.min() * 0.98, r.max() * 1.02, 200)
        sp = CubicSpline(np.log(rt), [rest(v) for v in rt])
        V += dd.moment * (np.cos(dd.psi) * X + np.sin(dd.psi) * Y) / r * sp(np.log(r))
    return V
