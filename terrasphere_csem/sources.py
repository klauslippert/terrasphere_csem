# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klaus Lippert
"""Point-dipole sources and their azimuthal mode amplitudes about an arbitrary vertical axis.

Dipole fields in a homogeneous conductive medium (k, sigma), G = e^{ikR}/(4 pi R) = int (lam/(4 pi u)) J0(lam rho') e^{-u|z-zs|} dlam:
  VMD (moment mz, z up):      H = grad div(mz G z^) + k^2 mz G z^,  E = i w mu curl(mz G z^)
                              -> H_z = mz lam^2 G,  J_z = 0
  HED (moment p = I dl, horizontal direction angle psi):  E = (1/sigma)(grad div + k^2)(p G d^), H = curl(p G d^)
                              -> H_z = p (d_y d_x G - d_x d_y G),  J_z = p d_z (d^.grad) G
  HMD (moment mh, direction psi):  H = (grad div + k^2)(mh G d^), E = i w mu curl(mh G d^)
                              -> H_z = mh d_z (d^.grad) G,  J_z = k^2 mh (d_y d_x G - d_x d_y G)
Azimuthal decomposition about a centre c (Graf/Neumann addition theorems; (rho_s, phi_s) = source position
relative to c, (rho, phi) = field point relative to c, rho' = |rho - rho_s|):
  J0(lam rho')          = sum_m J_m(lam rho) e^{im phi} C0_m,        C0_m  = J_m(lam rho_s) e^{-im phi_s}
  J1(lam rho') cos phi' = sum_m J_m(lam rho) e^{im phi} Ccos_m,     Ccos_m = [J_{m-1}(lam rho_s) e^{-i(m-1)phi_s} - J_{m+1}(lam rho_s) e^{-i(m+1)phi_s}]/2
  J1(lam rho') sin phi' = sum_m J_m(lam rho) e^{im phi} Csin_m,     Csin_m = -i [J_{m-1}(..) e^{-i(m-1)phi_s} + J_{m+1}(..) e^{-i(m+1)phi_s}]/2
with d_x J0(lam rho') = -lam J1(lam rho') cos phi', d_y J0 = -lam J1 sin phi'.
A source returns, per family q (TE: H_z, TM: J_z) and order m, the amplitudes A+ (wave going UP, z > z_s) and
A- (going down) of the mode J_m(lam rho) e^{im phi} e^{-u|z - z_s|}  (to be integrated dlam)."""
import numpy as np
from scipy.special import jv
from . import MU0


class Dipole:
    """typ in {'HED', 'VMD', 'HMD', 'LOOP'}; pos = (x, y, zeta) in the global frame (zeta = depth, positive down);
    moment: A m (HED), A m^2 (magnetic), current A (LOOP, horizontal circular loop of `radius`, counter-clockwise
    seen from above, exact spectrum H_z = (I a/2) int (lam^2/u) J_1(lam a) J_0(lam rho') e^{-u|dz|} dlam, TE only);
    psi: horizontal direction angle (rad) for HED/HMD."""
    def __init__(self, kind, pos, moment=1.0, psi=0.0, radius=None):
        """kind: 'HED', 'VMD', 'HMD' or 'LOOP'; pos: (x, y, depth) [m]; moment: see the class docstring; psi: horizontal
        direction angle [rad]; radius: loop radius [m] (LOOP only)."""
        self.kind = kind; self.pos = np.asarray(pos, float); self.moment = float(moment); self.psi = float(psi)
        self.radius = radius

    def as_polygon(self, n_seg=256, n_gauss=3):
        """LOOP -> list of HED segments (for the 1-D primary field via the digital filter)."""
        if self.kind != 'LOOP':
            return [self]
        return circular_loop(self.pos[:2], self.radius, self.moment, n_seg, n_gauss, self.pos[2])

    def __repr__(self):
        """Short text form for debugging."""
        return f"Dipol({self.kind}, pos={self.pos}, moment={self.moment}, psi={self.psi})"


def graf_coeffs(lam, m, rho_s, phi_s):
    """(C0, Ccos, Csin)_m for arrays lam."""
    lam = np.asarray(lam, float)
    Jm = jv(m, lam * rho_s); Jm1 = jv(m - 1, lam * rho_s); Jp1 = jv(m + 1, lam * rho_s)
    e0 = np.exp(-1j * m * phi_s); em1 = np.exp(-1j * (m - 1) * phi_s); ep1 = np.exp(-1j * (m + 1) * phi_s)
    C0 = Jm * e0
    Ccos = 0.5 * (Jm1 * em1 - Jp1 * ep1)
    Csin = -0.5j * (Jm1 * em1 + Jp1 * ep1)
    return C0, Ccos, Csin


def graf_coeffs_d(lam, m, rho_s, phi_s, axis):
    """d/d(x_s) or d/d(y_s) of (C0, Ccos, Csin)_m (source position relative to the centre), from
    d_x[J_nu(l rho) e^{-i nu phi}] = (l/2)[J_{nu-1} e^{-i(nu-1)phi} - J_{nu+1} e^{-i(nu+1)phi}],
    d_y[J_nu(l rho) e^{-i nu phi}] = -(i l/2)[J_{nu-1} e^{-i(nu-1)phi} + J_{nu+1} e^{-i(nu+1)phi}]."""
    lam = np.asarray(lam, float)
    def d(nu):
        """Derivative of J_nu(lam rho_s) e^{-i nu phi_s} with respect to x_s or y_s (as selected by `axis`) for order nu."""
        a = jv(nu - 1, lam * rho_s) * np.exp(-1j * (nu - 1) * phi_s); b = jv(nu + 1, lam * rho_s) * np.exp(-1j * (nu + 1) * phi_s)
        return 0.5 * lam * (a - b) if axis == 'x' else -0.5j * lam * (a + b)
    dC0 = d(m)
    dCcos = 0.5 * (d(m - 1) - d(m + 1))
    dCsin = -0.5j * (d(m - 1) + d(m + 1))
    return dC0, dCcos, dCsin


def bessel_factors(dip, center_xy, lam, m, d_axis=None):
    """Frequency-independent Bessel factors of a source for order m about center_xy: (C0, Ccos, Csin) of the Graf
    decomposition (or their x/y derivatives) and, for a LOOP, J_1(lam a).  Cached by SphereModel (Bessel cache)."""
    lam = np.asarray(lam, float)
    dx, dy = dip.pos[0] - center_xy[0], dip.pos[1] - center_xy[1]
    rho_s = np.hypot(dx, dy); phi_s = np.arctan2(dy, dx)
    C = graf_coeffs(lam, m, rho_s, phi_s) if d_axis is None else graf_coeffs_d(lam, m, rho_s, phi_s, d_axis)
    jl = jv(1, lam * dip.radius) if dip.kind == 'LOOP' else None
    return C, jl


def dipole_modes(dip, center_xy, lam, m, k2, u, d_axis=None, factors=None):
    """Mode amplitudes (A+_TE, A-_TE, A+_TM, A-_TM) of dipole `dip` about the vertical axis through center_xy,
    for arrays lam, with k2 = k^2 and u of the SOURCE layer. z up in the sign conventions.
    factors: precomputed result of bessel_factors (frequency independent); computed here if None."""
    lam = np.asarray(lam, float)
    (C0, Cc, Cs), jl = factors if factors is not None else bessel_factors(dip, center_xy, lam, m, d_axis)
    cp, sp = np.cos(dip.psi), np.sin(dip.psi)
    M = dip.moment
    z = np.zeros_like(lam, dtype=complex)
    if dip.kind == 'VMD':
        te = M * lam ** 3 / (4 * np.pi * u) * C0
        return te, te, z, z
    if dip.kind == 'LOOP':
        te = M * dip.radius / 2 * lam ** 2 / u * jl * C0
        return te, te, z, z
    if dip.kind == 'HED':
        te = M * lam ** 2 / (4 * np.pi * u) * (cp * Cs - sp * Cc)
        tm = M * lam ** 2 / (4 * np.pi) * (cp * Cc + sp * Cs)
        return te, te, tm, -tm
    if dip.kind == 'HMD':
        te = M * lam ** 2 / (4 * np.pi) * (cp * Cc + sp * Cs)
        tm = k2 * M * lam ** 2 / (4 * np.pi * u) * (cp * Cs - sp * Cc)
        return te, -te, tm, tm
    raise ValueError(dip.kind)


def grounded_wire(p0, p1, current, n_gauss=5, zeta=0.0):
    """Straight grounded wire from p0 to p1 (x, y) at depth zeta carrying current `current` [A] as HED segments
    (Gauss-Legendre points, moment = I * dl_i)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0); psi = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    x, w = np.polynomial.legendre.leggauss(n_gauss)
    out = []
    for xi, wi in zip(x, w):
        t = 0.5 * (xi + 1); p = p0 + t * (p1 - p0)
        out.append(Dipole('HED', (p[0], p[1], zeta), moment=current * 0.5 * wi * L, psi=psi))
    return out


def polygon_wire(points, current, n_gauss=5, zeta=0.0, closed=False):
    """Polyline (grounded wire) or closed polygon loop as HED segments."""
    P = [np.asarray(p, float) for p in points]
    if closed:
        P = P + [P[0]]
    out = []
    for a, b in zip(P[:-1], P[1:]):
        out += grounded_wire(a, b, current, n_gauss, zeta)
    return out


def circular_loop(center_xy, radius, current, n_seg=64, n_gauss=3, zeta=0.0):
    """Circular loop (counter-clockwise seen from above) as a regular polygon with area pi a^2."""
    r = radius * np.sqrt(np.pi / ((n_seg / 2) * np.sin(2 * np.pi / n_seg)))
    ph = (np.arange(n_seg + 1) + 0.5) * 2 * np.pi / n_seg
    pts = [(center_xy[0] + r * np.cos(p), center_xy[1] + r * np.sin(p)) for p in ph[:-1]]
    return polygon_wire(pts, current, n_gauss, zeta, closed=True)


def grounded_dipole(center, psi=0.0, length=1.0, current=1.0, zeta=0.0, n_gauss=None):
    """Grounded electric dipole: two electrodes at distance `length` [m] around the centre `mitte` (x, y), direction
    angle psi (rad, from x towards y; current flows in direction psi inside the wire), current `current` [A].
    n_gauss=None: point dipole (one HED of moment I L, valid at distances >> L); n_gauss=k: finite wire as k
    Gauss-Legendre HEDs (exact electrode terms for any distance away from the wire)."""
    center = np.asarray(center, float)
    if n_gauss is None:
        return [Dipole('HED', (center[0], center[1], zeta), moment=current * length, psi=psi)]
    d = 0.5 * length * np.array([np.cos(psi), np.sin(psi)])
    return grounded_wire(center - d, center + d, current, n_gauss, zeta)


def hed_list(sources):
    """Flatten a source specification (Dipole, list of Dipole, nested lists) into HED dipoles on the surface, as needed
    by the closed-form half-space fields and by the hill: LOOP -> polygon of HEDs; VMD/HMD are not supported there."""
    out = []
    stack = [sources]
    while stack:
        q = stack.pop()
        if isinstance(q, Dipole):
            if q.kind == 'LOOP':
                stack.extend(q.as_polygon())
                continue
            if q.kind != 'HED':
                raise NotImplementedError(f"{q.kind}: the surface/hill code takes HED-type sources only")
            if q.pos[2] != 0.0:
                raise ValueError("surface/hill sources must lie on the surface (zeta = 0)")
            out.append(q)
        else:
            stack.extend(reversed(list(q)))
    return out
