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


class Dipol:
    """typ in {'HED', 'VMD', 'HMD', 'LOOP'}; pos = (x, y, zeta) in the global frame (zeta = depth, positive down);
    moment: A m (HED), A m^2 (magnetic), current A (LOOP, horizontal circular loop of `radius`, counter-clockwise
    seen from above, exact spectrum H_z = (I a/2) int (lam^2/u) J_1(lam a) J_0(lam rho') e^{-u|dz|} dlam, TE only);
    psi: horizontal direction angle (rad) for HED/HMD."""
    def __init__(self, typ, pos, moment=1.0, psi=0.0, radius=None):
        self.typ = typ; self.pos = np.asarray(pos, float); self.moment = float(moment); self.psi = float(psi)
        self.radius = radius

    def als_polygon(self, n_seg=256, n_gauss=3):
        """LOOP -> list of HED segments (for the 1-D primary field via the digital filter)."""
        if self.typ != 'LOOP':
            return [self]
        return kreisschleife(self.pos[:2], self.radius, self.moment, n_seg, n_gauss, self.pos[2])

    def __repr__(self):
        return f"Dipol({self.typ}, pos={self.pos}, moment={self.moment}, psi={self.psi})"


def graf_coeffs(lam, m, rho_s, phi_s):
    """(C0, Ccos, Csin)_m for arrays lam."""
    lam = np.asarray(lam, float)
    Jm = jv(m, lam * rho_s); Jm1 = jv(m - 1, lam * rho_s); Jp1 = jv(m + 1, lam * rho_s)
    e0 = np.exp(-1j * m * phi_s); em1 = np.exp(-1j * (m - 1) * phi_s); ep1 = np.exp(-1j * (m + 1) * phi_s)
    C0 = Jm * e0
    Ccos = 0.5 * (Jm1 * em1 - Jp1 * ep1)
    Csin = -0.5j * (Jm1 * em1 + Jp1 * ep1)
    return C0, Ccos, Csin


def graf_coeffs_d(lam, m, rho_s, phi_s, achse):
    """d/d(x_s) or d/d(y_s) of (C0, Ccos, Csin)_m (source position relative to the centre), from
    d_x[J_nu(l rho) e^{-i nu phi}] = (l/2)[J_{nu-1} e^{-i(nu-1)phi} - J_{nu+1} e^{-i(nu+1)phi}],
    d_y[J_nu(l rho) e^{-i nu phi}] = -(i l/2)[J_{nu-1} e^{-i(nu-1)phi} + J_{nu+1} e^{-i(nu+1)phi}]."""
    lam = np.asarray(lam, float)
    def d(nu):
        a = jv(nu - 1, lam * rho_s) * np.exp(-1j * (nu - 1) * phi_s); b = jv(nu + 1, lam * rho_s) * np.exp(-1j * (nu + 1) * phi_s)
        return 0.5 * lam * (a - b) if achse == 'x' else -0.5j * lam * (a + b)
    dC0 = d(m)
    dCcos = 0.5 * (d(m - 1) - d(m + 1))
    dCsin = -0.5j * (d(m - 1) + d(m + 1))
    return dC0, dCcos, dCsin


def bessel_faktoren(dip, center_xy, lam, m, d_achse=None):
    """Frequency-independent Bessel factors of a source for order m about center_xy: (C0, Ccos, Csin) of the Graf
    decomposition (or their x/y derivatives) and, for a LOOP, J_1(lam a).  Cached by Modell (Bessel cache)."""
    lam = np.asarray(lam, float)
    dx, dy = dip.pos[0] - center_xy[0], dip.pos[1] - center_xy[1]
    rho_s = np.hypot(dx, dy); phi_s = np.arctan2(dy, dx)
    C = graf_coeffs(lam, m, rho_s, phi_s) if d_achse is None else graf_coeffs_d(lam, m, rho_s, phi_s, d_achse)
    jl = jv(1, lam * dip.radius) if dip.typ == 'LOOP' else None
    return C, jl


def dipol_moden(dip, center_xy, lam, m, k2, u, d_achse=None, faktoren=None):
    """Mode amplitudes (A+_TE, A-_TE, A+_TM, A-_TM) of dipole `dip` about the vertical axis through center_xy,
    for arrays lam, with k2 = k^2 and u of the SOURCE layer. z up in the sign conventions.
    faktoren: precomputed result of bessel_faktoren (frequency independent); computed here if None."""
    lam = np.asarray(lam, float)
    (C0, Cc, Cs), jl = faktoren if faktoren is not None else bessel_faktoren(dip, center_xy, lam, m, d_achse)
    cp, sp = np.cos(dip.psi), np.sin(dip.psi)
    M = dip.moment
    z = np.zeros_like(lam, dtype=complex)
    if dip.typ == 'VMD':
        te = M * lam ** 3 / (4 * np.pi * u) * C0
        return te, te, z, z
    if dip.typ == 'LOOP':
        te = M * dip.radius / 2 * lam ** 2 / u * jl * C0
        return te, te, z, z
    if dip.typ == 'HED':
        te = M * lam ** 2 / (4 * np.pi * u) * (cp * Cs - sp * Cc)
        tm = M * lam ** 2 / (4 * np.pi) * (cp * Cc + sp * Cs)
        return te, te, tm, -tm
    if dip.typ == 'HMD':
        te = M * lam ** 2 / (4 * np.pi) * (cp * Cc + sp * Cs)
        tm = k2 * M * lam ** 2 / (4 * np.pi * u) * (cp * Cs - sp * Cc)
        return te, -te, tm, tm
    raise ValueError(dip.typ)


def draht(p0, p1, strom, n_gauss=5, zeta=0.0):
    """Straight grounded wire from p0 to p1 (x, y) at depth zeta carrying current `strom` [A] as HED segments
    (Gauss-Legendre points, moment = I * dl_i)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0); psi = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    x, w = np.polynomial.legendre.leggauss(n_gauss)
    out = []
    for xi, wi in zip(x, w):
        t = 0.5 * (xi + 1); p = p0 + t * (p1 - p0)
        out.append(Dipol('HED', (p[0], p[1], zeta), moment=strom * 0.5 * wi * L, psi=psi))
    return out


def polygon_draht(punkte, strom, n_gauss=5, zeta=0.0, geschlossen=False):
    """Polyline (grounded wire) or closed polygon loop as HED segments."""
    P = [np.asarray(p, float) for p in punkte]
    if geschlossen:
        P = P + [P[0]]
    out = []
    for a, b in zip(P[:-1], P[1:]):
        out += draht(a, b, strom, n_gauss, zeta)
    return out


def kreisschleife(center_xy, radius, strom, n_seg=64, n_gauss=3, zeta=0.0):
    """Circular loop (counter-clockwise seen from above) as a regular polygon with area pi a^2."""
    r = radius * np.sqrt(np.pi / ((n_seg / 2) * np.sin(2 * np.pi / n_seg)))
    ph = (np.arange(n_seg + 1) + 0.5) * 2 * np.pi / n_seg
    pts = [(center_xy[0] + r * np.cos(p), center_xy[1] + r * np.sin(p)) for p in ph[:-1]]
    return polygon_draht(pts, strom, n_gauss, zeta, geschlossen=True)


def geerdeter_dipol(mitte, psi=0.0, laenge=1.0, strom=1.0, zeta=0.0, n_gauss=None):
    """Grounded electric dipole: two electrodes at distance `laenge` [m] around the centre `mitte` (x, y), direction
    angle psi (rad, from x towards y; current flows in direction psi inside the wire), current `strom` [A].
    n_gauss=None: point dipole (one HED of moment I L, valid at distances >> L); n_gauss=k: finite wire as k
    Gauss-Legendre HEDs (exact electrode terms for any distance away from the wire)."""
    mitte = np.asarray(mitte, float)
    if n_gauss is None:
        return [Dipol('HED', (mitte[0], mitte[1], zeta), moment=strom * laenge, psi=psi)]
    d = 0.5 * laenge * np.array([np.cos(psi), np.sin(psi)])
    return draht(mitte - d, mitte + d, strom, n_gauss, zeta)


def hed_liste(quellen):
    """Flatten a source specification (Dipol, list of Dipol, nested lists) into HED dipoles on the surface, as needed
    by the closed-form half-space fields and by the hill: LOOP -> polygon of HEDs; VMD/HMD are not supported there."""
    out = []
    stack = [quellen]
    while stack:
        q = stack.pop()
        if isinstance(q, Dipol):
            if q.typ == 'LOOP':
                stack.extend(q.als_polygon())
                continue
            if q.typ != 'HED':
                raise NotImplementedError(f"{q.typ}: the surface/hill code takes HED-type sources only")
            if q.pos[2] != 0.0:
                raise ValueError("surface/hill sources must lie on the surface (zeta = 0)")
            out.append(q)
        else:
            stack.extend(reversed(list(q)))
    return out
