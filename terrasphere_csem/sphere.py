"""Assembly: conductive (permeable) sphere in the layered earth, all degrees n <= nmax and orders |m| <= nmax,
TE and TM spherical modes, arbitrary dipole/loop/wire transmitters and dipole/loop/wire receivers.

Per angular frequency w and per azimuthal order m (the orders decouple because the layering is invariant under
rotations about the vertical axis through the sphere centre):
   regular field at the sphere:   c = c^p + R s          (c^p from the sources, R: sphere -> layers -> sphere)
   sphere (Mie T-matrix):         s = -T c
   ->  (I + T R) s = -T c^p,  then the scattered field at the receivers from s.
Vectors are ordered [(n=1..N, M-type), (n=1..N, N-type)]; everything is k-scaled:
   c_hat = k^n c,  s_hat = k^{-(n+1)} s,  T_hat = k^{-(2n+1)} T,  R_hat = k^n R k^{n'+1}.
Frames: global depth zeta (down); sphere frame z = zeta_c - zeta (up), horizontal coordinates relative to the
sphere axis. Receivers return E (V/m), H (A/m), dB/dt = -i w mu0 H (T/s, e^{-iwt}), or voltages of wires/loops."""
import numpy as np
from scipy.special import jv
from . import MU0
from .spectral import AB_regular_scaled, outgoing_spectrum_scaled, lam_grid, u_of
from .mie import mie_scaled, d_mie_scaled
from .sources import dipole_modes, bessel_factors, Dipole
from .fields import bessel_weights, mode_components
from .onedim import primary_field as primary_field_1d


class Sphere:
    """Homogeneous sphere: radius b [m], conductivity sigma_s [S/m], relative permeability mu_r."""
    def __init__(self, b, sigma_s, mu_r=1.0):
        """Store b, sigma_s and mu_r as floats."""
        self.b = float(b); self.sigma_s = float(sigma_s); self.mu_r = float(mu_r)


class Receiver:
    """typ 'E' / 'H' / 'dBdt': field component along vec (3-vector, z UP) at pos = (x, y, zeta) [V/m, A/m, T/s];
    typ 'LOOP': e.m.f. of a horizontal circular loop (radius, centre pos, counter-clockwise from above) [V]."""
    def __init__(self, kind, pos, vec=(0, 0, 1.0), radius=None):
        """kind: 'E', 'H', 'dBdt' or 'LOOP'; pos: (x, y, depth) [m]; vec: direction of the measured component
        (z up); radius: loop radius [m] (LOOP only)."""
        self.kind = kind; self.pos = np.asarray(pos, float); self.vec = np.asarray(vec, float); self.radius = radius

    def as_points(self, n_seg=64, n_gauss=3):
        """LOOP -> list of 'E' receivers with vec = dl (so that the sum of E.vec is the e.m.f.)."""
        if self.kind != 'LOOP':
            return [self]
        return loop_receivers(self.pos[:2], self.radius, self.pos[2], n_seg, n_gauss)


def wire_receivers(p0, p1, zeta=0.0, n_gauss=5):
    """Voltage integral int E.dl from p0 to p1 as 'E' receivers with vec = dl_i."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); L = np.linalg.norm(p1 - p0)
    t = (p1 - p0) / L
    x, w = np.polynomial.legendre.leggauss(n_gauss)
    return [Receiver('E', (p0[0] + 0.5 * (xi + 1) * (p1[0] - p0[0]), p0[1] + 0.5 * (xi + 1) * (p1[1] - p0[1]), zeta),
                     vec=(t[0] * 0.5 * wi * L, t[1] * 0.5 * wi * L, 0.0)) for xi, wi in zip(x, w)]


def loop_receivers(center_xy, radius, zeta=0.0, n_seg=64, n_gauss=3):
    """Voltage integral around a horizontal circular loop as a list of 'E' receivers with vec = dl_i.

    The loop (centre center_xy, radius [m], depth zeta) is replaced by an area-equivalent regular polygon of n_seg
    sides with n_gauss Gauss-Legendre points per side, counter-clockwise seen from above."""
    r = radius * np.sqrt(np.pi / ((n_seg / 2) * np.sin(2 * np.pi / n_seg)))
    ph = (np.arange(n_seg + 1) + 0.5) * 2 * np.pi / n_seg
    pts = [(center_xy[0] + r * np.cos(p), center_xy[1] + r * np.sin(p)) for p in ph]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        out += wire_receivers(a, b, zeta, n_gauss)
    return out


class SphereModel:
    """Conductive (permeable) sphere in a layered earth, solved for all degrees and orders up to nmax.

    layers: LayerStack (the sphere must lie entirely inside one ground layer); sphere: Sphere; center: (x_c, y_c,
    depth_c) of the sphere centre [m]; nmax: highest degree and order; lam_fac, npan_min, nq: extent, minimum panel
    number and nodes per panel of the Gauss-Legendre grid in the horizontal wavenumber. Sources are lists of Dipole
    objects, receivers lists of Receiver objects."""
    def __init__(self, layers, sphere, center, nmax=15, lam_fac=None, npan_min=80, nq=24):
        """Set up the geometry; see the class docstring for the parameters."""
        self.st = layers; self.sphere = sphere
        self.xc, self.yc, self.zc = (float(v) for v in center)
        self.s = layers.layer_of(self.zc)
        assert self.s >= 1, "sphere must be in the ground"
        self.h_up = self.zc - layers.top(self.s)                       # distance centre -> upper interface
        self.h_dn = layers.bottom(self.s) - self.zc                    # -> lower interface (inf for basement)
        assert self.h_up > sphere.b and self.h_dn > sphere.b, "sphere must lie entirely inside one layer"
        self.N = nmax; self.lam_fac = lam_fac or (45.0 + 2.5 * nmax); self.npan_min = npan_min; self.nq = nq
        self.sig_h = layers.sigma[self.s]
        self.m_tol = 1e-12          # relative truncation of the azimuthal orders

    # ------------------------------------------------------------------ grid
    # ------------------------------------------------------------------ Bessel cache
    # All Bessel functions of the model depend on the geometry (lambda grid, positions, orders) but not on the
    # frequency.  They are computed once and reused for every frequency of a transient.  The cache holds
    # the data of the most recent geometry only (a profile with moving sources builds one entry per position).
    def _geometry_key(self, sources, receivers):
        """Hashable description of the geometry (kinds, positions, orientations, sphere centre, nmax) that identifies
        the Bessel cache."""
        def key(o):
            """Hashable description of one source or receiver."""
            return (o.kind, tuple(np.round(o.pos, 9)), round(float(getattr(o, 'psi', 0.0)), 12), o.radius,
                    tuple(np.round(getattr(o, 'vec', ()), 12)) if hasattr(o, 'vec') else None)
        return (tuple(key(q) for q in sources), tuple(key(e) for e in receivers), self.xc, self.yc, self.zc, self.N)

    def _cache_for(self, sources, receivers):
        """Bessel cache of the geometry; rebuilt when the geometry changes (only the most recent one is kept)."""
        gk = self._geometry_key(sources, receivers)
        c = getattr(self, '_bcache', None)
        if c is None or c['key'] != gk:
            lam, wl = self.grid(sources, receivers)
            c = {'key': gk, 'lam': lam, 'wl': wl, 'src': {}, 'rx': {}, 'disk': {}}
            self._bcache = c
        return c

    def _source_factors(self, c, dip, m, d_axis):
        """Cached frequency-independent Bessel factors of the source dip for order m (see sources.bessel_factors)."""
        k = (id(dip), tuple(dip.pos), dip.kind, m, d_axis)
        if k not in c['src']:
            c['src'][k] = bessel_factors(dip, (self.xc, self.yc), c['lam'], m, d_axis)
        return c['src'][k]

    def _rx_weights(self, c, mm, rho):
        """Cached Bessel-weighted quadrature weights (J_mm, J_mm', (mm/rho) J_mm) at lateral distance rho."""
        k = (mm, rho)
        if k not in c['rx']:
            c['rx'][k] = bessel_weights(mm, c['lam'], rho, wl=c['wl'])
        return c['rx'][k]

    def _rx_disk(self, c, mm, rho, phi, radius):
        """Cached disk kernel of a horizontal loop receiver: (2 pi a/lam) J_1(lam a) J_mm(lam rho) e^{i mm phi}."""
        k = (mm, rho, phi, radius)
        if k not in c['disk']:
            lam = c['lam']
            c['disk'][k] = 2 * np.pi * radius / lam * jv(1, lam * radius) * jv(mm, lam * rho) * np.exp(1j * mm * phi)
        return c['disk'][k]

    def grid(self, sources, receivers):
        """Gauss-Legendre grid (lam, weights) in the horizontal wavenumber for the given sources and receivers.

        The upper limit L = lam_fac/min(h) follows from the smallest vertical distance h to the interfaces, sources and
        far receivers; the panel number from the largest lateral extent."""
        hs = [2 * self.h_up] + ([2 * self.h_dn] if np.isfinite(self.h_dn) else [])
        rmax = 1.0
        for q in list(sources):
            dz = abs(q.pos[2] - self.zc); assert dz > self.sphere.b, "sources must be outside the sphere's depth band"
            hs.append(dz); rmax = max(rmax, np.hypot(q.pos[0] - self.xc, q.pos[1] - self.yc) + (q.radius or 0.0))
        for q in list(receivers):
            dz = abs(q.pos[2] - self.zc)
            if self.is_near_field(q):
                continue
            hs.append(dz); rmax = max(rmax, np.hypot(q.pos[0] - self.xc, q.pos[1] - self.yc) + (q.radius or 0.0))
        L = self.lam_fac / min(hs)
        npan = max(self.npan_min, int(10 * L * rmax / np.pi) + 40)
        return lam_grid(L, npan, self.nq)

    def _balance(self):
        """Diagonal balancing factors (d_c, d_s) of the regular and outgoing coefficients, based on (2n+1)!! and b^n.

        They remove the spread of many orders of magnitude between the degrees (exact similarity transform)."""
        n = np.arange(1, self.N + 1); b = self.sphere.b
        df = lambda q: np.array([np.prod(np.arange(1, 2 * qq + 2, 2)) for qq in q], float)
        return (np.concatenate([b ** n / df(n)] * 2), np.concatenate([df(n - 1) / b ** (n + 1)] * 2))

    def is_near_field(self, e):
        """Receivers inside the sphere's depth band are evaluated directly with spherical waves (valid for
        b < r < min(h_up, h_dn), i.e. inside the sphere layer and the convergence ball of the reflected field)."""
        dz = abs(e.pos[2] - self.zc)
        if dz > self.sphere.b or e.kind == 'LOOP':
            return False
        r = np.sqrt((e.pos[0] - self.xc) ** 2 + (e.pos[1] - self.yc) ** 2 + dz ** 2)
        assert self.sphere.b < r < min(self.h_up, self.h_dn), "near-field receiver must be outside the sphere and inside its layer's convergence ball"
        return True

    # ------------------------------------------------------------------ per-frequency, per-order assembly
    def _prepare(self, w, lam):
        """Frequency-dependent ingredients for angular frequency w and wavenumber grid lam.

        Returns (k, ks, u, R, T): wavenumber of the host layer and of the sphere, decay constant u(lam) in the host layer,
        reflection data R[q] = (Rup, Rdn, Rup_s e^{-2 u h_up}, Rdn_s e^{-2 u h_dn}) for q = 'TE', 'TM', and the scaled Mie
        coefficients T (M-type then N-type)."""
        k = np.sqrt(1j * w * MU0 * self.sig_h + 0j); k = k if np.imag(k) >= 0 else -k
        ks = np.sqrt(1j * w * MU0 * self.sphere.mu_r * self.sphere.sigma_s + 0j); ks = ks if np.imag(ks) >= 0 else -ks
        u = u_of(lam, k)
        R = {}
        for q in ('TE', 'TM'):
            Rup, Rdn = self.st.reflections(q, lam, w)
            R[q] = (Rup, Rdn, Rup[self.s] * np.exp(-2 * u * self.h_up),
                    Rdn[self.s] * np.exp(-2 * u * self.h_dn) if np.isfinite(self.h_dn) else np.zeros_like(u))
        a_hat, b_hat = mie_scaled(self.N, k, ks, self.sphere.b, self.sphere.mu_r)
        T = np.concatenate([b_hat[1:], a_hat[1:]])            # (n, M) <- b_n ; (n, N) <- a_n
        return k, ks, u, R, T

    def _order(self, m, w, lam, wl, k, u, R, sources, need_R=True, d_axis=None, cache=None):
        """Returns (c_hat, R_hat, spectra) for order m (R_hat = None if need_R is False or no excitation).
        d_axis in {'x','y'}: c_hat is replaced by its derivative w.r.t. the source positions along that axis."""
        N = self.N
        yu, yd = 1j * u, -1j * u
        ATEu, BTEu, ATMu, BTMu = AB_regular_scaled(N, m, yu, k, lam)
        ATEd, BTEd, ATMd, BTMd = AB_regular_scaled(N, m, yd, k, lam)
        cTE = lam / (w * MU0); cTM = -self.sig_h * lam / k        # W-amplitude -> scalar (H_z, J_z)
        # regular-expansion rows (n = 1..N) for unit scalar waves: up (index u) and down (index d), per family
        rows = {'TE': ((ATEu[1:] / cTE, BTEu[1:] / cTE), (ATEd[1:] / cTE, BTEd[1:] / cTE)),
                'TM': ((ATMu[1:] / cTM, BTMu[1:] / cTM), (ATMd[1:] / cTM, BTMd[1:] / cTM))}
        # ---- c_hat^p from the sources
        ch = np.zeros(2 * N, complex)
        # sources at the same depth share the propagation: sum their mode amplitudes first (linear)
        sums = {}
        for dip in sources:
            sl = self.st.layer_of(dip.pos[2]); k2s = 1j * w * MU0 * self.st.sigma[sl]; us = self.st.u(sl, lam, w)
            factor = self._source_factors(cache, dip, m, d_axis) if cache is not None else None
            A = dipole_modes(dip, (self.xc, self.yc), lam, m, k2s, us, d_axis, factors=factor)
            key = float(dip.pos[2])
            if key not in sums:
                sums[key] = [np.zeros_like(lam, dtype=complex) for _ in range(4)]
            for i in range(4):
                sums[key][i] = sums[key][i] + A[i]
        for zeta_s, A in sums.items():
            for iq, q in enumerate(('TE', 'TM')):
                Aup, Adn = A[2 * iq], A[2 * iq + 1]
                if not np.any(Aup) and not np.any(Adn):
                    continue
                Pu, Pd = self.st.propagate(q, lam, w, zeta_s, Aup, Adn, self.zc, R[q][0], R[q][1])
                (Au, Bu), (Ad, Bd) = rows[q]
                ch[:N] += (Au * wl) @ Pu + (Ad * wl) @ Pd
                ch[N:] += (Bu * wl) @ Pu + (Bd * wl) @ Pd
        aMTEu, aMTMu, aNTEu, aNTMu = outgoing_spectrum_scaled(N, m, yu, k, lam)
        aMTEd, aMTMd, aNTEd, aNTMd = outgoing_spectrum_scaled(N, m, yd, k, lam)
        spectra = {'TE': {'M': (cTE * aMTEu[1:], cTE * aMTEd[1:]), 'N': (cTE * aNTEu[1:], cTE * aNTEd[1:])},
                   'TM': {'M': (cTM * aMTMu[1:], cTM * aMTMd[1:]), 'N': (cTM * aNTMu[1:], cTM * aNTMd[1:])}}
        if not need_R or (sources and not np.any(ch)):
            return ch, None, spectra
        # ---- R_hat: outgoing unit (n', X) -> direct scalars up/down -> incoming -> regular rows
        Rh = np.zeros((2 * N, 2 * N), complex)
        for q in ('TE', 'TM'):
            Ru, Rd = R[q][2], R[q][3]; den = 1 - Ru * Rd
            (Au, Bu), (Ad, Bd) = rows[q]
            for ix, X in enumerate(('M', 'N')):
                Ap, Am = spectra[q][X]                          # (N, nlam): direct up / down amplitudes
                Din = Ru * (Ap + Rd * Am) / den                 # incoming down-going (regular at the sphere)
                Uin = Rd * (Am + Ru * Ap) / den                 # incoming up-going
                Rh[:N, ix * N:(ix + 1) * N] += k * ((Ad * wl) @ Din.T + (Au * wl) @ Uin.T)
                Rh[N:, ix * N:(ix + 1) * N] += k * ((Bd * wl) @ Din.T + (Bu * wl) @ Uin.T)
        return ch, Rh, spectra

    def solve(self, w, sources, receivers=(), m_list=None):
        """Solve for all orders: returns dict m -> (s_hat, spectra, c_hat, R_hat) plus (lam, wl, k, u, R, cache)."""
        cache = self._cache_for(sources, receivers)
        lam, wl = cache['lam'], cache['wl']
        k, ks, u, R, T = self._prepare(w, lam)
        N = self.N
        out = {}
        # skin-depth guard (as in reference 2): the sphere's field at the nearest interface is ~ e^{-Im k (h-b)};
        # beyond e^{-60} the scattered field is zero to double precision and the Mie ratios would overflow
        if np.imag(k) * (min(self.h_up, self.h_dn) - self.sphere.b) > 60 or np.imag(ks) * self.sphere.b > 300:
            return out, (lam, wl, k, u, R, cache)
        if m_list is None:          # adaptive: orders 0, +-1, +-2, ... until the excitation becomes negligible
            m_list = [0] + [sgn * mm for mm in range(1, N + 1) for sgn in (1, -1)]
        cmax = 0.0; stale = 0
        # balancing (exact diagonal similarity transform): regular coefficients ~ (kb)^n/(2n+1)!!, outgoing
        # ~ (2n-1)!!/(kb)^(n+1); without it the rows of degrees > 25 spread over 50 orders of magnitude and
        # round-off from the high degrees pollutes the low ones
        n = np.arange(1, N + 1); b = self.sphere.b
        df = lambda q: np.array([np.prod(np.arange(1, 2 * qq + 2, 2)) for qq in q], float)   # (2q+1)!!
        d_c = np.concatenate([b ** n / df(n), b ** n / df(n)])
        d_s = np.concatenate([df(n - 1) / b ** (n + 1), df(n - 1) / b ** (n + 1)])
        for m in m_list:
            ch, Rh, spectra = self._order(m, w, lam, wl, k, u, R, sources, cache=cache)
            if Rh is None:
                continue
            Tt = T * d_s / d_c; Rt = (d_c[:, None] * Rh) / d_s[None, :]
            s_hat = np.linalg.solve(np.eye(2 * N) + Tt[:, None] * Rt, -Tt * (d_c * ch)) / d_s
            out[m] = (s_hat, spectra, ch, Rh)
            cm = np.max(np.abs(T * ch)); cmax = max(cmax, cm)
            if cm < self.m_tol * cmax:
                stale += 1
                if stale >= 2:
                    break
            else:
                stale = 0
        return out, (lam, wl, k, u, R, cache)

    def scattered_field(self, w, sources, receivers, solution=None, s_override=None, d_axis=None):
        """Scattered (sphere) part of the receiver responses, array over receivers.  s_override: dict m -> s_hat
        to evaluate the same receiver map for other outgoing coefficients (derivatives); d_axis: differentiate
        the receiver map w.r.t. the receiver positions along 'x' or 'y'."""
        if solution is None:
            solution = self.solve(w, sources, receivers)
        sol, (lam, wl, k, u, R, cache) = solution
        if cache['key'] != self._geometry_key(sources, receivers):          # receivers differ from those of the solution
            cache = self._cache_for(sources, receivers)
            assert cache['lam'] is lam or np.array_equal(cache['lam'], lam), "receiver set changes the lambda grid"
        N = self.N
        vals = np.zeros(len(receivers), complex)
        for m, (s_hat0, spectra, ch, Rh) in sol.items():
            s_hat = s_override[m] if s_override is not None else s_hat0
            # direct scalar amplitudes of the sphere as a source at zeta_c (up / down), per family
            amp = {}
            for q in ('TE', 'TM'):
                Ap = k * (s_hat[:N] @ spectra[q]['M'][0] + s_hat[N:] @ spectra[q]['N'][0])
                Am = k * (s_hat[:N] @ spectra[q]['M'][1] + s_hat[N:] @ spectra[q]['N'][1])
                amp[q] = (Ap, Am)
            for ie, e in enumerate(receivers):
                dx, dy = e.pos[0] - self.xc, e.pos[1] - self.yc
                rho, phi = np.hypot(dx, dy), np.arctan2(dy, dx)
                if self.is_near_field(e):
                    assert d_axis is None, "position derivatives not available for near-field receivers"
                    vals[ie] += self._near_field_value(e, m, s_hat, Rh, k, w, dx, dy, self.zc - e.pos[2])
                    continue
                o = self.st.layer_of(e.pos[2]); sig_o = self.st.sigma[o]; uo = self.st.u(o, lam, w)
                # mode-shift representation of d/dx_r, d/dy_r:  d_x psi_m = (l/2)(psi_{m-1} - psi_{m+1}),
                # d_y psi_m = (i l/2)(psi_{m-1} + psi_{m+1});  terms = [(order, factor)]
                if d_axis is None:
                    terms = [(m, np.ones_like(lam))]
                elif d_axis == 'x':
                    terms = [(m - 1, 0.5 * lam), (m + 1, -0.5 * lam)]
                else:
                    terms = [(m - 1, 0.5j * lam), (m + 1, 0.5j * lam)]
                if e.kind == 'LOOP':
                    Pu, Pd = self.st.propagate('TE', lam, w, self.zc, amp['TE'][0], amp['TE'][1], e.pos[2], R['TE'][0], R['TE'][1])
                    for mm, fac in terms:
                        disk = self._rx_disk(cache, mm, rho, phi, e.radius)
                        vals[ie] += 1j * w * MU0 * np.sum(wl * fac * disk * (Pu + Pd))   # e.m.f. = i w mu0 int H_z dA (e^{-iwt})
                    continue
                for q in ('TE', 'TM'):
                    if q == 'TM' and sig_o == 0:
                        continue
                    Pu, Pd = self.st.propagate(q, lam, w, self.zc, amp[q][0], amp[q][1], e.pos[2], R[q][0], R[q][1])
                    for mm, fac in terms:
                        wJ, wdJ, wmJ = (g * fac for g in self._rx_weights(cache, mm, rho))
                        E, H = mode_components(q, mm, lam, wJ, wdJ, wmJ, Pu, Pd, uo, sig_o, w, phi)
                        if e.kind == 'E':
                            vals[ie] += E @ e.vec
                        elif e.kind == 'H':
                            vals[ie] += H @ e.vec
                        elif e.kind == 'dBdt':
                            vals[ie] += -1j * w * MU0 * (H @ e.vec)
        return vals

    def _near_field_value(self, e, m, s_hat, Rh, k, w, dx, dy, dz):
        """Direct spherical-wave evaluation: outgoing s (M^(3), N^(3)) + reflected regular part c = R s."""
        from .vswf import MN_point
        N = self.N
        n = np.arange(1, N + 1)
        s_phys = s_hat * np.concatenate([k ** (n + 1), k ** (n + 1)])          # unscale s
        c_hat = Rh @ s_hat                                                   # regular part, scaled by k^n
        c_phys = c_hat / np.concatenate([k ** n, k ** n])
        r = np.sqrt(dx * dx + dy * dy + dz * dz); th = np.arccos(dz / r); ph = np.arctan2(dy, dx)
        E = np.zeros(3, complex); H = np.zeros(3, complex)
        for i, nn in enumerate(n):
            if nn < abs(m):
                continue
            M3, N3 = MN_point(nn, m, k, r, th, ph, 3); M1, N1 = MN_point(nn, m, k, r, th, ph, 1)
            E += s_phys[i] * M3 + s_phys[N + i] * N3 + c_phys[i] * M1 + c_phys[N + i] * N1
            H += (k / (1j * w * MU0)) * (s_phys[i] * N3 + s_phys[N + i] * M3 + c_phys[i] * N1 + c_phys[N + i] * M1)
        if e.kind == 'E':
            return E @ e.vec
        if e.kind == 'H':
            return H @ e.vec
        return -1j * w * MU0 * (H @ e.vec)

    def derivative(self, w, sources, receivers, param, solution=None):
        """Analytic derivative of the scattered responses with respect to param in {'sigma_s', 'b', 'mu_r'}:
        only the T-matrix depends on these, so  d s = -(I + T R)^{-1} dT (c + R s)  and dV = g . ds."""
        if solution is None:
            solution = self.solve(w, sources, receivers)
        sol, (lam, wl, k, u, R, cache) = solution
        ks = np.sqrt(1j * w * MU0 * self.sphere.mu_r * self.sphere.sigma_s + 0j); ks = ks if np.imag(ks) >= 0 else -ks
        a_hat, b_hat = mie_scaled(self.N, k, ks, self.sphere.b, self.sphere.mu_r)
        da, db = d_mie_scaled(self.N, k, ks, self.sphere.b, self.sphere.mu_r, param)
        if param == 'sigma_s':
            da, db = da / self.sphere.sigma_s, db / self.sphere.sigma_s
        T = np.concatenate([b_hat[1:], a_hat[1:]]); dT = np.concatenate([db[1:], da[1:]])
        ds = {}
        d_c, d_s = self._balance()
        for m, (s_hat, spectra, ch, Rh) in sol.items():
            Tt = T * d_s / d_c; Rt = (d_c[:, None] * Rh) / d_s[None, :]
            ds[m] = np.linalg.solve(np.eye(2 * self.N) + Tt[:, None] * Rt, -(dT * d_s / d_c) * (d_c * (ch + Rh @ s_hat))) / d_s
        return self.scattered_field(w, sources, receivers, solution=solution, s_override=ds)

    def derivative_position(self, w, sources, receivers, axis, solution=None):
        """Analytic derivative of the scattered responses w.r.t. the sphere-centre coordinate x_c ('x') or y_c ('y'):
        moving the centre by +d equals moving all sources and receivers by -d, so
        dV/dx_c = -[ g . ds + (dg/dx_r) . s ],  ds = -(I + T R)^{-1} T dc/dx_s  (R, T do not depend on x_c)."""
        if solution is None:
            solution = self.solve(w, sources, receivers)
        sol, (lam, wl, k, u, R, cache) = solution
        ks = np.sqrt(1j * w * MU0 * self.sphere.mu_r * self.sphere.sigma_s + 0j); ks = ks if np.imag(ks) >= 0 else -ks
        a_hat, b_hat = mie_scaled(self.N, k, ks, self.sphere.b, self.sphere.mu_r)
        T = np.concatenate([b_hat[1:], a_hat[1:]])
        ds = {}
        d_c, d_s = self._balance()
        for m, (s_hat, spectra, ch, Rh) in sol.items():
            dch, _, _ = self._order(m, w, lam, wl, k, u, R, sources, need_R=False, d_axis=axis, cache=cache)
            Tt = T * d_s / d_c; Rt = (d_c[:, None] * Rh) / d_s[None, :]
            ds[m] = np.linalg.solve(np.eye(2 * self.N) + Tt[:, None] * Rt, -Tt * (d_c * dch)) / d_s
        return -(self.scattered_field(w, sources, receivers, solution=solution, s_override=ds)
                 + self.scattered_field(w, sources, receivers, solution=solution, d_axis=axis))

    def derivative_depth(self, w, sources, receivers, h=None):
        """d/d zeta_c of the scattered responses by Richardson-extrapolated central differences (the depth enters
        every exponential factor; the analytic form is straightforward but not implemented)."""
        h = h or 1e-3 * self.sphere.b
        def f(d):
            """Scattered response with the sphere centre moved by d [m] in depth."""
            return SphereModel(self.st, self.sphere, (self.xc, self.yc, self.zc + d), self.N, self.lam_fac, self.npan_min, self.nq).scattered_field(w, sources, receivers)
        d1 = (f(h) - f(-h)) / (2 * h); d2 = (f(2 * h) - f(-2 * h)) / (4 * h)
        return (4 * d1 - d2) / 3

    def primary_field(self, w, sources, receivers):
        """1-D part (no sphere) of the receiver responses, from the digital-filter kernels (loops as polygons)."""
        dip1d = []
        for q in sources:
            dip1d += q.as_polygon()
        vals = np.zeros(len(receivers), complex)
        for ie, e in enumerate(receivers):
            pts = e.as_points()
            E, H = primary_field_1d(self.st, dip1d, [p.pos for p in pts], w)
            for ip, p in enumerate(pts):
                if p.kind == 'E':
                    vals[ie] += E[ip] @ p.vec
                elif p.kind == 'H':
                    vals[ie] += H[ip] @ p.vec
                elif p.kind == 'dBdt':
                    vals[ie] += -1j * w * MU0 * (H[ip] @ p.vec)
        return vals

    def response(self, w, sources, receivers, primary=True):
        """Total (primary + scattered) response at angular frequency w."""
        s = self.scattered_field(w, sources, receivers)
        return s + (self.primary_field(w, sources, receivers) if primary else 0.0)
