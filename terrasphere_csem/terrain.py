"""Gaussian hill or valley on a homogeneous half-space: field expansion in the terrain height
(time convention e^{-iwt}).

Physics (quasi-static, mu = mu0 everywhere, time e^{-iwt}):
  earth z < h(x):  curl H = sigma E,  curl E = i w mu H   ->  (Lap + k^2) H = 0, div H = 0,  k^2 = i w mu sigma
  air   z > h(x):  curl H = 0, div H = 0 (sources lie on the flat ground far from the hill)  ->  H = H0_air - grad Phi
  surface z = h:   H continuous (all three components).  n.J = 0 follows from tangential continuity and curl H_air = 0.
The total field is primary (flat half-space, closed form: halfspace.py) + scattered.  Surface z = h = H f(x),
f = exp(-r^2/2s^2) (z UP).  Field expansion (Bruno-Reitich): scattered fields = sum_{n>=1} H^n (H_n,e ; H_n,a); the
boundary condition H_e(x, h) = H_a(x, h) is Taylor-expanded about z = 0; order n:
    H_n,e(x,0) - H_n,a(x,0) = - sum_{j=1}^{n} f^j/j! [ d_z^j H_{n-j,e} - d_z^j H_{n-j,a} ](x, 0),
with H_{0,e}, H_{0,a} the earth/air continuations of the primary field.  Earth modes e^{i xi.x + gamma z},
gamma = sqrt(kappa^2 - i w mu sigma) (Re gamma > 0); air modes e^{i xi.x - kappa z}.  Per wavenumber a 3x3 solve
(closed form).  Primary z-derivatives: d_z H0,e = (dx H0z + sigma E0y, dy H0z - sigma E0x, -(dx H0x + dy H0y)),
d_z^2 -> gamma^2; air: same without the sigma terms, d_z^2 -> kappa^2.  The first-order source is the thin sheet
sigma h E0_h.  Outputs on the terrain (earth side): Q(x, h) = sum_n H^n C_n(x), C_n = sum_{j=0}^{n} f^j/j! d_z^j
Q_{n-j}(x, 0), Q = H (continuous) and E = curl H / sigma.  The only place where the time convention enters is gamma
(e^{+iwt}: gamma = sqrt(kappa^2 + i w mu sigma)); the results are the complex conjugates of the e^{+iwt} code.
Horizontal: periodic box N x N, spacing dx; the primary field is multiplied by the analytic window
0.5 erfc((r - R)/w) before its horizontal derivatives are taken spectrally (exact where the window is 1).

Sources: HED-type (Dipole 'HED' on the surface, grounded wires, grounded dipoles, polygon loops; sources.py).  They
must lie off the hill: the window must vanish there (checked in Grid).
"""
import os
import numpy as np
import scipy.fft as sfft
from scipy.special import erfc

from . import MU0
from . import halfspace
from .sources import hed_list

WORKERS = int(os.environ.get('TERRASPHERE_WORKERS', '-1'))          # FFT threads


def _fft2(a):
    """Two-dimensional FFT over the last two axes with WORKERS threads."""
    return sfft.fft2(a, axes=(-2, -1), workers=WORKERS)


def _ifft2(a):
    """Two-dimensional inverse FFT over the last two axes with WORKERS threads."""
    return sfft.ifft2(a, axes=(-2, -1), workers=WORKERS)


def terrain_height(x, y, H, s):
    """Terrain height h(x, y) = H exp(-(x^2 + y^2)/(2 s^2)) (z up; H < 0: valley)."""
    return H * np.exp(-(np.asarray(x) ** 2 + np.asarray(y) ** 2) / (2 * s ** 2))


def window(r, r_c, w):
    """Analytic window 0.5 erfc((r - r_c)/w): spectrum ~ exp(-kappa^2 w^2 / 4), so spectral derivatives of high
    order are not polluted (a C-infinity but non-analytic step made the series diverge)."""
    return 0.5 * erfc((r - r_c) / w)


class Grid:
    """Periodic box centred on the hill.  sources: the sources (to exclude grid points closer than min_distance to a
    source dipole, and to check that the window vanishes at the sources).  y_cut > 0: additional cut towards
    y < -y_cut (coarse grids)."""

    def __init__(self, sources, n=512, dx=10.0, s=150.0, r_win=860.0, w_win=40.0, min_distance=100.0,
                 y_cut=0.0, w_y=70.0, max_window_at_source=1e-15):
        """n: grid points per side; dx: spacing [m]; s: hill width [m]; r_win, w_win: radius and width of the window [m];
        min_distance: grid points closer than this to a source are excluded [m]; y_cut, w_y: optional additional window
        towards y < -y_cut; max_window_at_source: largest allowed window value at a source."""
        self.n, self.dx, self.s = n, dx, s
        self.x = (np.arange(n) - n // 2) * dx                       # x[n//2] = 0
        self.X, self.Y = np.meshgrid(self.x, self.x, indexing='xy')  # arrays [iy, ix]
        k = 2 * np.pi * sfft.fftfreq(n, dx)
        self.KX, self.KY = np.meshgrid(k, k, indexing='xy')
        self.KAP = np.hypot(self.KX, self.KY)
        kap = np.where(self.KAP > 0, self.KAP, 1.0)
        self.EX, self.EY = self.KX / kap, self.KY / kap              # unit xi (0 at xi = 0)
        self.null = self.KAP == 0
        self.f = np.exp(-(self.X ** 2 + self.Y ** 2) / (2 * s ** 2))
        self.W = window(np.hypot(self.X, self.Y), r_win, w_win)
        if y_cut > 0:
            self.W = self.W * 0.5 * erfc((-self.Y - y_cut) / w_y)
        self.sources = hed_list(sources)
        sxy = np.array([d.pos[:2] for d in self.sources])
        wq = window(np.hypot(sxy[:, 0], sxy[:, 1]), r_win, w_win)
        if y_cut > 0:
            wq = wq * 0.5 * erfc((-sxy[:, 1] - y_cut) / w_y)
        if wq.max() > max_window_at_source:
            raise ValueError(f"a source lies inside the window (W = {wq.max():.1e}): sources must be off the hill")
        dmin = np.full(self.X.shape, np.inf)
        for xs, ys in sxy:
            dmin = np.minimum(dmin, np.hypot(self.X - xs, self.Y - ys))
        self.mask = (self.W > 1e-30) & (dmin > min_distance)
        self.W = np.where(self.mask, self.W, 0.0)
        self.i0 = n // 2                                             # index of x = 0 / y = 0

    def point_index(self, px, py, tol=1e-6):
        """Grid indices (iy, ix) of surface points (px, py) [m]; the points must lie on the grid (receivers are
        evaluated where the series lives; the grid spacing is the receiver spacing)."""
        px = np.atleast_1d(np.asarray(px, float)); py = np.atleast_1d(np.asarray(py, float))
        ix = np.rint(px / self.dx).astype(int) + self.n // 2; iy = np.rint(py / self.dx).astype(int) + self.n // 2
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        assert ok.all(), "receiver outside the box"
        assert np.allclose(self.x[ix], px, atol=tol * self.dx) and np.allclose(self.x[iy], py, atol=tol * self.dx), \
            "receivers must lie on grid points"
        assert (self.W[iy, ix] > 1 - 1e-8).all(), "receivers must lie where the window is 1 (r < R - 4.5 w)"
        return iy, ix

    def lines(self, spec):
        """spec (..., N, N) spectral -> physical values on the two lines: (..., 2, N) [0: along y at x=0, 1: along x
        at y=0]."""
        ex = np.exp(1j * self.KX[0] * self.i0 * self.dx)              # FFT positions are index * dx
        coly = sfft.ifft((spec * ex[None, :]).sum(-1) / self.n, axis=-1, workers=WORKERS)
        rowx = sfft.ifft((spec * ex[:, None]).sum(-2) / self.n, axis=-1, workers=WORKERS)
        return np.stack([coly, rowx], axis=-2)


class _PrimaryContinuation:
    """Spectral primary field at z = 0 and its z-derivatives, continued into the earth and into the air.

    The primary field (windowed and low-passed) is the order-0 solution; `earth(j)` and `air(j)` give its j-th
    z-derivative at z = 0 for the earth side (gamma^2 per double derivative) and the air side (kappa^2)."""

    def __init__(self, grid, prim, lowpass, gam, sigma):
        """prim: dict of the physical primary fields; lowpass: spectral low-pass (N, N); gam: earth wavenumber
        gamma(kappa); sigma: conductivity of the top layer."""
        KX, KY, KAP = grid.KX, grid.KY, grid.KAP
        Wn = grid.W
        self.KX, self.KY, self.sigma = KX, KY, sigma
        P = _fft2(np.stack([Wn * prim['Hx'], Wn * prim['Hy'], Wn * prim['Hz']])) * lowpass
        Q = _fft2(np.stack([Wn * prim['Ex'], Wn * prim['Ey']])) * lowpass
        divh = -(1j * KX * P[0] + 1j * KY * P[1])
        self.P = P
        self.d1 = np.stack([1j * KX * P[2] + sigma * Q[1], 1j * KY * P[2] - sigma * Q[0], divh])
        self.a1 = np.stack([1j * KX * P[2], 1j * KY * P[2], divh])
        self.g2, self.k2 = gam ** 2, KAP ** 2

    def earth(self, j):
        """Spectral j-th z-derivative at z = 0 of the primary field continued into the earth."""
        return (self.g2 ** (j // 2)) * (self.P if j % 2 == 0 else self.d1)

    def air(self, j):
        """Spectral j-th z-derivative at z = 0 of the primary field continued into the air."""
        return (self.k2 ** (j // 2)) * (self.P if j % 2 == 0 else self.a1)

    def curl_over_sigma(self, Hj, Hj1):
        """Spectral (curl H)/sigma, i.e. E, from the z-derivatives Hj = d_z^j H and Hj1 = d_z^{j+1} H at z = 0."""
        KX, KY = self.KX, self.KY
        return np.stack([1j * KY * Hj[2] - Hj1[1], Hj1[0] - 1j * KX * Hj[2], 1j * KX * Hj[1] - 1j * KY * Hj[0]]) / self.sigma


class _OrderAmplitudes:
    """Spectral amplitudes of the scattered fields of the orders computed so far.

    For each order m >= 1: hd and hu, the amplitudes of the e^{gamma z} and e^{-gamma z} parts in the earth, and the
    amplitude Bs of the air potential (index 0 is unused: order 0 is the primary field)."""

    def __init__(self, gam):
        self.gam = gam
        self.Hd, self.Hu, self.Bs = [None], [None], [None]

    def append(self, hd, hu, B):
        """Store the amplitudes of the next order."""
        self.Hd.append(hd); self.Hu.append(hu); self.Bs.append(B)

    def derivative(self, m, j):
        """Spectral j-th z-derivative at z = 0 (earth side) of the order-m scattered magnetic field."""
        return self.gam ** j * (self.Hd[m] + (-1) ** j * self.Hu[m])


class Series:
    """Taylor coefficients in H of the terrain fields for one frequency.

    `compute(w, prim, nord)` with prim = dict of physical arrays (N, N) Ex, Ey, Hx, Hy, Hz (unwindowed, valid where
    W > 0) returns CE, CH of shape (nord+1, 3, 2, N): coefficient n, component (x, y, z), line, position.
    With points = (iy, ix) (Grid.point_index) the outputs are taken at these grid points instead (full inverse
    FFT per term): CE, CH of shape (nord+1, 3, P).  E and H are the fields on the terrain, earth side.
    """

    def __init__(self, grid, ground, kc=0.2):
        """ground: conductivity of the homogeneous ground (float) or a LayerStack stack whose top layer carries the
        terrain (all interfaces below the valley floor)."""
        self.g = grid
        self.st = ground_layers(ground)
        self.sigma = self.st.sigma[1]
        assert kc < np.pi / grid.dx, "low-pass edge must lie below the Nyquist wavenumber"
        # steep spectral low-pass: the field expansion multiplies by gamma^j, kappa^j; any content above the
        # physical band (round-off, primary-table interpolation) would grow like (kappa s)^n
        self.phi = np.exp(-(grid.KAP / kc) ** 16) if kc else np.ones_like(grid.KAP)
        F = [np.ones_like(grid.f)]
        for j in range(1, 80):
            F.append(F[-1] * grid.f / j)                          # f^j / j!
        self.F = np.array(F)
        self.Flin = np.stack([self.F[:, :, grid.i0], self.F[:, grid.i0, :]], axis=1)   # (j, 2, N)

    def _output_mapping(self, points):
        """How the spectra are evaluated: (extract, Fout, shape).

        extract maps a spectrum to physical values on the two profiles (points is None) or at the grid points;
        Fout(j) is the weight f^j/j! at those positions; shape is the shape of one coefficient."""
        g = self.g
        if points is None:
            extract = g.lines
            Fout = lambda j: self.Flin[j][None]                              # (1, 2, N)
            shape = (3, 2, g.n)
        else:
            iy, ix = points
            extract = lambda spec: _ifft2(spec)[..., iy, ix]
            Fout = lambda j: self.F[j][iy, ix][None]                         # (1, P)
            shape = (3, len(iy))
        return extract, Fout, shape

    def _wave_reflections(self, w, gam):
        """Reflection factors (rhoE, rhoM) of the waves reflected by the layers below the top layer.

        The mode in the top layer is e^{gamma z} + rho e^{-gamma z} (z up), per family (TE: scalar H_z, our a_par mode;
        TM: scalar J_z, our a_perp mode), rho = R_dn(top layer) e^{-2 gamma d_1}; zero for a homogeneous ground."""
        KAP = self.g.KAP
        if self.st.J > 1:
            d_1 = self.st.depths[1]
            rhoE = self.st.reflections('TE', KAP, w)[1][1] * np.exp(-2 * gam * d_1)
            rhoM = self.st.reflections('TM', KAP, w)[1][1] * np.exp(-2 * gam * d_1)
        else:
            rhoE = rhoM = np.zeros_like(gam)
        return rhoE, rhoM

    def _boundary_residual(self, n, cont, amp, wv):
        """Low-passed spectrum Rh of the order-n boundary source.

        R = -sum_{j=1}^{n} f^j/j! [d_z^j H_{n-j,e} - d_z^j H_{n-j,a}] at z = 0 (physical space), where the order-0
        fields are the continuations of the primary field and the higher orders are the stored amplitudes (wv: air
        mode polarisation)."""
        g, KAP = self.g, self.g.KAP
        specs = np.empty((n, 3, g.n, g.n), complex)
        for j in range(1, n + 1):
            m = n - j
            if m == 0:
                specs[j - 1] = cont.earth(j) - cont.air(j)
            else:
                specs[j - 1] = amp.derivative(m, j) - ((-KAP) ** j * amp.Bs[m])[None] * wv
        real_space = _ifft2(specs)                                       # (n, 3, N, N)
        R = -np.einsum('jyx,jcyx->cyx', self.F[1:n + 1], real_space)
        return _fft2(R) * self.phi

    def _order_amplitudes(self, Rh, gam, rhoE, rhoM):
        """Amplitudes (hd, hu, B) of the order-n solution from the boundary source Rh (closed-form 3 x 3 solve per
        wavenumber, with the kappa = 0 limit handled separately)."""
        g = self.g
        KAP, EX, EY = g.KAP, g.EX, g.EY
        kap_s = np.where(g.null, 1.0, KAP)
        Rpar = EX * Rh[0] + EY * Rh[1]
        Rper = -EY * Rh[0] + EX * Rh[1]
        # per wavenumber: a_perp (1 + rhoM) = R_perp;  a_par [gamma (1 - rhoE) + kappa (1 + rhoE)] = R_par + i R_z
        a2 = (Rpar + 1j * Rh[2]) / (gam * (1 - rhoE) + KAP * (1 + rhoE))
        a1n = Rper / (1 + rhoM)
        B = -(Rh[2] + 1j * KAP * (1 + rhoE) * a2) / kap_s
        hd = np.stack([-EY * a1n + gam * EX * a2, EX * a1n + gam * EY * a2, -1j * KAP * a2])
        hu = np.stack([-EY * a1n * rhoM - gam * EX * a2 * rhoE, EX * a1n * rhoM - gam * EY * a2 * rhoE,
                       -1j * KAP * a2 * rhoE])
        # kappa = 0: uniform horizontal field, normal incidence (TM-type impedance 1/sigma)
        for c in (0, 1):
            hd[c][g.null] = Rh[c][g.null] / (1 + rhoM[g.null]); hu[c][g.null] = rhoM[g.null] * hd[c][g.null]
        hd[2][g.null] = 0.0; hu[2][g.null] = 0.0
        B[g.null] = 0.0
        return hd, hu, B

    def _order_coefficients(self, n, cont, amp, extract, Fout, shape):
        """Order-n coefficients (CE_n, CH_n) on the output positions: sum_j f^j/j! d_z^j Q_{n-j}(x, 0)."""
        accH = np.zeros(shape, complex); accE = np.zeros_like(accH)
        for j in range(0, n + 1):
            m = n - j
            if m == 0:
                sh = cont.earth(j); se = cont.curl_over_sigma(cont.earth(j), cont.earth(j + 1))
            else:
                sh = amp.derivative(m, j); se = cont.curl_over_sigma(amp.derivative(m, j), amp.derivative(m, j + 1))
            fl = Fout(j)
            accH += fl * extract(sh)
            accE += fl * extract(se)
        return accE, accH

    def compute(self, w, prim, nord, return_internal=False, points=None, sphere=None):
        """Taylor coefficients of the terrain fields for the angular frequency w.

        prim: dict of physical arrays (N, N) Ex, Ey, Hx, Hy, Hz of the primary field (unwindowed, valid where W > 0);
        nord: highest order. Returns CE, CH of shape (nord + 1, 3, 2, N) (or (nord + 1, 3, P) at `points` = (iy, ix));
        with return_internal also the per-order amplitudes. `sphere` is an optional hook adding the sphere response of
        each order."""
        g = self.g
        extract, Fout, shape = self._output_mapping(points)
        gam = np.sqrt(g.KAP ** 2 - 1j * w * MU0 * self.sigma)                 # e^{-iwt}
        rhoE, rhoM = self._wave_reflections(w, gam)
        cont = _PrimaryContinuation(g, prim, self.phi, gam, self.sigma)
        amp = _OrderAmplitudes(gam)
        wv = np.stack([-1j * g.KX, -1j * g.KY, g.KAP])                     # air mode polarisation
        CE = np.zeros((nord + 1,) + shape, complex)
        CH = np.zeros_like(CE)
        CH[0] = extract(cont.P)
        CE[0] = extract(cont.curl_over_sigma(cont.earth(0), cont.earth(1)))
        for n in range(1, nord + 1):
            Rh = self._boundary_residual(n, cont, amp, wv)
            hd, hu, B = self._order_amplitudes(Rh, gam, rhoE, rhoM)
            if sphere is not None:                                     # sphere below the terrain
                dhd, dhu, dB = sphere(w, hd, hu, gam)
                hd = hd + dhd; hu = hu + dhu; B = B + dB
            amp.append(hd, hu, B)
            CE[n], CH[n] = self._order_coefficients(n, cont, amp, extract, Fout, shape)
        if return_internal:
            return CE, CH, dict(Hd=amp.Hd, Hu=amp.Hu, Bs=amp.Bs, gam=gam)
        return CE, CH


def ground_layers(ground):
    """float sigma -> homogeneous LayerStack; LayerStack unchanged."""
    from .layers import LayerStack
    return ground if isinstance(ground, LayerStack) else LayerStack([0.0], [float(ground)])


def primary_on_grid(g, freqs, ground, block=40, sphere=None):
    """Primary surface fields of the grid's sources over the flat (layered) earth on all grid points with window > 0
    (closed-form half-space of the top layer + layer correction, halfspace.surface_fields_layered).
    Returns dict comp -> (nf, N, N)."""
    idx = np.nonzero(g.mask)
    x, y = g.X[idx], g.Y[idx]
    freqs = np.atleast_1d(freqs)
    out = {c: np.zeros((len(freqs), g.n, g.n), complex) for c in ['Ex', 'Ey', 'Hx', 'Hy', 'Hz']}
    for i0 in range(0, len(freqs), block):
        fb = freqs[i0:i0 + block]
        p = halfspace.surface_fields_layered(x, y, fb, ground_layers(ground), g.sources)
        for c in out:
            out[c][i0:i0 + len(fb)][(slice(None),) + idx] = p[c]
        if sphere is not None:                                        # sphere response to the sources (order 0, exact)
            for i, f in enumerate(fb):
                pk = sphere.surface_fields(2 * np.pi * f, g.sources, x, y)
                for c in out:
                    out[c][i0 + i][idx] += pk[c]
    return out


def coefficients(freqs, g, ground, nord=20, kc=0.2, prim=None, verbose=False, points=None, sphere=None):
    """Taylor coefficients for all frequencies (Hz): CE, CH arrays (nf, nord+1, 3, 2, N) on the two profiles, or
    (nf, nord+1, 3, P) at the grid points points = (iy, ix).  ground: sigma or LayerStack; sphere: SphereTerrain
    (sphere below the terrain, sphere_terrain.py) or None."""
    r = Series(g, ground, kc)
    freqs = np.atleast_1d(freqs)
    prim = prim if prim is not None else primary_on_grid(g, freqs, ground, sphere=sphere)
    shape = (3, 2, g.n) if points is None else (3, len(points[0]))
    CE = np.zeros((len(freqs), nord + 1) + shape, complex); CH = np.zeros_like(CE)
    for i, f in enumerate(freqs):
        p = {c: prim[c][i] for c in prim}
        CE[i], CH[i] = r.compute(2 * np.pi * f, p, nord, points=points,
                                 sphere=sphere.response if sphere is not None else None)
        if verbose:
            print(f"  f = {f:.4g} Hz ({i + 1}/{len(freqs)})", flush=True)
    return CE, CH


def partial_sum(C, H, nmax=None):
    """Partial sum sum_{n<=nmax} C_n H^n (C shape (nord+1, ...))."""
    nmax = C.shape[0] - 1 if nmax is None else nmax
    p = H ** np.arange(nmax + 1)
    return np.tensordot(p, C[:nmax + 1], axes=(0, 0))


def optimal_sum(C, H, nmin=4):
    """Optimal truncation: sum up to (but excluding) the order where |c_n H^n| (all components and positions of the
    leading axes jointly, per trailing element) is smallest for n >= nmin.  Returns (sum, last-term estimate).
    C shape (nord+1, ...)."""
    nord = C.shape[0] - 1
    T = C * (H ** np.arange(nord + 1)).reshape((-1,) + (1,) * (C.ndim - 1))
    a = np.abs(T)
    b = np.maximum(a[:-1], a[1:])                                    # odd/even orders may vanish by symmetry
    nbest = nmin + np.argmin(b[nmin:], axis=0)
    n = np.arange(nord + 1).reshape((-1,) + (1,) * (C.ndim - 1))
    S = np.where(n <= nbest[None], T, 0).sum(0)
    err = np.take_along_axis(b, nbest[None], axis=0)[0]
    return S, err


def pade(C, H, L=None, M=None):
    """Pade approximant [L/M] of sum c_n H^n evaluated at H, elementwise over trailing axes.
    Uses scaled coefficients d_n = c_n H^n and evaluates at eps = 1.  Default L = M = nord//2 (L+M <= nord)."""
    nord = C.shape[0] - 1
    if M is None:
        M = nord // 2
    if L is None:
        L = nord - M
    d = C * (H ** np.arange(nord + 1)).reshape((-1,) + (1,) * (C.ndim - 1))
    sh = d.shape[1:]
    d = d.reshape(nord + 1, -1)
    out = np.empty(d.shape[1], complex)
    for i in range(d.shape[1]):
        c = d[:, i]
        if M == 0:
            out[i] = c[:L + 1].sum(); continue
        A = np.array([[c[l - k] if l - k >= 0 else 0.0 for k in range(1, M + 1)] for l in range(L + 1, L + M + 1)])
        rhs = -np.array([c[l] for l in range(L + 1, L + M + 1)])
        sc = np.max(np.abs(A)) or 1.0
        try:
            q = np.linalg.lstsq(A / sc, rhs / sc, rcond=1e-13)[0]
        except np.linalg.LinAlgError:
            out[i] = c.sum(); continue
        q = np.concatenate([[1.0], q])
        p = np.array([sum(q[k] * c[l - k] for k in range(0, min(l, M) + 1)) for l in range(L + 1)])
        out[i] = p.sum() / q.sum()
    return out.reshape(sh)
