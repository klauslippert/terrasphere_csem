"""Gaussian hill or valley on a homogeneous half-space: field expansion in the terrain height (from gausshill,
GH-012/013; parameters instead of a global configuration and time convention e^{-iwt}, KH-003).

Physics (quasi-static, mu = mu0 everywhere, time e^{-iwt}):
  earth z < h(x):  curl H = sigma E,  curl E = i w mu H   ->  (Lap + k^2) H = 0, div H = 0,  k^2 = i w mu sigma
  air   z > h(x):  curl H = 0, div H = 0 (sources lie on the flat ground far from the hill)  ->  H = H0_air - grad Phi
  surface z = h:   H continuous (all three components).  n.J = 0 follows from tangential continuity and curl H_air = 0.
The total field is primary (flat half-space, closed form: halbraum.py) + scattered.  Surface z = h = H f(x),
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

Sources: HED-type (Dipol 'HED' on the surface, grounded wires, grounded dipoles, polygon loops; quellen.py).  They
must lie off the hill: the window must vanish there (checked in Gitter).
"""
import os
import numpy as np
import scipy.fft as sfft
from scipy.special import erfc

from . import MU0
from . import halbraum
from .quellen import hed_liste

WORKERS = int(os.environ.get('KH_WORKERS', '-1'))          # FFT threads


def _fft2(a):
    return sfft.fft2(a, axes=(-2, -1), workers=WORKERS)


def _ifft2(a):
    return sfft.ifft2(a, axes=(-2, -1), workers=WORKERS)


def form(x, y, H, s):
    """Terrain height h(x, y) = H exp(-(x^2 + y^2)/(2 s^2)) (z up; H < 0: valley)."""
    return H * np.exp(-(np.asarray(x) ** 2 + np.asarray(y) ** 2) / (2 * s ** 2))


def fenster(r, r_c, w):
    """Analytic window 0.5 erfc((r - r_c)/w): spectrum ~ exp(-kappa^2 w^2 / 4), so spectral derivatives of high
    order are not polluted (a C-infinity but non-analytic step made the series diverge, GH-013)."""
    return 0.5 * erfc((r - r_c) / w)


class Gitter:
    """Periodic box centred on the hill.  quellen: the sources (to exclude grid points closer than abstand_min to a
    source dipole, and to check that the window vanishes at the sources).  y_cut > 0: additional cut towards
    y < -y_cut (coarse grids, GH-018)."""

    def __init__(self, quellen, n=512, dx=10.0, s=150.0, r_win=860.0, w_win=40.0, abstand_min=100.0,
                 y_cut=0.0, w_y=70.0, w_quelle_max=1e-15):
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
        self.W = fenster(np.hypot(self.X, self.Y), r_win, w_win)
        if y_cut > 0:
            self.W = self.W * 0.5 * erfc((-self.Y - y_cut) / w_y)
        self.quellen = hed_liste(quellen)
        sxy = np.array([d.pos[:2] for d in self.quellen])
        wq = fenster(np.hypot(sxy[:, 0], sxy[:, 1]), r_win, w_win)
        if y_cut > 0:
            wq = wq * 0.5 * erfc((-sxy[:, 1] - y_cut) / w_y)
        if wq.max() > w_quelle_max:
            raise ValueError(f"a source lies inside the window (W = {wq.max():.1e}): sources must be off the hill")
        dmin = np.full(self.X.shape, np.inf)
        for xs, ys in sxy:
            dmin = np.minimum(dmin, np.hypot(self.X - xs, self.Y - ys))
        self.maske = (self.W > 1e-30) & (dmin > abstand_min)
        self.W = np.where(self.maske, self.W, 0.0)
        self.i0 = n // 2                                             # index of x = 0 / y = 0

    def punkte_index(self, px, py, tol=1e-6):
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

    def linien(self, spec):
        """spec (..., N, N) spectral -> physical values on the two lines: (..., 2, N) [0: along y at x=0, 1: along x
        at y=0]."""
        ex = np.exp(1j * self.KX[0] * self.i0 * self.dx)              # FFT positions are index * dx
        coly = sfft.ifft((spec * ex[None, :]).sum(-1) / self.n, axis=-1, workers=WORKERS)
        rowx = sfft.ifft((spec * ex[:, None]).sum(-2) / self.n, axis=-1, workers=WORKERS)
        return np.stack([coly, rowx], axis=-2)


class Reihe:
    """Taylor coefficients in H of the terrain fields for one frequency.

    `rechne(w, prim, nord)` with prim = dict of physical arrays (N, N) Ex, Ey, Hx, Hy, Hz (unwindowed, valid where
    W > 0) returns CE, CH of shape (nord+1, 3, 2, N): coefficient n, component (x, y, z), line, position.
    With punkte = (iy, ix) (Gitter.punkte_index) the outputs are taken at these grid points instead (full inverse
    FFT per term): CE, CH of shape (nord+1, 3, P).  E and H are the fields on the terrain, earth side.
    """

    def __init__(self, gitter, boden, kc=0.2):
        """boden: conductivity of the homogeneous ground (float) or a Schichten stack whose top layer carries the
        terrain (all interfaces below the valley floor, KH-010)."""
        self.g = gitter
        self.st = boden_schichten(boden)
        self.sigma = self.st.sigma[1]
        assert kc < np.pi / gitter.dx, "low-pass edge must lie below the Nyquist wavenumber (GH-020)"
        # steep spectral low-pass: the field expansion multiplies by gamma^j, kappa^j; any content above the
        # physical band (round-off, primary-table interpolation) would grow like (kappa s)^n (GH-013)
        self.phi = np.exp(-(gitter.KAP / kc) ** 16) if kc else np.ones_like(gitter.KAP)
        F = [np.ones_like(gitter.f)]
        for j in range(1, 80):
            F.append(F[-1] * gitter.f / j)                          # f^j / j!
        self.F = np.array(F)
        self.Flin = np.stack([self.F[:, :, gitter.i0], self.F[:, gitter.i0, :]], axis=1)   # (j, 2, N)

    def rechne(self, w, prim, nord, return_internal=False, punkte=None, kugel=None):
        g, sig = self.g, self.sigma
        if punkte is None:
            aus = g.linien
            Fout = lambda j: self.Flin[j][None]                              # (1, 2, N)
            shape = (3, 2, g.n)
        else:
            iy, ix = punkte
            aus = lambda spec: _ifft2(spec)[..., iy, ix]
            Fout = lambda j: self.F[j][iy, ix][None]                         # (1, P)
            shape = (3, len(iy))
        KX, KY, KAP, EX, EY = g.KX, g.KY, g.KAP, g.EX, g.EY
        gam = np.sqrt(KAP ** 2 - 1j * w * MU0 * sig)                 # e^{-iwt}
        # waves reflected by the layers below the top layer: mode = e^{gamma z} + rho e^{-gamma z} (z up), per family
        # (TE: scalar H_z, our a_par mode; TM: scalar J_z, our a_perp mode), rho = R_dn(top layer) e^{-2 gamma d_1}
        if self.st.J > 1:
            d_1 = self.st.tiefen[1]
            rhoE = self.st.reflections('TE', KAP, w)[1][1] * np.exp(-2 * gam * d_1)
            rhoM = self.st.reflections('TM', KAP, w)[1][1] * np.exp(-2 * gam * d_1)
        else:
            rhoE = rhoM = np.zeros_like(gam)
        Wn = g.W
        P = _fft2(np.stack([Wn * prim['Hx'], Wn * prim['Hy'], Wn * prim['Hz']])) * self.phi
        Q = _fft2(np.stack([Wn * prim['Ex'], Wn * prim['Ey']])) * self.phi
        divh = -(1j * KX * P[0] + 1j * KY * P[1])
        d1 = np.stack([1j * KX * P[2] + sig * Q[1], 1j * KY * P[2] - sig * Q[0], divh])
        a1 = np.stack([1j * KX * P[2], 1j * KY * P[2], divh])
        g2, k2 = gam ** 2, KAP ** 2

        def De(j):                                                    # spectral d_z^j H0 (earth continuation)
            return (g2 ** (j // 2)) * (P if j % 2 == 0 else d1)

        def Da(j):
            return (k2 ** (j // 2)) * (P if j % 2 == 0 else a1)

        def curl_s(Hj, Hj1):                                          # spectral (curl H)/sigma given d_z^j, d_z^{j+1}
            return np.stack([1j * KY * Hj[2] - Hj1[1], Hj1[0] - 1j * KX * Hj[2], 1j * KX * Hj[1] - 1j * KY * Hj[0]]) / sig

        wv = np.stack([-1j * KX, -1j * KY, KAP])                     # air mode polarisation
        Hd = [None]; Hu = [None]                                      # earth amplitudes per order: e^{gz}, e^{-gz} parts
        Bs = [None]                                                   # spectral air potential amplitudes

        def D(m, j):                                                  # spectral d_z^j H_m at z = 0 (earth side)
            return gam ** j * (Hd[m] + (-1) ** j * Hu[m])
        CE = np.zeros((nord + 1,) + shape, complex)
        CH = np.zeros_like(CE)
        CH[0] = aus(P)
        CE[0] = aus(curl_s(De(0), De(1)))
        kap_s = np.where(g.null, 1.0, KAP)
        for n in range(1, nord + 1):
            specs = np.empty((n, 3, g.n, g.n), complex)
            for j in range(1, n + 1):
                m = n - j
                if m == 0:
                    specs[j - 1] = De(j) - Da(j)
                else:
                    specs[j - 1] = D(m, j) - ((-KAP) ** j * Bs[m])[None] * wv
            phys = _ifft2(specs)                                       # (n, 3, N, N)
            R = -np.einsum('jyx,jcyx->cyx', self.F[1:n + 1], phys)
            Rh = _fft2(R) * self.phi
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
            if kugel is not None:                                     # sphere below the terrain (stage C hook)
                dhd, dhu, dB = kugel(w, hd, hu, gam)
                hd = hd + dhd; hu = hu + dhu; B = B + dB
            Hd.append(hd); Hu.append(hu); Bs.append(B)
            accH = np.zeros(shape, complex); accE = np.zeros_like(accH)
            for j in range(0, n + 1):
                m = n - j
                if m == 0:
                    sh = De(j); se = curl_s(De(j), De(j + 1))
                else:
                    sh = D(m, j); se = curl_s(D(m, j), D(m, j + 1))
                fl = Fout(j)
                accH += fl * aus(sh)
                accE += fl * aus(se)
            CH[n] = accH; CE[n] = accE
        if return_internal:
            return CE, CH, dict(Hd=Hd, Hu=Hu, Bs=Bs, gam=gam)
        return CE, CH


def boden_schichten(boden):
    """float sigma -> homogeneous Schichten; Schichten unchanged."""
    from .schichten import Schichten
    return boden if isinstance(boden, Schichten) else Schichten([0.0], [float(boden)])


def primaer_gitter(g, freqs, boden, block=40, kugel=None):
    """Primary surface fields of the grid's sources over the flat (layered) earth on all grid points with window > 0
    (closed-form half-space of the top layer + layer correction, halbraum.oberflaeche_schichten).
    Returns dict comp -> (nf, N, N)."""
    idx = np.nonzero(g.maske)
    x, y = g.X[idx], g.Y[idx]
    freqs = np.atleast_1d(freqs)
    out = {c: np.zeros((len(freqs), g.n, g.n), complex) for c in ['Ex', 'Ey', 'Hx', 'Hy', 'Hz']}
    for i0 in range(0, len(freqs), block):
        fb = freqs[i0:i0 + block]
        p = halbraum.oberflaeche_schichten(x, y, fb, boden_schichten(boden), g.quellen)
        for c in out:
            out[c][i0:i0 + len(fb)][(slice(None),) + idx] = p[c]
        if kugel is not None:                                        # sphere response to the sources (order 0, exact)
            for i, f in enumerate(fb):
                pk = kugel.oberflaeche(2 * np.pi * f, g.quellen, x, y)
                for c in out:
                    out[c][i0 + i][idx] += pk[c]
    return out


def koeffizienten(freqs, g, boden, nord=20, kc=0.2, prim=None, verbose=False, punkte=None, kugel=None):
    """Taylor coefficients for all frequencies (Hz): CE, CH arrays (nf, nord+1, 3, 2, N) on the two profiles, or
    (nf, nord+1, 3, P) at the grid points punkte = (iy, ix).  boden: sigma or Schichten; kugel: KugelGelaende
    (sphere below the terrain, kugel_gelaende.py) or None."""
    r = Reihe(g, boden, kc)
    freqs = np.atleast_1d(freqs)
    prim = prim if prim is not None else primaer_gitter(g, freqs, boden, kugel=kugel)
    shape = (3, 2, g.n) if punkte is None else (3, len(punkte[0]))
    CE = np.zeros((len(freqs), nord + 1) + shape, complex); CH = np.zeros_like(CE)
    for i, f in enumerate(freqs):
        p = {c: prim[c][i] for c in prim}
        CE[i], CH[i] = r.rechne(2 * np.pi * f, p, nord, punkte=punkte,
                                kugel=kugel.antwort if kugel is not None else None)
        if verbose:
            print(f"  f = {f:.4g} Hz ({i + 1}/{len(freqs)})", flush=True)
    return CE, CH


def summe(C, H, nmax=None):
    """Partial sum sum_{n<=nmax} C_n H^n (C shape (nord+1, ...))."""
    nmax = C.shape[0] - 1 if nmax is None else nmax
    p = H ** np.arange(nmax + 1)
    return np.tensordot(p, C[:nmax + 1], axes=(0, 0))


def summe_optimal(C, H, nmin=4):
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
