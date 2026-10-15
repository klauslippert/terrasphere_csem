# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klaus Lippert
"""Field expansion of the Gaussian terrain with Hankel transforms instead of the periodic 2-D FFT.

The terrain h = H f(r), f = exp(-r^2/2s^2), is rotationally symmetric about the vertical axis through the summit.
Every field is decomposed into azimuthal orders m about that axis; multiplication by the axisymmetric factors
F_j(r) = f^j/j! does not mix the orders, and the per-wavenumber operations of terrain.py depend only on
lambda = |xi|.  Each order m is therefore a one-dimensional problem in r, transformed with Hankel transforms of the
orders m-1, m, m+1; there is no periodic box.

Representation of an order m.  Physical: P_m = (V_r + i V_phi)_m, Q_m = (V_r - i V_phi)_m, Z_m = (V_z)_m (coefficients of
e^{i m phi}).  Spectral (common factor (1/2pi)(-i)^m e^{i m beta} dropped): with the Hankel pair
S(lambda) = int s(r) J_nu(lambda r) r dr,  s(r) = int S(lambda) J_nu(lambda r) lambda dlambda,
   V_par + i V_perp = -i H_{m+1}[P_m],   V_par - i V_perp = i H_{m-1}[Q_m],   V_z = H_m[Z_m],
(V_par, V_perp, V_z the components along xi^, z^ x xi^, z^ of the 2-D spectrum).  The order-n problem is then exactly
that of terrain.Series with (KX, KY) replaced by the components along xi^:
   curl:  (curl V)_par = -d_z V_perp,  (curl V)_perp = d_z V_par - i lambda V_z,  (curl V)_z = i lambda V_perp;
   d_z H0 (earth) = (i lambda H0z + sigma E0perp, -sigma E0par, -i lambda H0par), air without sigma terms;
   earth modes down (u a_par, a_perp, -i lambda a_par), up (-u rhoE a_par, rhoM a_perp, -i lambda rhoE a_par);
   air mode B (-i lambda, 0, lambda).
Quadrature: Gauss-Legendre panels in r on [0, r_max] (all forward transforms act on functions that vanish beyond
r_max: windowed reference data and products with F_j) and in lambda on [0, lambda_max] (low-pass
exp[-(lambda/lambda_c)^16]); the reference field is windowed by 0.5 erfc((r - R)/w) as in terrain.py, because source and
receiver lie in one plane and the spectral integrals of its high z-derivatives do not converge otherwise.
Orders |m| <= M; the azimuthal content of the reference field decays like (r/rho_source)^|m|.
Sphere: on the hill axis only (its orders coincide with the hill orders); an off-axis sphere needs terrain.py (FFT).
"""
import numpy as np
from scipy.special import jv, erfc

from . import MU0
from . import halfspace
from .sources import hed_list


def _mm(A, M):
    """complex A @ real M with two real matrix products (no complex copy of M)."""
    return A.real @ M + 1j * (A.imag @ M)


def gl_panels(a, b, npan, nq=16):
    """Composite Gauss-Legendre nodes and weights on [a, b] with npan panels of nq nodes each."""
    x, w = np.polynomial.legendre.leggauss(nq)
    e = np.linspace(a, b, npan + 1)
    t = np.concatenate([0.5 * (e[i + 1] - e[i]) * x + 0.5 * (e[i + 1] + e[i]) for i in range(npan)])
    wt = np.concatenate([0.5 * (e[i + 1] - e[i]) * w for i in range(npan)])
    return t, wt


class HankelGrid:
    """Radial and spectral quadrature, azimuthal sampling of the reference field, receivers.

    sources: sources (HED type, on the surface, outside r_max); receivers: receiver positions (px, py) on the terrain;
    s: Gaussian width; M: highest azimuthal order; n_phi: azimuthal samples of the reference field (> 2M)."""

    def __init__(self, sources, receivers, s=150.0, M=40, n_phi=128, r_win=860.0, w_win=40.0, r_max=None,
                 lam_max=0.25, kc=0.2, npan_r=64, npan_l=64, nq=16):
        """See the class docstring; s: hill width [m]; r_win, w_win: window radius and width [m]; r_max: outer radius of the
        radial grid (default r_win + 8 w_win); lam_max, kc: upper wavenumber limit and low-pass wavenumber [1/m];
        npan_r, npan_l, nq: panels in r and lam and nodes per panel."""
        self.s, self.M, self.n_phi, self.r_win, self.w_win = s, M, n_phi, r_win, w_win
        self.r_max = r_max or r_win + 8.0 * w_win
        self.sources = hed_list(sources)
        rs = np.array([np.hypot(*d.pos[:2]) for d in self.sources])
        assert rs.min() > self.r_max + 50.0, "sources must lie outside r_max (off the hill)"
        assert n_phi > 2 * M + 1
        self.r, self.wr = gl_panels(0.0, self.r_max, npan_r, nq)
        self.lam, self.wl = gl_panels(0.0, lam_max, npan_l, nq)
        self.phi_lp = np.exp(-(self.lam / kc) ** 16)
        self.f = np.exp(-self.r ** 2 / (2 * s ** 2))
        self.W = 0.5 * erfc((self.r - r_win) / w_win)
        px, py = (np.asarray(v, float) for v in receivers)
        self.px, self.py = px, py
        self.rp = np.hypot(px, py); self.php = np.arctan2(py, px)
        assert self.rp.max() < r_win - 4.5 * w_win, "receivers must lie where the window is 1"
        self.fp = np.exp(-self.rp ** 2 / (2 * s ** 2))
        self._K, self._L, self._Lp = {}, {}, {}

    # Bessel matrices, signed orders via J_{-n} = (-1)^n J_n
    def _J(self, nu, a, b):
        """Bessel matrix J_nu(a_i b_j) for a signed order nu (J_{-n} = (-1)^n J_n)."""
        n = abs(nu)
        J = jv(n, np.outer(a, b))
        return J if (nu >= 0 or n % 2 == 0) else -J

    def K(self, nu):                                                 # forward: (n_lam, n_r)
        """Forward Hankel matrix of order nu, shape (n_r, n_lam): J_nu(lam r) r w_r (stored for reuse)."""
        if nu not in self._K:
            self._K[nu] = np.ascontiguousarray((self._J(nu, self.lam, self.r) * (self.r * self.wr)[None, :]).T)
        return self._K[nu]                                           # stored transposed: (n_r, n_lam)

    def L(self, nu):                                                 # inverse at the r nodes: (n_r, n_lam)
        """Inverse Hankel matrix of order nu at the r nodes, shape (n_lam, n_r): J_nu(lam r) lam w_lam (stored for reuse)."""
        if nu not in self._L:
            self._L[nu] = np.ascontiguousarray((self._J(nu, self.r, self.lam) * (self.lam * self.wl)[None, :]).T)
        return self._L[nu]                                           # stored transposed: (n_lam, n_r)

    def Lp(self, nu):                                                # inverse at the receivers: (P, n_lam)
        """Inverse Hankel matrix of order nu at the receivers, shape (n_lam, P) (stored for reuse)."""
        if nu not in self._Lp:
            self._Lp[nu] = np.ascontiguousarray((self._J(nu, self.rp, self.lam) * (self.lam * self.wl)[None, :]).T)
        return self._Lp[nu]                                          # stored transposed: (n_lam, P)

    def to_spectral(self, m, P, Q, Z):
        """physical (P_m, Q_m, Z_m) on the r nodes (..., n_r) -> spectral (par, perp, z) (..., 3, n_lam)."""
        a = -1j * _mm(P, self.K(m + 1)); b = 1j * _mm(Q, self.K(m - 1)); c = _mm(Z, self.K(m))
        return np.stack([(a + b) / 2, (a - b) / 2j, c], axis=-2)

    def to_physical(self, m, S, at_rx=False):
        """spectral (..., 3, n_lam) -> physical (P_m, Q_m, Z_m) (..., 3, n_r) [or at the receivers]."""
        L = self.Lp if at_rx else self.L
        a = S[..., 0, :] + 1j * S[..., 1, :]; b = S[..., 0, :] - 1j * S[..., 1, :]
        return np.stack([_mm(1j * a, L(m + 1)), _mm(-1j * b, L(m - 1)), _mm(S[..., 2, :], L(m))], axis=-2)

    def ring_points(self):
        """Polar mesh (R, PH), both (n_r, n_phi), on which the reference field is sampled."""
        ph = 2 * np.pi * np.arange(self.n_phi) / self.n_phi
        R, PH = np.meshgrid(self.r, ph, indexing='ij')               # (n_r, n_phi)
        return R, PH

    def decompose(self, F):
        """Cartesian field (3, n_r, n_phi) on the rings -> per order m: (P, Q, Z) (3, n_r) for m = -M..M (dict)."""
        R, PH = self.ring_points()
        c, s = np.cos(PH), np.sin(PH)
        Vr = F[0] * c + F[1] * s; Vp = -F[0] * s + F[1] * c
        Pc = np.fft.fft(Vr + 1j * Vp, axis=-1) / self.n_phi
        Qc = np.fft.fft(Vr - 1j * Vp, axis=-1) / self.n_phi
        Zc = np.fft.fft(F[2], axis=-1) / self.n_phi
        return {m: np.stack([Pc[:, m % self.n_phi], Qc[:, m % self.n_phi], Zc[:, m % self.n_phi]])
                for m in range(-self.M, self.M + 1)}

    def assemble(self, Vm):
        """per order m physical (P, Q, Z) at the receivers (..., 3, P) -> Cartesian (..., 3, P)."""
        Pr = sum(v[..., 0, :] * np.exp(1j * m * self.php) for m, v in Vm.items())
        Qr = sum(v[..., 1, :] * np.exp(1j * m * self.php) for m, v in Vm.items())
        Zr = sum(v[..., 2, :] * np.exp(1j * m * self.php) for m, v in Vm.items())
        Vr = (Pr + Qr) / 2; Vp = (Pr - Qr) / 2j
        c, s = np.cos(self.php), np.sin(self.php)
        return np.stack([Vr * c - Vp * s, Vr * s + Vp * c, Zr], axis=-2)


def primary_on_rings(g, freqs, ground, sphere=None):
    """Reference surface fields on the rings: dict comp -> (nf, n_r, n_phi) (flat layered earth + sphere)."""
    from .terrain import ground_layers
    R, PH = g.ring_points()
    x, y = (R * np.cos(PH)).ravel(), (R * np.sin(PH)).ravel()
    p = halfspace.surface_fields_layered(x, y, freqs, ground_layers(ground), g.sources)
    if sphere is not None:
        for i, f in enumerate(np.atleast_1d(freqs)):
            pk = sphere.surface_fields(2 * np.pi * f, g.sources, x, y)
            for c in p:
                p[c][i] += pk[c]
    return {c: v.reshape(len(np.atleast_1d(freqs)), *R.shape) for c, v in p.items()}


class _PrimaryContinuation:
    """Spectral primary field of one azimuthal order at z = 0 and its z-derivatives in the earth and in the air.

    Components are (par, perp, z) along xi, z x xi, z. `earth(j)` and `air(j)` give the j-th z-derivative of the
    continuation of the primary field (u^2 resp. lam^2 per double derivative)."""

    def __init__(self, lam, P, Q, u, sigma):
        """lam: wavenumber nodes; P, Q: spectra of H and of E (low-passed) of the order; u: earth decay constant."""
        self.lam, self.u, self.sigma = lam, u, sigma
        self.P = P
        self.d1 = np.stack([1j * lam * P[2] + sigma * Q[1], -sigma * Q[0], -1j * lam * P[0]])
        self.a1 = np.stack([1j * lam * P[2], np.zeros_like(lam), -1j * lam * P[0]])

    def earth(self, j):
        """Spectral j-th z-derivative at z = 0 of the primary field continued into the earth."""
        return (self.u ** 2) ** (j // 2) * (self.P if j % 2 == 0 else self.d1)

    def air(self, j):
        """Spectral j-th z-derivative at z = 0 of the primary field continued into the air."""
        return (self.lam ** 2) ** (j // 2) * (self.P if j % 2 == 0 else self.a1)

    def curl_over_sigma(self, A, A1):
        """Spectral (curl V)/sigma from the z-derivatives A = d_z^j V and A1 = d_z^{j+1} V, components along xi, z x xi, z."""
        lam = self.lam
        return np.stack([-A1[1], A1[0] - 1j * lam * A[2], 1j * lam * A[1]]) / self.sigma


class _OrderAmplitudes:
    """Spectral amplitudes of the scattered fields of one azimuthal order, per Taylor order computed so far."""

    def __init__(self, u, cont):
        self.u, self.cont = u, cont
        self.Hd, self.Hu, self.Bs = [None], [None], [None]

    def append(self, hd, hu, B):
        """Store the amplitudes (e^{uz} part, e^{-uz} part, air potential) of the next Taylor order."""
        self.Hd.append(hd); self.Hu.append(hu); self.Bs.append(B)

    def derivative(self, mm, j):
        """Spectral j-th z-derivative at z = 0 (earth side) of the Taylor-order-mm field (mm = 0: primary field)."""
        return self.cont.earth(j) if mm == 0 else self.u ** j * (self.Hd[mm] + (-1) ** j * self.Hu[mm])


class HankelSeries:
    """Taylor coefficients in H of the terrain fields at the receivers, one frequency at a time."""

    def __init__(self, g, ground):
        """g: HankelGrid; ground: conductivity of a homogeneous ground or a LayerStack (top layer carries the terrain)."""
        from .terrain import ground_layers
        self.g = g
        self.st = ground_layers(ground)
        self.sigma = self.st.sigma[1]
        F = [np.ones_like(g.f)]; Fp = [np.ones_like(g.fp)]
        for j in range(1, 80):
            F.append(F[-1] * g.f / j); Fp.append(Fp[-1] * g.fp / j)
        self.F, self.Fp = np.array(F), np.array(Fp)

    def _wave_reflections(self, w, u):
        """Reflection factors (rhoE, rhoM) of the waves reflected by the layers below the top layer (zero for a
        homogeneous ground)."""
        lam = self.g.lam
        if self.st.J > 1:
            d2 = self.st.depths[1]
            rhoE = self.st.reflections('TE', lam, w)[1][1] * np.exp(-2 * u * d2)
            rhoM = self.st.reflections('TM', lam, w)[1][1] * np.exp(-2 * u * d2)
        else:
            rhoE = rhoM = np.zeros_like(u)
        return rhoE, rhoM

    def _boundary_residual(self, m, n, cont, amp, wv):
        """Low-passed spectrum (par, perp, z) of the boundary source of Taylor order n for azimuthal order m."""
        g, lam = self.g, self.g.lam
        specs = np.empty((n, 3, len(lam)), complex)
        for j in range(1, n + 1):
            mm = n - j
            if mm == 0:
                specs[j - 1] = cont.earth(j) - cont.air(j)
            else:
                specs[j - 1] = amp.derivative(mm, j) - ((-lam) ** j * amp.Bs[mm])[None] * wv
        ph = g.to_physical(m, specs)                                  # (n, 3, n_r)
        R = -np.einsum('jr,jcr->cr', self.F[1:n + 1], ph)
        return g.to_spectral(m, R[0], R[1], R[2]) * g.phi_lp

    def _order_amplitudes(self, Rh, u, rhoE, rhoM):
        """Amplitudes (hd, hu, B) of the Taylor-order-n solution from the boundary source Rh (closed-form solve per
        wavenumber)."""
        lam = self.g.lam
        Rpar, Rper, Rz = Rh
        a2 = (Rpar + 1j * Rz) / (u * (1 - rhoE) + lam * (1 + rhoE))
        a1n = Rper / (1 + rhoM)
        B = -(Rz + 1j * lam * (1 + rhoE) * a2) / lam
        hd = np.stack([u * a2, a1n, -1j * lam * a2])
        hu = np.stack([-u * rhoE * a2, rhoM * a1n, -1j * lam * rhoE * a2])
        return hd, hu, B

    def _order_coefficients(self, m, n, cont, amp):
        """Taylor-order-n coefficients (cE, cH) of azimuthal order m at the receivers, shape (3, P) each."""
        g = self.g
        sh = np.stack([amp.derivative(n - j, j) for j in range(n + 1)])            # (n+1, 3, n_lam)
        se = np.stack([cont.curl_over_sigma(amp.derivative(n - j, j), amp.derivative(n - j, j + 1)) for j in range(n + 1)])
        ph_h = g.to_physical(m, sh, at_rx=True); ph_e = g.to_physical(m, se, at_rx=True)   # (n+1, 3, P)
        fl = self.Fp[:n + 1][:, None, :]
        return (fl * ph_e).sum(0), (fl * ph_h).sum(0)

    def _order_series(self, w, m, Hm, Em, nord, u, rhoE, rhoM, wv, sphere):
        """Taylor coefficients (cE, cH), each (nord + 1, 3, P), of the azimuthal order m at the receivers."""
        g, lam = self.g, self.g.lam
        lp = g.phi_lp
        cont = _PrimaryContinuation(lam, g.to_spectral(m, *Hm) * lp, g.to_spectral(m, *Em) * lp, u, self.sigma)
        amp = _OrderAmplitudes(u, cont)
        cH = np.zeros((nord + 1, 3, len(g.rp)), complex); cE = np.zeros_like(cH)
        cH[0] = g.to_physical(m, cont.P, at_rx=True)
        cE[0] = g.to_physical(m, cont.curl_over_sigma(cont.earth(0), cont.earth(1)), at_rx=True)
        for n in range(1, nord + 1):
            Rh = self._boundary_residual(m, n, cont, amp, wv)
            hd, hu, B = self._order_amplitudes(Rh, u, rhoE, rhoM)
            if sphere is not None:
                dhd, dhu, dB = sphere(w, m, hd, hu, u)
                hd, hu, B = hd + dhd, hu + dhu, B + dB
            amp.append(hd, hu, B)
            cE[n], cH[n] = self._order_coefficients(m, n, cont, amp)
        return cE, cH

    def compute(self, w, prim, nord, sphere=None):
        """prim: dict Ex, Ey, Hx, Hy, Hz (n_r, n_phi) unwindowed reference field on the rings.
        Returns CE, CH (nord+1, 3, P) Cartesian at the receivers."""
        g = self.g
        lam = g.lam
        u = np.sqrt(lam ** 2 - 1j * w * MU0 * self.sigma)                    # e^{-iwt}, Re u > 0
        rhoE, rhoM = self._wave_reflections(w, u)
        W = g.W[:, None]
        Hm = g.decompose(np.stack([prim['Hx'] * W, prim['Hy'] * W, prim['Hz'] * W]))
        Em = g.decompose(np.stack([prim['Ex'] * W, prim['Ey'] * W, np.zeros_like(prim['Ex'])]))
        wv = np.stack([-1j * lam, np.zeros_like(lam), lam])
        outH = {}; outE = {}
        for m in range(-g.M, g.M + 1):
            outE[m], outH[m] = self._order_series(w, m, Hm[m], Em[m], nord, u, rhoE, rhoM, wv, sphere)
        return g.assemble(outE), g.assemble(outH)


def coefficients(freqs, g, ground, nord=20, prim=None, sphere=None, verbose=False):
    """CE, CH (nf, nord+1, 3, P) at the receivers of HankelGrid g."""
    r = HankelSeries(g, ground)
    freqs = np.atleast_1d(freqs)
    prim = prim if prim is not None else primary_on_rings(g, freqs, ground, sphere=sphere)
    CE = np.zeros((len(freqs), nord + 1, 3, len(g.rp)), complex); CH = np.zeros_like(CE)
    for i, f in enumerate(freqs):
        p = {c: prim[c][i] for c in prim}
        CE[i], CH[i] = r.compute(2 * np.pi * f, p, nord, sphere=sphere.response_order if sphere is not None else None)
        if verbose:
            print(f"  f = {f:.4g} Hz ({i + 1}/{len(freqs)})", flush=True)
    return CE, CH
