"""Field expansion of the Gaussian terrain with Hankel transforms instead of the periodic 2-D FFT (KH-030).

The terrain h = H f(r), f = exp(-r^2/2s^2), is rotationally symmetric about the vertical axis through the summit.
Every field is decomposed into azimuthal orders m about that axis; multiplication by the axisymmetric factors
F_j(r) = f^j/j! does not mix the orders, and the per-wavenumber operations of huegel.py depend only on
lambda = |xi|.  Each order m is therefore a one-dimensional problem in r, transformed with Hankel transforms of the
orders m-1, m, m+1; there is no periodic box.

Representation of an order m.  Physical: P_m = (V_r + i V_phi)_m, Q_m = (V_r - i V_phi)_m, Z_m = (V_z)_m (coefficients of
e^{i m phi}).  Spectral (common factor (1/2pi)(-i)^m e^{i m beta} dropped): with the Hankel pair
S(lambda) = int s(r) J_nu(lambda r) r dr,  s(r) = int S(lambda) J_nu(lambda r) lambda dlambda,
   V_par + i V_perp = -i H_{m+1}[P_m],   V_par - i V_perp = i H_{m-1}[Q_m],   V_z = H_m[Z_m],
(V_par, V_perp, V_z the components along xi^, z^ x xi^, z^ of the 2-D spectrum).  The order-n problem is then exactly
that of huegel.Reihe with (KX, KY) replaced by the components along xi^:
   curl:  (curl V)_par = -d_z V_perp,  (curl V)_perp = d_z V_par - i lambda V_z,  (curl V)_z = i lambda V_perp;
   d_z H0 (earth) = (i lambda H0z + sigma E0perp, -sigma E0par, -i lambda H0par), air without sigma terms;
   earth modes down (u a_par, a_perp, -i lambda a_par), up (-u rhoE a_par, rhoM a_perp, -i lambda rhoE a_par);
   air mode B (-i lambda, 0, lambda).
Quadrature: Gauss-Legendre panels in r on [0, r_max] (all forward transforms act on functions that vanish beyond
r_max: windowed reference data and products with F_j) and in lambda on [0, lambda_max] (low-pass
exp[-(lambda/lambda_c)^16]); the reference field is windowed by 0.5 erfc((r - R)/w) as in huegel.py, because source and
receiver lie in one plane and the spectral integrals of its high z-derivatives do not converge otherwise.
Orders |m| <= M; the azimuthal content of the reference field decays like (r/rho_source)^|m|.
Sphere: on the hill axis only (its orders coincide with the hill orders); an off-axis sphere needs huegel.py (FFT).
"""
import numpy as np
from scipy.special import jv, erfc

from . import MU0
from . import halbraum
from .quellen import hed_liste


def _mm(A, M):
    """complex A @ real M with two real matrix products (no complex copy of M)."""
    return A.real @ M + 1j * (A.imag @ M)


def gl_panels(a, b, npan, nq=16):
    x, w = np.polynomial.legendre.leggauss(nq)
    e = np.linspace(a, b, npan + 1)
    t = np.concatenate([0.5 * (e[i + 1] - e[i]) * x + 0.5 * (e[i + 1] + e[i]) for i in range(npan)])
    wt = np.concatenate([0.5 * (e[i + 1] - e[i]) * w for i in range(npan)])
    return t, wt


class Hankelgitter:
    """Radial and spectral quadrature, azimuthal sampling of the reference field, receivers.

    quellen: sources (HED type, on the surface, outside r_max); empf: receiver positions (px, py) on the terrain;
    s: Gaussian width; M: highest azimuthal order; n_phi: azimuthal samples of the reference field (> 2M)."""

    def __init__(self, quellen, empf, s=150.0, M=40, n_phi=128, r_win=860.0, w_win=40.0, r_max=None,
                 lam_max=0.25, kc=0.2, npan_r=64, npan_l=64, nq=16):
        self.s, self.M, self.n_phi, self.r_win, self.w_win = s, M, n_phi, r_win, w_win
        self.r_max = r_max or r_win + 8.0 * w_win
        self.quellen = hed_liste(quellen)
        rs = np.array([np.hypot(*d.pos[:2]) for d in self.quellen])
        assert rs.min() > self.r_max + 50.0, "sources must lie outside r_max (off the hill)"
        assert n_phi > 2 * M + 1
        self.r, self.wr = gl_panels(0.0, self.r_max, npan_r, nq)
        self.lam, self.wl = gl_panels(0.0, lam_max, npan_l, nq)
        self.phi_lp = np.exp(-(self.lam / kc) ** 16)
        self.f = np.exp(-self.r ** 2 / (2 * s ** 2))
        self.W = 0.5 * erfc((self.r - r_win) / w_win)
        px, py = (np.asarray(v, float) for v in empf)
        self.px, self.py = px, py
        self.rp = np.hypot(px, py); self.php = np.arctan2(py, px)
        assert self.rp.max() < r_win - 4.5 * w_win, "receivers must lie where the window is 1"
        self.fp = np.exp(-self.rp ** 2 / (2 * s ** 2))
        self._K, self._L, self._Lp = {}, {}, {}

    # Bessel matrices, signed orders via J_{-n} = (-1)^n J_n
    def _J(self, nu, a, b):
        n = abs(nu)
        J = jv(n, np.outer(a, b))
        return J if (nu >= 0 or n % 2 == 0) else -J

    def K(self, nu):                                                 # forward: (n_lam, n_r)
        if nu not in self._K:
            self._K[nu] = np.ascontiguousarray((self._J(nu, self.lam, self.r) * (self.r * self.wr)[None, :]).T)
        return self._K[nu]                                           # stored transposed: (n_r, n_lam)

    def L(self, nu):                                                 # inverse at the r nodes: (n_r, n_lam)
        if nu not in self._L:
            self._L[nu] = np.ascontiguousarray((self._J(nu, self.r, self.lam) * (self.lam * self.wl)[None, :]).T)
        return self._L[nu]                                           # stored transposed: (n_lam, n_r)

    def Lp(self, nu):                                                # inverse at the receivers: (P, n_lam)
        if nu not in self._Lp:
            self._Lp[nu] = np.ascontiguousarray((self._J(nu, self.rp, self.lam) * (self.lam * self.wl)[None, :]).T)
        return self._Lp[nu]                                          # stored transposed: (n_lam, P)

    def spek(self, m, P, Q, Z):
        """physical (P_m, Q_m, Z_m) on the r nodes (..., n_r) -> spectral (par, perp, z) (..., 3, n_lam)."""
        a = -1j * _mm(P, self.K(m + 1)); b = 1j * _mm(Q, self.K(m - 1)); c = _mm(Z, self.K(m))
        return np.stack([(a + b) / 2, (a - b) / 2j, c], axis=-2)

    def phys(self, m, S, at_rx=False):
        """spectral (..., 3, n_lam) -> physical (P_m, Q_m, Z_m) (..., 3, n_r) [or at the receivers]."""
        L = self.Lp if at_rx else self.L
        a = S[..., 0, :] + 1j * S[..., 1, :]; b = S[..., 0, :] - 1j * S[..., 1, :]
        return np.stack([_mm(1j * a, L(m + 1)), _mm(-1j * b, L(m - 1)), _mm(S[..., 2, :], L(m))], axis=-2)

    def ringpunkte(self):
        ph = 2 * np.pi * np.arange(self.n_phi) / self.n_phi
        R, PH = np.meshgrid(self.r, ph, indexing='ij')               # (n_r, n_phi)
        return R, PH

    def zerlege(self, F):
        """Cartesian field (3, n_r, n_phi) on the rings -> per order m: (P, Q, Z) (3, n_r) for m = -M..M (dict)."""
        R, PH = self.ringpunkte()
        c, s = np.cos(PH), np.sin(PH)
        Vr = F[0] * c + F[1] * s; Vp = -F[0] * s + F[1] * c
        Pc = np.fft.fft(Vr + 1j * Vp, axis=-1) / self.n_phi
        Qc = np.fft.fft(Vr - 1j * Vp, axis=-1) / self.n_phi
        Zc = np.fft.fft(F[2], axis=-1) / self.n_phi
        return {m: np.stack([Pc[:, m % self.n_phi], Qc[:, m % self.n_phi], Zc[:, m % self.n_phi]])
                for m in range(-self.M, self.M + 1)}

    def zusammensetzen(self, Vm):
        """per order m physical (P, Q, Z) at the receivers (..., 3, P) -> Cartesian (..., 3, P)."""
        Pr = sum(v[..., 0, :] * np.exp(1j * m * self.php) for m, v in Vm.items())
        Qr = sum(v[..., 1, :] * np.exp(1j * m * self.php) for m, v in Vm.items())
        Zr = sum(v[..., 2, :] * np.exp(1j * m * self.php) for m, v in Vm.items())
        Vr = (Pr + Qr) / 2; Vp = (Pr - Qr) / 2j
        c, s = np.cos(self.php), np.sin(self.php)
        return np.stack([Vr * c - Vp * s, Vr * s + Vp * c, Zr], axis=-2)


def primaer_ringe(g, freqs, boden, kugel=None):
    """Reference surface fields on the rings: dict comp -> (nf, n_r, n_phi) (flat layered earth + sphere)."""
    from .huegel import boden_schichten
    R, PH = g.ringpunkte()
    x, y = (R * np.cos(PH)).ravel(), (R * np.sin(PH)).ravel()
    p = halbraum.oberflaeche_schichten(x, y, freqs, boden_schichten(boden), g.quellen)
    if kugel is not None:
        for i, f in enumerate(np.atleast_1d(freqs)):
            pk = kugel.oberflaeche(2 * np.pi * f, g.quellen, x, y)
            for c in p:
                p[c][i] += pk[c]
    return {c: v.reshape(len(np.atleast_1d(freqs)), *R.shape) for c, v in p.items()}


class HankelReihe:
    """Taylor coefficients in H of the terrain fields at the receivers, one frequency at a time."""

    def __init__(self, g, boden):
        from .huegel import boden_schichten
        self.g = g
        self.st = boden_schichten(boden)
        self.sigma = self.st.sigma[1]
        F = [np.ones_like(g.f)]; Fp = [np.ones_like(g.fp)]
        for j in range(1, 80):
            F.append(F[-1] * g.f / j); Fp.append(Fp[-1] * g.fp / j)
        self.F, self.Fp = np.array(F), np.array(Fp)

    def rechne(self, w, prim, nord, kugel=None):
        """prim: dict Ex, Ey, Hx, Hy, Hz (n_r, n_phi) unwindowed reference field on the rings.
        Returns CE, CH (nord+1, 3, P) Cartesian at the receivers."""
        g, sig = self.g, self.sigma
        lam = g.lam
        u = np.sqrt(lam ** 2 - 1j * w * MU0 * sig)                    # e^{-iwt}, Re u > 0
        if self.st.J > 1:
            d2 = self.st.tiefen[1]
            rhoE = self.st.reflections('TE', lam, w)[1][1] * np.exp(-2 * u * d2)
            rhoM = self.st.reflections('TM', lam, w)[1][1] * np.exp(-2 * u * d2)
        else:
            rhoE = rhoM = np.zeros_like(u)
        W = g.W[:, None]
        Hm = g.zerlege(np.stack([prim['Hx'] * W, prim['Hy'] * W, prim['Hz'] * W]))
        Em = g.zerlege(np.stack([prim['Ex'] * W, prim['Ey'] * W, np.zeros_like(prim['Ex'])]))
        lp = g.phi_lp
        wv = np.stack([-1j * lam, np.zeros_like(lam), lam])
        outH = {}; outE = {}
        for m in range(-g.M, g.M + 1):
            P = g.spek(m, *Hm[m]) * lp; Q = g.spek(m, *Em[m]) * lp
            d1 = np.stack([1j * lam * P[2] + sig * Q[1], -sig * Q[0], -1j * lam * P[0]])
            a1 = np.stack([1j * lam * P[2], np.zeros_like(lam), -1j * lam * P[0]])

            def De(j):
                return (u ** 2) ** (j // 2) * (P if j % 2 == 0 else d1)

            def Da(j):
                return (lam ** 2) ** (j // 2) * (P if j % 2 == 0 else a1)

            Hd, Hu, Bs = [None], [None], [None]

            def D(mm, j):
                return De(j) if mm == 0 else u ** j * (Hd[mm] + (-1) ** j * Hu[mm])

            def curl_s(A, A1):
                return np.stack([-A1[1], A1[0] - 1j * lam * A[2], 1j * lam * A[1]]) / sig

            cH = np.zeros((nord + 1, 3, len(g.rp)), complex); cE = np.zeros_like(cH)
            cH[0] = g.phys(m, P, at_rx=True); cE[0] = g.phys(m, curl_s(De(0), De(1)), at_rx=True)
            for n in range(1, nord + 1):
                specs = np.empty((n, 3, len(lam)), complex)
                for j in range(1, n + 1):
                    mm = n - j
                    specs[j - 1] = (De(j) - Da(j)) if mm == 0 else (D(mm, j) - ((-lam) ** j * Bs[mm])[None] * wv)
                ph = g.phys(m, specs)                                  # (n, 3, n_r)
                R = -np.einsum('jr,jcr->cr', self.F[1:n + 1], ph)
                Rh = g.spek(m, R[0], R[1], R[2]) * lp
                Rpar, Rper, Rz = Rh
                a2 = (Rpar + 1j * Rz) / (u * (1 - rhoE) + lam * (1 + rhoE))
                a1n = Rper / (1 + rhoM)
                B = -(Rz + 1j * lam * (1 + rhoE) * a2) / lam
                hd = np.stack([u * a2, a1n, -1j * lam * a2])
                hu = np.stack([-u * rhoE * a2, rhoM * a1n, -1j * lam * rhoE * a2])
                if kugel is not None:
                    dhd, dhu, dB = kugel(w, m, hd, hu, u)
                    hd, hu, B = hd + dhd, hu + dhu, B + dB
                Hd.append(hd); Hu.append(hu); Bs.append(B)
                sh = np.stack([D(n - j, j) for j in range(n + 1)])            # (n+1, 3, n_lam)
                se = np.stack([curl_s(D(n - j, j), D(n - j, j + 1)) for j in range(n + 1)])
                ph_h = g.phys(m, sh, at_rx=True); ph_e = g.phys(m, se, at_rx=True)   # (n+1, 3, P)
                fl = self.Fp[:n + 1][:, None, :]
                cH[n] = (fl * ph_h).sum(0); cE[n] = (fl * ph_e).sum(0)
            outH[m] = cH; outE[m] = cE
        return g.zusammensetzen(outE), g.zusammensetzen(outH)


def koeffizienten(freqs, g, boden, nord=20, prim=None, kugel=None, verbose=False):
    """CE, CH (nf, nord+1, 3, P) at the receivers of Hankelgitter g."""
    r = HankelReihe(g, boden)
    freqs = np.atleast_1d(freqs)
    prim = prim if prim is not None else primaer_ringe(g, freqs, boden, kugel=kugel)
    CE = np.zeros((len(freqs), nord + 1, 3, len(g.rp)), complex); CH = np.zeros_like(CE)
    for i, f in enumerate(freqs):
        p = {c: prim[c][i] for c in prim}
        CE[i], CH[i] = r.rechne(2 * np.pi * f, p, nord, kugel=kugel.antwort_m if kugel is not None else None)
        if verbose:
            print(f"  f = {f:.4g} Hz ({i + 1}/{len(freqs)})", flush=True)
    return CE, CH
