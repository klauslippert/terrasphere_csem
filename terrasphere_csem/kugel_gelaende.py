"""Sphere below the terrain (KH-011): coupling of the sphere model (kugel.py: conductive sphere in the layered earth,
flat surface) with the field expansion of the terrain (huegel.py).

Every order of the field expansion is a problem with the FLAT surface z = 0 over the layered earth; with the sphere it
is the same problem with the sphere present.  Its solution is the flat-layered solution (huegel.Reihe) plus the field
the sphere radiates when it is excited by that solution (including all interactions with the layers and the flat
surface, contained in the reflection operator R of kugel.Modell).  The sphere-induced field needs no jump at z = 0.

Order 0: primary field of the sources over the layered earth + sphere response to the sources (kugel.Modell.loesen),
evaluated EXACTLY at the surface points of the window (radial tables per azimuthal order m) -> part of the order-0
surface data, which the series windows and differentiates like the primary field.
Orders n >= 1 (hook `antwort`, called by huegel.Reihe.rechne): the down-going part h_d of the order-n solution at
z = 0 (spectral, on the FFT grid) is transmitted through the layers to the sphere depth, expanded in regular
spherical waves (identity III for every grid plane wave, Jacobi-Anger e^{i xi.r} = sum_m i^m e^{-i m beta} J_m e^{i m phi}),
the sphere is solved per order m, (I + T R) s = -T c, and its outgoing field (identity IV at the grid wavenumbers)
is propagated to z = 0 (earth side, up- and down-going parts) and returned as spectral amplitudes on the grid:
   2-D density  S(xi) = sum_m P_m(lambda) e^{i m beta} / (2 pi i^m lambda),  FFT coefficient = N^2 dxi^2 S e^{-i xi.(x0 + xc)}.
Scalars -> our vectors: TE (H_z = P): a_par = i P / kappa; TM (J_z = P): a_perp = -i P / kappa; air potential
B = H_z(0) / kappa.  The grid sum is periodic like the rest of the series (box error handled as for the terrain).
Conditions: sphere entirely below z = 0 (and inside one layer, kugel.Modell); the terrain must not reach it.
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import jv

from . import MU0
from .spektral import AB_regular_scaled, outgoing_spectrum_scaled, lam_grid, u_of
from .mie import mie_scaled


class KugelGelaende:
    def __init__(self, km, g, kc=0.2, n_rho=700, nq=16):
        """km: kugel.Modell (layers incl. the sphere's layer; its sphere centre (xc, yc, zeta_c)); g: huegel.Gitter."""
        self.km, self.g = km, g
        b = km.kugel.b
        assert km.zc - b > 0, "the sphere must lie entirely below the surface z = 0"
        self.N = km.N
        n, dx = g.n, g.dx
        self.dxi = 2 * np.pi / (n * dx)
        x0 = (n // 2) * dx
        self.msk = (g.KAP > 0) & (g.KAP < kc * 1.05)                  # the low-pass removes everything beyond kc
        self.lam = g.KAP[self.msk]
        self.beta = np.arctan2(g.KY, g.KX)[self.msk]
        ph = g.KX * (x0 + km.xc) + g.KY * (x0 + km.yc)
        self.ein = np.exp(1j * ph)[self.msk] / n ** 2                 # FFT coefficient -> plane-wave amplitude at the axis
        self.aus = (n ** 2 * self.dxi ** 2) * np.exp(-1j * ph)[self.msk]
        self.mlist = [0] + [s * m for m in range(1, self.N + 1) for s in (1, -1)]
        self.ex = {m: np.exp(-1j * m * self.beta) * (1j) ** m for m in self.mlist}       # i^m e^{-i m beta}
        self.n_rho, self.nq = n_rho, nq
        self._w = None

    # ------------------------------------------------------------------ per frequency
    def frequenz(self, w):
        """Prepare frequency w: T, R per order m (LU of the balanced system), grid reflections."""
        km, st = self.km, self.km.st
        lam_ls, wl_ls = km.grid([], [])
        k, ks, u, R, T = km._prep(w, lam_ls)
        self.k, self.T = k, T
        self.aktiv = not (np.imag(k) * (min(km.h_up, km.h_dn) - km.kugel.b) > 60 or np.imag(ks) * km.kugel.b > 300)
        d_c, d_s = km._balance()
        self.d_c, self.d_s = d_c, d_s
        self.lu = {}
        if self.aktiv:
            for m in self.mlist:
                _, Rh, _ = km._order(m, w, lam_ls, wl_ls, k, u, R, [], need_R=True)
                Tt = T * d_s / d_c; Rt = (d_c[:, None] * Rh) / d_s[None, :]
                self.lu[m] = (lu_factor(np.eye(2 * self.N) + Tt[:, None] * Rt), Tt)
        lam = self.lam
        self.Rg = {q: st.reflections(q, lam, w) for q in ('TE', 'TM')}
        self.Rg0 = {}
        for q in ('TE', 'TM'):                                        # incident: total down-going wave given at z = 0
            Rup, Rdn = self.Rg[q]
            Rup0 = list(Rup); Rup0[1] = np.zeros_like(lam, dtype=complex)
            self.Rg0[q] = (Rup0, Rdn)
        self.u_s = u_of(lam, k)                                      # sphere layer
        self.cq = {'TE': lam / (w * MU0), 'TM': -km.sig_h * lam / k}
        self._w = w

    def _loese(self, m, ch):
        luf, Tt = self.lu[m]
        return lu_solve(luf, -Tt * (self.d_c * ch)) / self.d_s

    def _abstrahlung(self, sol, lam, w, rows=None):
        """Outgoing coefficients sol (m -> s_hat) -> scalars (Pu, Pd) per family at z = 0 (earth side), per m, for
        the wavenumbers lam."""
        km, st, N, k = self.km, self.km.st, self.N, self.k
        u = u_of(lam, k)
        out = {}
        cq = {'TE': lam / (w * MU0), 'TM': -km.sig_h * lam / k}
        Rg = st.reflections('TE', lam, w), st.reflections('TM', lam, w)
        for m, s_hat in sol.items():
            aMTEu, aMTMu, aNTEu, aNTMu = outgoing_spectrum_scaled(N, m, 1j * u, k, lam)
            aMTEd, aMTMd, aNTEd, aNTMd = outgoing_spectrum_scaled(N, m, -1j * u, k, lam)
            spec = {'TE': ((aMTEu[1:], aMTEd[1:]), (aNTEu[1:], aNTEd[1:])),
                    'TM': ((aMTMu[1:], aMTMd[1:]), (aNTMu[1:], aNTMd[1:]))}
            res = {}
            for iq, q in enumerate(('TE', 'TM')):
                (Mu, Md), (Nu, Nd) = spec[q]
                Ap = k * cq[q] * (s_hat[:N] @ Mu + s_hat[N:] @ Nu)
                Am = k * cq[q] * (s_hat[:N] @ Md + s_hat[N:] @ Nd)
                Rup, Rdn = Rg[iq]
                res[q] = st.propagate(q, lam, w, km.zc, Ap, Am, 0.0, Rup, Rdn)
            out[m] = res
        return out

    # ------------------------------------------------------------------ orders n >= 1
    def antwort(self, w, hd, hu, gam):
        """Hook for huegel.Reihe: sphere response to the order-n flat-layered solution (spectral amplitudes hd, hu of
        the e^{gamma z} and e^{-gamma z} parts at z = 0).  Returns (dhd, dhu, dB) on the full grid."""
        g = self.g
        z3 = np.zeros_like(hd); zB = np.zeros_like(hd[0])
        if self._w != w:
            self.frequenz(w)
        if not self.aktiv:
            return z3, z3.copy(), zB
        km, st, N, k = self.km, self.km.st, self.N, self.k
        msk, lam = self.msk, self.lam
        EX, EY = g.EX[msk], g.EY[msk]
        # incident scalars at z = 0: TE H_z (down part), TM J_z = i kappa a_perp
        inc = {'TE': hd[2][msk] * self.ein, 'TM': 1j * lam * (-EY * hd[0][msk] + EX * hd[1][msk]) * self.ein}
        Pinc = {}
        for q in ('TE', 'TM'):
            Rup0, Rdn = self.Rg0[q]
            Pinc[q] = st.propagate(q, lam, w, 0.0, np.zeros_like(lam, dtype=complex), inc[q], km.zc, Rup0, Rdn)
        sol = {}
        for m in self.mlist:
            ATEu, BTEu, ATMu, BTMu = AB_regular_scaled(N, m, 1j * self.u_s, k, lam)
            ATEd, BTEd, ATMd, BTMd = AB_regular_scaled(N, m, -1j * self.u_s, k, lam)
            rows = {'TE': ((ATEu[1:], BTEu[1:]), (ATEd[1:], BTEd[1:])), 'TM': ((ATMu[1:], BTMu[1:]), (ATMd[1:], BTMd[1:]))}
            ch = np.zeros(2 * N, complex)
            for q in ('TE', 'TM'):
                Pu, Pd = Pinc[q]
                (Au, Bu), (Ad, Bd) = rows[q]
                vu = Pu * self.ex[m] / self.cq[q]; vd = Pd * self.ex[m] / self.cq[q]
                ch[:N] += Au @ vu + Ad @ vd
                ch[N:] += Bu @ vu + Bd @ vd
            sol[m] = self._loese(m, ch)
        P0 = self._abstrahlung(sol, lam, w)
        S = {q: [np.zeros_like(lam, dtype=complex), np.zeros_like(lam, dtype=complex)] for q in ('TE', 'TM')}
        for m, res in P0.items():
            fac = np.exp(1j * m * self.beta) / (2 * np.pi * (1j) ** m * lam) * self.aus
            for q in ('TE', 'TM'):
                S[q][0] += res[q][0] * fac; S[q][1] += res[q][1] * fac            # up, down parts at z = 0
        dhd = np.zeros_like(hd); dhu = np.zeros_like(hd); dB = np.zeros_like(hd[0])
        gm = gam[msk]
        for part, arr, sgn in ((1, dhd, 1.0), (0, dhu, -1.0)):
            a_par = 1j * S['TE'][part] / lam
            a_perp = -1j * S['TM'][part] / lam
            arr[0][msk] = -EY * a_perp + sgn * gm * EX * a_par
            arr[1][msk] = EX * a_perp + sgn * gm * EY * a_par
            arr[2][msk] = -1j * lam * a_par
        dB[msk] = (S['TE'][0] + S['TE'][1]) / lam
        return dhd, dhu, dB

    # ------------------------------------------------------------------ order 0 (exact, non-periodic)
    def oberflaeche(self, w, quellen, x, y):
        """Sphere response to the sources at surface points (x, y) (earth side): dict Ex, Ey, Hx, Hy, Hz (1-D arrays),
        from radial tables per azimuthal order (Gauss-Legendre in lambda, cubic splines in rho)."""
        from scipy.interpolate import CubicSpline
        km, st = self.km, self.km.st
        x = np.asarray(x, float); y = np.asarray(y, float)
        out = {c: np.zeros(len(x), complex) for c in ('Ex', 'Ey', 'Hx', 'Hy', 'Hz')}
        sol_all, (lam_ls, wl_ls, k, u, R, cache) = km.loesen(w, quellen)
        if not sol_all:
            return out
        self.k = k
        sol = {m: v[0] for m, v in sol_all.items()}
        dX, dY = x - km.xc, y - km.yc
        rho_p = np.hypot(dX, dY); phi_p = np.arctan2(dY, dX)
        rmax = rho_p.max() * 1.01 + 1.0
        L = 40.0 / (km.zc - km.kugel.b) if km.zc > km.kugel.b else 1.0
        L = min(L, 45.0 / km.zc + 40.0 / km.zc)
        npan = int(2 * L * rmax / np.pi) + 20
        key = (round(L, 12), npan, round(rmax, 6))
        if getattr(self, '_tab', None) is None or self._tab[0] != key:   # frequency-independent Bessel table
            lam, wl = lam_grid(L, npan, self.nq)
            self._tab = (key, lam, wl, np.linspace(0.0, rmax, self.n_rho), {})
        _, lam, wl, rt, Jc = self._tab
        P0 = self._abstrahlung(sol, lam, w)
        u1 = st.u(1, lam, w); s1 = st.sigma[1]; L2 = lam ** 2

        def J(nu):
            a = abs(nu)
            if a not in Jc:
                Jc[a] = jv(a, lam[:, None] * rt[None, :])
            return Jc[a] * ((-1) ** a if nu < 0 else 1)
        for m, res in P0.items():
            Jm1, Jm, Jp1 = J(m - 1), J(m), J(m + 1)
            wJ = wl[:, None] * Jm; wdJ = wl[:, None] * 0.5 * (Jm1 - Jp1); wmJ = wl[:, None] * 0.5 * lam[:, None] * (Jm1 + Jp1)
            comp = {c: np.zeros(len(rt), complex) for c in ('Er', 'Ep', 'Hr', 'Hp', 'Hz')}
            for q in ('TE', 'TM'):
                Pu, Pd = res[q]; P = Pu + Pd; dP = u1 * (Pd - Pu)
                if q == 'TE':
                    comp['Er'] += ((1j * w * MU0 / L2) * 1j * P) @ wmJ; comp['Ep'] += -((1j * w * MU0 / L2) * lam * P) @ wdJ
                    comp['Hr'] += (lam * dP / L2) @ wdJ; comp['Hp'] += (1j * dP / L2) @ wmJ; comp['Hz'] += P @ wJ
                else:
                    comp['Hr'] += (1j * P / L2) @ wmJ; comp['Hp'] += -(lam * P / L2) @ wdJ
                    comp['Er'] += (lam * dP / L2) @ wdJ / s1; comp['Ep'] += (1j * dP / L2) @ wmJ / s1
            e = np.exp(1j * m * phi_p); cp, sp = np.cos(phi_p), np.sin(phi_p)
            val = {c: (CubicSpline(rt, v.real)(rho_p) + 1j * CubicSpline(rt, v.imag)(rho_p)) * e for c, v in comp.items()}
            out['Ex'] += val['Er'] * cp - val['Ep'] * sp; out['Ey'] += val['Er'] * sp + val['Ep'] * cp
            out['Hx'] += val['Hr'] * cp - val['Hp'] * sp; out['Hy'] += val['Hr'] * sp + val['Hp'] * cp
            out['Hz'] += val['Hz']
        return out


class KugelGelaendeHankel(KugelGelaende):
    """Sphere coupling for the Hankel series (huegel_hankel.py, KH-030): sphere on the hill axis, so the azimutal
    orders of hill and sphere coincide and every order m couples separately.  The spectral scalar c(lambda) of an
    order (physical s_m(r) = int c J_m(lambda r) lambda dlambda) is the cylindrical-wave density P = c lambda of
    kugel.Modell; the lambda nodes are those of the Hankel grid."""

    def __init__(self, km, hg):
        assert abs(km.xc) < 1e-9 and abs(km.yc) < 1e-9, "Hankel series: sphere must lie on the hill axis"
        assert km.zc - km.kugel.b > 0, "the sphere must lie entirely below the surface z = 0"
        self.km, self.g = km, hg
        self.N = km.N
        self.lam = hg.lam; self.wl = hg.wl
        self.mlist = [0] + [s * m for m in range(1, self.N + 1) for s in (1, -1)]
        self._w = None

    def antwort_m(self, w, m, hd, hu, u):
        """hd, hu (3, n_lam) in (par, perp, z) of order m -> (dhd, dhu, dB)."""
        z3 = np.zeros_like(hd); zB = np.zeros_like(hd[0])
        if abs(m) > self.N:
            return z3, z3.copy(), zB
        if self._w != w:
            self.frequenz(w)
        if not self.aktiv:
            return z3, z3.copy(), zB
        km, st, N, k = self.km, self.km.st, self.N, self.k
        lam, wl = self.lam, self.wl
        inc = {'TE': hd[2], 'TM': 1j * lam * hd[1]}                   # H_z and J_z of the down-going part
        ch = np.zeros(2 * N, complex)
        ATEu, BTEu, ATMu, BTMu = AB_regular_scaled(N, m, 1j * self.u_s, k, lam)
        ATEd, BTEd, ATMd, BTMd = AB_regular_scaled(N, m, -1j * self.u_s, k, lam)
        rows = {'TE': ((ATEu[1:], BTEu[1:]), (ATEd[1:], BTEd[1:])), 'TM': ((ATMu[1:], BTMu[1:]), (ATMd[1:], BTMd[1:]))}
        for q in ('TE', 'TM'):
            Rup0, Rdn = self.Rg0[q]
            Pu, Pd = st.propagate(q, lam, w, 0.0, np.zeros_like(lam, dtype=complex), inc[q], km.zc, Rup0, Rdn)
            (Au, Bu), (Ad, Bd) = rows[q]
            vu = Pu * lam * wl / self.cq[q]; vd = Pd * lam * wl / self.cq[q]
            ch[:N] += Au @ vu + Ad @ vd
            ch[N:] += Bu @ vu + Bd @ vd
        s_hat = self._loese(m, ch)
        res = self._abstrahlung({m: s_hat}, lam, w)[m]
        dhd = np.zeros_like(hd); dhu = np.zeros_like(hd)
        for part, arr, sgn in ((1, dhd, 1.0), (0, dhu, -1.0)):
            cte = res['TE'][part] / lam; ctm = res['TM'][part] / lam      # density P -> spectral scalar c
            a_par = 1j * cte / lam; a_perp = -1j * ctm / lam
            arr[0] = sgn * u * a_par; arr[1] = a_perp; arr[2] = -1j * lam * a_par
        dB = (res['TE'][0] + res['TE'][1]) / lam / lam
        return dhd, dhu, dB
