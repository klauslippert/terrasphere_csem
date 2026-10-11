"""One-dimensional (no sphere) fields of dipole sources in the layered earth, from the same spectral kernels
that feed the sphere coupling.  The lambda integral is done with Key's 401-point Hankel filter key_401_2009
(empymod filters; the 201-point filter limits the accuracy to ~1e-5 near the source), the modes are taken about each dipole's own axis (m = 0, +-1 only)."""
import numpy as np
from . import MU0
from .sources import dipole_modes
from .fields import bessel_weights, mode_components, dlf_grid


def primary_field(stack, dipole, points, w, filt=None, rho_min=1e-3):
    """E, H [V/m, A/m] at points [(x, y, zeta)] due to a list of Dipole objects, angular frequency w.
    Returns arrays E, H of shape (npts, 3), our e^{-iwt} convention."""
    points = np.atleast_2d(np.asarray(points, float))
    E = np.zeros((len(points), 3), complex); H = np.zeros((len(points), 3), complex)
    Rcache = {}
    for ip, (x, y, zo) in enumerate(points):
        o = stack.layer_of(zo); sig_o = stack.sigma[o]
        for dip in dipole:
            dx, dy = x - dip.pos[0], y - dip.pos[1]
            rho = max(np.hypot(dx, dy), rho_min); phi = np.arctan2(dy, dx)
            lam, w0, w1 = dlf_grid(rho, filt)
            s = stack.layer_of(dip.pos[2])
            k2s = 1j * w * MU0 * stack.sigma[s]; us = stack.u(s, lam, w); uo = stack.u(o, lam, w)
            key = (rho, 'TE'); 
            for q in ('TE', 'TM'):
                if q == 'TM' and sig_o == 0:
                    continue                       # TM part is not evaluated in insulating air
                Rup, Rdn = stack.reflections(q, lam, w)
                for m in (-1, 0, 1):
                    Ate_u, Ate_d, Atm_u, Atm_d = dipole_modes(dip, dip.pos[:2], lam, m, k2s, us)
                    Aup, Adn = (Ate_u, Ate_d) if q == 'TE' else (Atm_u, Atm_d)
                    if not np.any(Aup) and not np.any(Adn):
                        continue
                    Pu, Pd = stack.propagate(q, lam, w, dip.pos[2], Aup, Adn, zo, Rup, Rdn)
                    wJ, wdJ, wmJ = bessel_weights(m, lam, rho, w01=(w0, w1))
                    Ei, Hi = mode_components(q, m, lam, wJ, wdJ, wmJ, Pu, Pd, uo, sig_o, w, phi)
                    E[ip] += Ei; H[ip] += Hi
    return E, H


def loop_to_loop_emf(stack, tx, rx, w, npan=600, nq=24, L_fac=None):
    """Exact 1-D e.m.f. [V] in a horizontal circular receiver loop `rx` (Receiver 'LOOP') due to a horizontal
    circular transmitter loop `tx` (Dipole 'LOOP'), any lateral offset d, any depths (TE only):
       V = i w mu0 int_disk H_z dA,   H_z = int (I a/2)(lam^2/u_s) J1(lam a) [propagated] J0(lam rho') dlam,
       int_disk J0(lam |r - r_tx|) dA = (2 pi a_r/lam) J1(lam a_r) J0(lam d).
    Gauss-Legendre panels of width pi/max(a, a_r, d) up to L; the slowly converging free-space part of a coplanar
    pair is purely imaginary (inductive) and does not affect Re V (time domain)."""
    from scipy.special import jv
    from .sources import dipole_modes
    from .spectral import lam_grid
    a, ar = tx.radius, rx.radius
    d = np.hypot(rx.pos[0] - tx.pos[0], rx.pos[1] - tx.pos[1])
    L = L_fac / max(a, ar) if L_fac else npan * np.pi / max(a, ar, d)
    lam, wl = lam_grid(L, npan, nq)
    s = stack.layer_of(tx.pos[2]); us = stack.u(s, lam, w); k2s = 1j * w * MU0 * stack.sigma[s]
    A = dipole_modes(tx, tx.pos[:2], lam, 0, k2s, us)                  # m = 0 about its own axis: C0 = 1
    Pu, Pd = stack.propagate('TE', lam, w, tx.pos[2], A[0], A[1], rx.pos[2])
    disk = 2 * np.pi * ar / lam * jv(1, lam * ar) * jv(0, lam * d)
    return 1j * w * MU0 * np.sum(wl * (Pu + Pd) * disk)
