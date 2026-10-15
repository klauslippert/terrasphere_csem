# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klaus Lippert
"""Field components from the scalar modes.  For a mode  phi = [Pu + Pd] J_m(lam rho) e^{im phi}  with
d phi/dz = u (Pd - Pu)   (z UP; Pu is the part travelling up, Pd the part travelling down):
  TE (phi = H_z):  E = -(i w mu/lam^2) z^ x grad_h H_z,   H_h = (1/lam^2) grad_h d_z H_z
     E_rho = (i w mu/lam^2)(i m/rho) J_m e H,   E_phi = -(i w mu/lam^2) lam J_m' e H,   E_z = 0
     H_rho = (1/lam^2) lam J_m' e dH,           H_phi = (1/lam^2)(i m/rho) J_m e dH,     H_z = J_m e H
  TM (phi = J_z = sigma E_z):  H = -(1/lam^2) z^ x grad_h J_z,   E_h = (1/(sigma lam^2)) grad_h d_z J_z
     H_rho = (1/lam^2)(i m/rho) J_m e Jz,       H_phi = -(1/lam^2) lam J_m' e Jz,        H_z = 0
     E_rho = (1/(sigma lam^2)) lam J_m' e dJz,  E_phi = (1/(sigma lam^2))(i m/rho) J_m e dJz,  E_z = Jz/sigma
(e = e^{im phi}, ' = d/d(arg), dH = d H_z/dz).  The lambda integral is a weighted sum (Gauss-Legendre panels or a
digital Hankel filter)."""
import numpy as np
from scipy.special import jv, jvp
from . import MU0


def bessel_weights(m, lam, rho, wl=None, w01=None):
    """Quadrature weights including the Bessel factors for the three kernels that occur in the components:
    wJ ~ J_m(lam rho), wdJ ~ J_m'(lam rho) = (J_{m-1} - J_{m+1})/2, wmJ ~ (m/rho) J_m = (lam/2)(J_{m-1} + J_{m+1}).
    Either Gauss-Legendre weights wl (any m, needs rho >= 0) or digital-filter weights w01 = (w0, w1) for
    J_0 and J_1 (then |m| <= 1; J_2 from the recurrence; J_{-nu} = (-1)^nu J_nu)."""
    lam = np.asarray(lam, float)
    if w01 is None:
        Jm1 = jv(m - 1, lam * rho); Jp1 = jv(m + 1, lam * rho); J = jv(m, lam * rho)
        return wl * J, wl * 0.5 * (Jm1 - Jp1), wl * 0.5 * lam * (Jm1 + Jp1)
    w0, w1 = w01
    w2 = (2.0 / (lam * rho)) * w1 - w0
    W = {0: w0, 1: w1, -1: -w1, 2: w2, -2: w2}
    return W[m], 0.5 * (W[m - 1] - W[m + 1]), 0.5 * lam * (W[m - 1] + W[m + 1])


def mode_components(q, m, lam, wJ, wdJ, wmJ, Pu, Pd, u, sigma, w, phi):
    """E, H (cartesian, shape (3,)) at one point from the mode amplitudes Pu (up-going), Pd (down-going) of order m,
    arrays over lam, with the Bessel-weighted quadrature weights of bessel_weights; u: array over lam in the
    observation layer; sigma: its conductivity; phi: azimuth of the point about the mode axis."""
    lam = np.asarray(lam, float)
    e = np.exp(1j * m * phi)
    P = Pu + Pd; dP = u * (Pd - Pu)
    L2 = lam ** 2
    if q == 'TE':
        Er = np.sum((1j * w * MU0 / L2) * 1j * wmJ * P) * e
        Ep = -np.sum((1j * w * MU0 / L2) * lam * wdJ * P) * e
        Ez = 0.0
        Hr = np.sum(lam * wdJ * dP / L2) * e
        Hp = np.sum(1j * wmJ * dP / L2) * e
        Hz = np.sum(wJ * P) * e
    else:
        Hr = np.sum(1j * wmJ * P / L2) * e
        Hp = -np.sum(lam * wdJ * P / L2) * e
        Hz = 0.0
        Er = np.sum(lam * wdJ * dP / L2) * e / sigma
        Ep = np.sum(1j * wmJ * dP / L2) * e / sigma
        Ez = np.sum(wJ * P) * e / sigma
    cp, sp = np.cos(phi), np.sin(phi)
    E = np.array([Er * cp - Ep * sp, Er * sp + Ep * cp, Ez])
    H = np.array([Hr * cp - Hp * sp, Hr * sp + Hp * cp, Hz])
    return E, H


def dlf_grid(rho, filt=None):
    """Digital-filter abscissae and weights for int_0^inf f(lam) J_m(lam rho) ... as a weighted sum over lam.
    Returns lam and the two weight sets (w0 for J_0-type, w1 for J_1-type), such that
       int f(lam) J_nu(lam rho) dlam  ~=  sum_i f(lam_i) w_nu_i / (lam_i)  ... see dlf_integrate."""
    from . import filters as F
    filt = filt or F.key_401_2009()
    lam = filt.base / rho
    return lam, filt.j0 / rho, filt.j1 / rho


def dlf_integrate(f_lam, rho, nu, filt=None):
    """int_0^inf f(lam) J_nu(lam rho) lam dlam  via the digital filter: (1/rho) sum f(b_i/rho) w_i."""
    lam, w0, w1 = dlf_grid(rho, filt)
    return np.sum(f_lam(lam) * (w0 if nu == 0 else w1))
