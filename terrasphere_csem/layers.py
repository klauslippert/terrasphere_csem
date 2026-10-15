# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klaus Lippert
"""Layered earth as a scalar propagator for the two plane-wave families.

Global frame: depth zeta (positive DOWN), surface zeta = 0, interfaces at depths[0] = 0 < depths[1] < ...;
layer 0 = air (sigma 0), layer j = 1..J has conductivity sigma[j-1]; layer J is the basement.
The scalars are H_z (TE_z family, 'TE') and J_z = sigma E_z (TM_z family, 'TM'), both continuous across
interfaces; their z-derivatives satisfy  zeta_j d/dz(phi)  continuous with  zeta_j = 1 (TE), 1/sigma_j (TM).
Hence the Fresnel coefficient for a wave in layer i meeting layer j is
    r_ij = (zeta_i u_i - zeta_j u_j) / (zeta_i u_i + zeta_j u_j),      u_j = sqrt(lambda^2 - k_j^2), Re u > 0,
with r = -1 for TM at the air (insulator: J_z = 0).  Everything is vectorised over lambda.
Amplitude convention: a wave 'up' travels towards the surface (smaller zeta) and decays as e^{-u (zeta_ref - zeta)},
a wave 'down' as e^{-u (zeta - zeta_ref)}.  The sphere frame has z = zeta_c - zeta (z UP), so d/dz = -d/dzeta."""
import numpy as np
from . import MU0


class LayerStack:
    """Horizontally layered earth: air (layer 0, conductivity sigma_air) above the ground layers 1..J.

    depths: depths of the interfaces [m] (first = 0 = surface), sigma: conductivities of the ground layers [S/m]
    (the last layer is the basement). Provides the layer lookup, the reflection coefficients of the TE/TM plane-wave
    families and the propagation of mode amplitudes between depths."""
    def __init__(self, depths, sigma, sigma_air=0.0):
        """depths: interface depths (first = 0), sigma: conductivities of the ground layers; sigma_air: air
        conductivity (0 = insulating air; set equal to sigma[0] to emulate a full space)."""
        depths = np.atleast_1d(np.asarray(depths, float)); sigma = np.atleast_1d(np.asarray(sigma, float))
        assert depths[0] == 0.0 and len(sigma) == len(depths) and np.all(np.diff(depths) > 0)
        self.depths = depths; self.sigma_g = sigma
        self.J = len(sigma)                        # ground layers 1..J
        self.sigma = np.concatenate([[sigma_air], sigma])  # index = layer number (0 = air)
        self.d = np.diff(depths)                   # thickness of layers 1..J-1 (index j-1)

    def layer_of(self, zeta):
        """Layer number containing depth zeta (zeta < 0 -> 0; on an interface -> layer below)."""
        return int(np.searchsorted(self.depths, zeta, side='right'))

    def top(self, j):
        """Depth of the top interface of layer j [m] (-inf for the air)."""
        return self.depths[j - 1] if j >= 1 else -np.inf

    def bottom(self, j):
        """Depth of the bottom interface of layer j [m] (+inf for the basement)."""
        return self.depths[j] if j < self.J else np.inf

    def thickness(self, j):
        """Thickness of layer j [m] (inf for the air and the basement)."""
        return self.d[j - 1] if 1 <= j < self.J else np.inf

    def u(self, j, lam, w):
        """Vertical decay constant u_j = sqrt(lam^2 - k_j^2), Re u > 0, with k_j^2 = i w mu0 sigma_j, for horizontal
        wavenumbers lam [1/m] at the angular frequency w [rad/s]."""
        k2 = 1j * w * MU0 * self.sigma[j]
        u = np.sqrt(np.asarray(lam, complex) ** 2 - k2)
        return np.where(np.real(u) < 0, -u, u)

    def zeta_imp(self, j, q):
        """Impedance factor zeta_j of layer j for the family q: 1 for 'TE', 1/sigma_j for 'TM' (inf in an insulator)."""
        return 1.0 if q == 'TE' else (np.inf if self.sigma[j] == 0 else 1.0 / self.sigma[j])

    def fresnel(self, i, j, q, lam, w):
        """r_ij for a wave in layer i incident on layer j (adjacent)."""
        ui, uj = self.u(i, lam, w), self.u(j, lam, w)
        zi, zj = self.zeta_imp(i, q), self.zeta_imp(j, q)
        if np.isinf(zj):
            return -np.ones_like(ui)
        if np.isinf(zi):
            return np.ones_like(ui)
        return (zi * ui - zj * uj) / (zi * ui + zj * uj)

    def reflections(self, q, lam, w):
        """R_up[j]: reflection seen from inside layer j looking UP at its top interface (all layers above);
        R_dn[j]: looking DOWN at its bottom interface (all layers below). Arrays indexed by layer, each over lam."""
        lam = np.asarray(lam, complex); J = self.J
        Rup = [None] * (J + 1); Rdn = [None] * (J + 1)
        Rup[0] = np.zeros_like(lam)
        if J >= 1:
            Rup[1] = self.fresnel(1, 0, q, lam, w)
        for j in range(2, J + 1):
            e = np.exp(-2 * self.u(j - 1, lam, w) * self.thickness(j - 1)) if np.isfinite(self.thickness(j - 1)) else np.zeros_like(lam)
            r = self.fresnel(j, j - 1, q, lam, w)
            Rup[j] = (r + Rup[j - 1] * e) / (1 + r * Rup[j - 1] * e)
        Rdn[J] = np.zeros_like(lam)
        for j in range(J - 1, -1, -1):
            e = np.exp(-2 * self.u(j + 1, lam, w) * self.thickness(j + 1)) if np.isfinite(self.thickness(j + 1)) else np.zeros_like(lam)
            r = self.fresnel(j, j + 1, q, lam, w)
            Rdn[j] = (r + Rdn[j + 1] * e) / (1 + r * Rdn[j + 1] * e)
        return Rup, Rdn

    def propagate(self, q, lam, w, zeta_s, Aup, Adn, zeta_o, Rup=None, Rdn=None):
        """Scalar field of a source at depth zeta_s with direct amplitudes Aup (wave towards the surface, valid
        for zeta < zeta_s) and Adn (towards depth), evaluated at depth zeta_o (may be in any layer).
        Returns (Pu, Pd): the up- and down-going parts at zeta_o, phi = Pu + Pd, d phi/d zeta = u (Pu - Pd)."""
        lam = np.asarray(lam, complex)
        if Rup is None:
            Rup, Rdn = self.reflections(q, lam, w)
        s = self.layer_of(zeta_s); o = self.layer_of(zeta_o)
        us = self.u(s, lam, w)
        ht = zeta_s - self.top(s); hb = self.bottom(s) - zeta_s
        Ru = Rup[s] * np.exp(-2 * us * ht) if np.isfinite(ht) else np.zeros_like(lam)
        Rd = Rdn[s] * np.exp(-2 * us * hb) if np.isfinite(hb) else np.zeros_like(lam)
        den = 1 - Ru * Rd
        U = (Aup + Rd * Adn) / den; D = (Adn + Ru * Aup) / den     # total up/down amplitudes at zeta_s
        if o == s:
            if zeta_o == zeta_s:                    # on the source plane: mean of the two one-sided limits of the
                Pu = 0.5 * Aup + Rd * D             # direct wave (value and z-derivative), plus the reflections
                Pd = 0.5 * Adn + Ru * U
                return Pu, Pd
            if zeta_o < zeta_s:
                Pu = U * np.exp(-us * (zeta_s - zeta_o))
                Pd = Rup[s] * U * np.exp(-us * ht) * np.exp(-us * (zeta_o - self.top(s))) if np.isfinite(ht) else np.zeros_like(lam)
            else:
                Pd = D * np.exp(-us * (zeta_o - zeta_s))
                Pu = Rdn[s] * D * np.exp(-us * hb) * np.exp(-us * (self.bottom(s) - zeta_o)) if np.isfinite(hb) else np.zeros_like(lam)
            return Pu, Pd
        if o < s:                                   # transmit upwards layer by layer
            I = U * np.exp(-us * ht)                # up-going amplitude at the top interface of s
            j = s - 1
            while True:
                uj = self.u(j, lam, w); dj = self.thickness(j)
                rho_j = Rup[j] * np.exp(-2 * uj * dj) if np.isfinite(dj) else np.zeros_like(lam)
                Uj = I * (1 + Rup[j + 1]) / (1 + rho_j)          # up-going amplitude at the bottom of layer j
                if j == o:
                    zb = self.bottom(j)
                    Pu = Uj * np.exp(-uj * (zb - zeta_o))
                    Pd = Uj * Rup[j] * np.exp(-uj * dj) * np.exp(-uj * (zeta_o - self.top(j))) if np.isfinite(dj) else np.zeros_like(lam)
                    return Pu, Pd
                I = Uj * np.exp(-uj * dj); j -= 1
        else:                                       # transmit downwards
            I = D * np.exp(-us * hb)
            j = s + 1
            while True:
                uj = self.u(j, lam, w); dj = self.thickness(j)
                rho_j = Rdn[j] * np.exp(-2 * uj * dj) if np.isfinite(dj) else np.zeros_like(lam)
                Dj = I * (1 + Rdn[j - 1]) / (1 + rho_j)          # down-going amplitude at the top of layer j
                if j == o:
                    zt = self.top(j)
                    Pd = Dj * np.exp(-uj * (zeta_o - zt))
                    Pu = Dj * Rdn[j] * np.exp(-uj * dj) * np.exp(-uj * (self.bottom(j) - zeta_o)) if np.isfinite(dj) else np.zeros_like(lam)
                    return Pu, Pd
                I = Dj * np.exp(-uj * dj); j += 1
