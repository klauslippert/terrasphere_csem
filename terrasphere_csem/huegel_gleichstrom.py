"""Independent direct-current check for the hill (GH-013): scalar potential with the same field expansion.

Earth z < h: Lap V = 0; surface: n.grad V = 0, i.e. d_z V - grad h . grad_h V = 0 at z = h (no current into the air);
primary potential of the HED-type sources on the flat ground (halbraum.potential_dc; for a wire the electrode pair).
V = V0 + sum_n H^n V_n,  V_n ~ e^{kappa z};  order n (F_j = f^j/j!):
   d_z V_n(0) = - sum_{j>=1} F_j d_z^{j+1} V_{n-j}(0) + sum_{j>=0} F_j grad f . grad_h d_z^j V_{n-1-j}(0)
Primary: d_z^{2m} V0 -> kappa^{2m} V0^, odd derivatives vanish (d_z V0 = 0 on the flat surface).
E = -grad V at the terrain, Taylor-continued like the H-formulation.  Same grid, window and low-pass as huegel.py.
Layered ground (KH-010): V_n ~ e^{kappa z} + rho e^{-kappa z} in the top layer, rho = -R_TM,dn(kappa, w=0) e^{-2 kappa d_1}
(potential reflection); then d_z V_n(0) = kappa (1 - rho) C_n and d_z^j V_n(0) = kappa^j (1 + (-1)^j rho) C_n.
"""
import numpy as np
import scipy.fft as sfft
from . import halbraum


def koeffizienten(g, boden, nord, kc=0.2):
    """CE (nord+1, 3, 2, N): DC electric-field coefficients on the two profiles of Gitter g (its sources)."""
    from .huegel import boden_schichten
    st = boden_schichten(boden)
    V0 = np.zeros_like(g.f)
    m = g.maske                                                      # only where the window is non-negligible
    V0[m] = halbraum.potential_dc_schichten(g.X[m], g.Y[m], st, g.quellen)
    if st.J > 1:
        kap = np.where(g.KAP > 0, g.KAP, 1e-12)                     # u = kappa at DC; kappa = 0 is the 0/0 limit
        rho = -st.reflections('TM', kap, 0.0)[1][1] * np.exp(-2 * kap * st.tiefen[1])
    else:
        rho = np.zeros_like(g.KAP)
    phi = np.exp(-(g.KAP / kc) ** 16)
    fft2 = lambda a: sfft.fft2(a, axes=(-2, -1), workers=-1)
    ifft2 = lambda a: sfft.ifft2(a, axes=(-2, -1), workers=-1)
    V0h = fft2(g.W * V0) * phi
    KX, KY, KAP = g.KX, g.KY, g.KAP
    fx, fy = -g.X / g.s ** 2 * g.f, -g.Y / g.s ** 2 * g.f          # grad f
    F = [np.ones_like(g.f)]
    for j in range(1, nord + 2):
        F.append(F[-1] * g.f / j)

    def dz(m, j):                                                    # spectral d_z^j V_m at z = 0
        if m == 0:
            return KAP ** j * V0h if j % 2 == 0 else np.zeros_like(V0h)
        return KAP ** j * (1 + (-1) ** j * rho) * C[m]

    C = [None]
    CE = np.zeros((nord + 1, 3, 2, g.n), complex)

    def efeld(spec_j):                                               # E = -grad V for d_z^j V (spectral), d_z^{j+1}
        s, s1 = spec_j
        return np.stack([-1j * KX * s, -1j * KY * s, -s1])

    CE[0] = g.linien(efeld((dz(0, 0), dz(0, 1))))
    for n in range(1, nord + 1):
        R = np.zeros((g.n, g.n), complex)
        for j in range(1, n + 1):
            R -= F[j] * ifft2(dz(n - j, j + 1))
        for j in range(0, n):
            s = dz(n - 1 - j, j)
            R += F[j] * (fx * ifft2(1j * KX * s) + fy * ifft2(1j * KY * s))
        Rh = fft2(R) * phi
        c = np.where(KAP > 0, Rh / np.where(KAP > 0, KAP * (1 - rho), 1.0), 0.0)
        C.append(c)
        acc = np.zeros((3, 2, g.n), complex)
        for j in range(0, n + 1):
            m = n - j
            fl = np.stack([F[j][:, g.i0], F[j][g.i0, :]])[None]
            acc += fl * g.linien(efeld((dz(m, j), dz(m, j + 1))))
        CE[n] = acc
    return CE
