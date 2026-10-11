"""Laplace-domain route to the time domain (fixed Talbot): (1) scattered response of the coaxial loop (1:100
design) against the sine-filter route; (2) primary half-space voltage against Lee's late-time series at 1-100 ms
(the open 'late time' problem of the filter route)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'referenz'))
from terrasphere_csem.schichten import Schichten
from terrasphere_csem.quellen import Dipol
from terrasphere_csem.kugel import Modell, Kugel, Empfaenger
from terrasphere_csem.zeit import ausschalten_talbot, spektrum_modell, talbot as talbot_fn


def test_talbot_primary_offset_loop():
    """Primary half-space voltage, Rx loop radius 0.7 a.  Laplace route: V = V_delta + i w M_free with the free-space
    mutual inductance M_free (coplanar coaxial loops, elliptic integrals) removed analytically, so that
    V_off(t) = L^{-1}[ -V_delta(i s)/s + M_free ]  (the bracket -> 0 for s -> inf: perfect-conductor limit).
    Against the sine-filter route and empymod."""
    from scipy.special import ellipk, ellipe, jv
    import empymod_schleife as ES
    from singh_mehrpol import step_off, k_of
    from terrasphere_csem import MU0
    from terrasphere_csem.spektral import lam_grid
    A, sh = 200.0, 0.01; Ar = 0.7 * A
    kk2 = 4 * A * Ar / (A + Ar) ** 2; kk = np.sqrt(kk2)
    M_free = MU0 * np.sqrt(A * Ar) * ((2 / kk - kk) * ellipk(kk2) - (2 / kk) * ellipe(kk2))
    lam, wl = lam_grid(400 * np.pi / A, 400, 24); JJ = jv(1, lam * A) * jv(1, lam * Ar)
    def V_delta(w):
        k = k_of(sh, w); u = np.sqrt(lam ** 2 - k * k + 0j); u = np.where(np.real(u) < 0, -u, u)
        return 2 * np.pi * Ar * 1j * w * MU0 * A * np.sum(wl * (lam / (lam + u) - 0.5) * JJ)
    print(f"M_free closed form {M_free:.6e} H;  numeric check 1j w M = V(w) - V_delta(w) at w=1e3: {(2*np.pi*Ar*1j*1e3*MU0*A*np.sum(wl*0.5*JJ))/(1j*1e3):.6e}")
    T = np.array([1e-3, 1e-2, 3.162e-2, 1e-1, 3e-1])
    res = {}
    for M in (16, 24, 32):
        res[M] = talbot_fn(lambda s: -V_delta(1j * s) / s + M_free, T, M=M)
    vfilt = step_off(lambda w: np.array([V_delta(wi) for wi in w]), T)
    vemp = ES.V_off_time(A, T, depth=[0], res=[2e14, 1 / sh], a_rx=Ar)
    worst = 0.0
    for i, t in enumerate(T):
        print(f"t = {t*1e3:6.1f} ms: Talbot M16/24/32 {res[16][i]: .6e} {res[24][i]: .6e} {res[32][i]: .6e}   filter {vfilt[i]: .6e}   empymod {vemp[i]: .6e}   Talbot24/emp {res[24][i]/vemp[i]:.6f}")
        if t <= 0.05:
            worst = max(worst, abs(res[24][i] / vemp[i] - 1))
    assert worst < 2e-3


def test_talbot_scattered_vs_filter():
    A, H, B, sh, ss = 200.0, 250.0, 180.0, 0.01, 1.0
    mod = Modell(Schichten([0.0], [sh]), Kugel(B, ss), (0.0, 0.0, H), nmax=15)
    tx = [Dipol('LOOP', (0.0, 0.0, 0.0), 1.0, radius=A)]; rx = [Empfaenger('LOOP', (0.0, 0.0, 0.0), radius=A)]
    T = np.array([1e-3, 1e-2, 3.162e-2, 1e-1])
    t0 = time.time()
    vt = ausschalten_talbot(lambda w: mod.streufeld(w, tx, rx), T, M=24)[:, 0]
    dt = time.time() - t0
    vt16 = ausschalten_talbot(lambda w: mod.streufeld(w, tx, rx), T, M=16)[:, 0]
    vt32 = ausschalten_talbot(lambda w: mod.streufeld(w, tx, rx), T, M=32)[:, 0]
    sp = spektrum_modell(mod, tx, rx, T); vf = sp.ausschalten(T)[:, 0]
    for i, t in enumerate(T):
        print(f"t = {t*1e3:6.1f} ms: Talbot M16/24/32 {vt16[i]: .6e} {vt[i]: .6e} {vt32[i]: .6e}   sine filter {vf[i]: .6e}   ratio24 {vt[i]/vf[i]:.6f}")
    print(f"(Talbot: {dt:.0f} s for {len(T)} times)")
    assert np.max(np.abs(vt / vf - 1)) < 1e-6           # exact filter evaluation (KSA-025)


if __name__ == '__main__':
    test_talbot_scattered_vs_filter(); test_talbot_primary_offset_loop()
