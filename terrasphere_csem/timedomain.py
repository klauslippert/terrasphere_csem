# SPDX-License-Identifier: MIT
"""Time domain from the frequency domain (shared by sphere and hill).  With V(w) = Re V(w) + i Im V(w) (our e^{-iwt} convention) and the
direct-current value V(0) (zero for inductive responses, non-zero for the galvanic part of grounded-wire
responses, including the sphere's galvanic anomaly):
   switch-on step response    V_on(t)  = (2/pi) int_0^inf Re V(w) sin(w t)/w dw          (-> V(0) for t -> inf)
   switch-off                 V_off(t) = V(0) - V_on(t)
   impulse response           V_imp(t) = (2/pi) int_0^inf Re V(w) cos(w t) dw  = -d V_off/dt   (t > 0)
   linear ramp-off of length T V_ramp(t) = (1/T) int_t^{t+T} V_off(tau) dtau
V(0) is taken as Re V at the lowest frequency used (base[0]/t_max, far below any time constant of interest).

Two evaluations are provided:
  * `Transient` (default, exact): V is evaluated directly at the filter abscissae w = base_i/t.  On a lagged time
    grid t_k = t_max exp(-k d), d = ln(base[1]/base[0]) = 0.139, all times share abscissae, so K+1 times need only
    201 + K frequencies; arbitrary times are allowed (cost up to 201 per time, shared values are reused).
  * `Spectrum` (legacy): cubic spline of Re V over a logarithmic grid.  Its interpolation error dominates at late
    times (1e-2 at 1e-8 of the early value for the sphere response); kept for comparison only.
Both use the 201-point sine/cosine filter key_201_2012 (Key 2012, tables in filters/)."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from . import filters as _flt
_F = _flt.key_201_2012()


def omega_grid(t, nw=260):
    """Logarithmic grid of nw angular frequencies [rad/s] covering the filter abscissae of all times t."""
    t = np.atleast_1d(t)
    return np.geomspace(_F.base[0] / t.max(), _F.base[-1] / t.min(), nw)


class Spectrum:
    """Holds V(w) on a log grid (shape (nw, nrx)) and interpolates Re V in log w."""
    def __init__(self, w, V):
        """w: angular frequencies [rad/s] (increasing); V: complex spectrum, shape (nw, nrx) or (nw,)."""
        self.w = np.asarray(w, float); V = np.atleast_2d(np.asarray(V, complex))
        if V.shape[0] != len(self.w):
            V = V.T
        self.V = V
        self.sp_re = CubicSpline(np.log(self.w), np.real(V), axis=0)
        self.sp_im = CubicSpline(np.log(self.w), np.imag(V), axis=0)

    def re(self, ww):
        """Re V interpolated by cubic splines in log w at the angular frequencies ww (shape (len(ww), nrx)).

        Above the grid the end value is held; below it the value at the lowest grid frequency."""
        ww = np.asarray(ww, float)
        out = self.sp_re(np.log(np.clip(ww, self.w[0], self.w[-1])))
        # below the grid: Re V - V(0) -> 0 like w^2; above: keep the end value (integrand damped by 1/w)
        lo = ww < self.w[0]
        if np.any(lo):
            v0 = np.real(self.V[0])
            out[lo] = v0[None, :] + (self.sp_re(np.log(self.w[0])) - v0)[None, :] * (ww[lo] / self.w[0])[:, None] ** 2
        return out

    def V0(self):
        """Direct-current value: Re V at the lowest grid frequency."""
        return np.real(self.V[0])

    def switch_off(self, t):
        """V_off(t) = V(0) - V_on(t) = -(2/pi) int [Re V(w) - V(0)] sin(wt)/w dw: the direct-current part is removed
        before the filter (it would make the kernel ~ 1/w, which the digital filter integrates poorly)."""
        t = np.atleast_1d(t); res = np.zeros((len(t), self.V.shape[1])); v0 = self.V0()
        for i, ti in enumerate(t):
            ww = _F.base / ti
            res[i] = -(2 / np.pi) * np.sum((self.re(ww) - v0[None, :]) / ww[:, None] * _F.sin[:, None], axis=0) / ti
        return res

    def switch_on(self, t):
        """Switch-on step response V(0) - V_off(t) at the times t."""
        return self.V0()[None, :] - self.switch_off(t)

    def impulse(self, t):
        """Impulse response (2/pi) int Re V(w) cos(wt) dw at the times t."""
        t = np.atleast_1d(t); res = np.zeros((len(t), self.V.shape[1]))
        for i, ti in enumerate(t):
            ww = _F.base / ti
            res[i] = (2 / np.pi) * np.sum(self.re(ww) * _F.cos[:, None], axis=0) / ti
        return res

    def ramp_off(self, t, T, n=64):
        """Linear ramp-off of duration T (current decreasing linearly from I to 0 over [-T, 0])."""
        t = np.atleast_1d(t); res = np.zeros((len(t), self.V.shape[1]))
        x, wq = np.polynomial.legendre.leggauss(n)
        for i, ti in enumerate(t):
            tau = ti + 0.5 * (x + 1) * T
            res[i] = (0.5 * wq[:, None] * self.switch_off(tau)).sum(0)
        return res


DELTA = float(np.log(_F.base[1] / _F.base[0]))


def time_axis(tmin, tmax):
    """Lagged time grid t_k = tmax exp(-k DELTA) down to tmin (all times share filter abscissae)."""
    K = int(np.ceil(np.log(tmax / tmin) / DELTA))
    return tmax * np.exp(-DELTA * np.arange(K + 1))[::-1]


class Transient:
    """Exact filter evaluation: V(w) is computed at the union of the abscissae base_i/t for the given times.
    Vfun(w) -> array over receivers.  Methods return arrays (len(t), nrx) for the same times t."""
    def __init__(self, Vfun, t, verbose=False):
        """Vfun(w): spectrum (array over receivers) at the angular frequency w; t: times [s]; verbose: print progress.

        Vfun is called once for each distinct filter abscissa needed by the times t."""
        self.t = np.atleast_1d(np.asarray(t, float))
        keys = {}
        self.idx = []
        for ti in self.t:
            ww = _F.base / ti
            row = []
            for x in ww:
                key = float(f"{x:.12e}")
                if key not in keys:
                    keys[key] = len(keys)
                row.append(keys[key])
            self.idx.append(np.array(row))
        self.w = np.array(sorted(keys, key=keys.get))
        vals = []
        for i, wi in enumerate(self.w):
            vals.append(np.atleast_1d(Vfun(wi)))
            if verbose and i % 50 == 0:
                print(f"  w {wi:.3e} ({i+1}/{len(self.w)})", flush=True)
        self.V = np.array(vals)                         # (nw, nrx)
        self.v0 = np.real(self.V[int(np.argmin(self.w))])

    def V0(self):
        """Direct-current value Re V at the lowest angular frequency evaluated."""
        return self.v0

    def _sum(self, kern, sub_v0):
        """Apply the filter kernel `kern(re, ww, ti)` at every time; sub_v0 subtracts the DC value before filtering."""
        out = np.zeros((len(self.t), self.V.shape[1]))
        for k, ti in enumerate(self.t):
            ww = _F.base / ti; re = np.real(self.V[self.idx[k]]) - (self.v0[None, :] if sub_v0 else 0.0)
            out[k] = kern(re, ww, ti)
        return out

    def switch_off(self, t=None):
        """Switch-off step response V(0) - V_on(t) at the construction times; t, if given, must equal them."""
        self._check(t)
        return self._sum(lambda re, ww, ti: -(2 / np.pi) * np.sum(re / ww[:, None] * _F.sin[:, None], axis=0) / ti, True)

    def switch_on(self, t=None):
        """Switch-on step response V(0) - V_off(t) at the construction times."""
        return self.v0[None, :] - self.switch_off(t)

    def impulse(self, t=None):
        """Impulse response (2/pi) int Re V(w) cos(wt) dw at the construction times."""
        self._check(t)
        return self._sum(lambda re, ww, ti: (2 / np.pi) * np.sum(re * _F.cos[:, None], axis=0) / ti, False)

    def _check(self, t):
        """Assert that the requested times equal the construction times."""
        if t is not None:
            assert np.allclose(np.atleast_1d(t), self.t), "Transient evaluates only at the times it was built for"


def model_transient(model, sources, receivers, t, primary=False, verbose=False, nw=None):
    """Exact transient of a SphereModel at times t (scattered part by default; primary=True: total).
    (nw is ignored; kept for backward compatibility with the spline version.)"""
    f = (lambda wi: model.response(wi, sources, receivers, primary=True)) if primary else (lambda wi: model.scattered_field(wi, sources, receivers))
    return Transient(f, t, verbose)


def talbot(Ffun, t, M=24):
    """Inverse Laplace transform by the fixed Talbot method (Abate & Valko 2004):
       f(t) = (r/M) [ F(r) e^{r t}/2 + sum_{k=1}^{M-1} Re( e^{t s_k} F(s_k) (1 + i sigma_k) ) ],
       r = 2M/(5t), s_k = r theta_k (cot theta_k + i), sigma_k = theta_k + (theta_k cot theta_k - 1) cot theta_k,
       theta_k = k pi/M.  Ffun(s) may return an array (several receivers).  Our frequency response V(w) at
       complex w = i s IS the Laplace transform of the impulse response: F(s) = V(i s)."""
    t = np.atleast_1d(t)
    out = None
    for i, ti in enumerate(t):
        r = 2 * M / (5 * ti)
        acc = 0.5 * np.exp(r * ti) * np.asarray(Ffun(r + 0j))
        for kk in range(1, M):
            th = kk * np.pi / M; ct = 1 / np.tan(th)
            sk = r * th * (ct + 1j); sig = th + (th * ct - 1) * ct
            acc = acc + np.real(np.exp(ti * sk) * np.asarray(Ffun(sk)) * (1 + 1j * sig))
        acc = (r / M) * np.real(acc)
        if out is None:
            out = np.zeros((len(t),) + np.shape(acc))
        out[i] = acc
    return out


def switch_off_talbot(Vfun, t, M=24, V0=0.0):
    """Switch-off step response:  V_off(t) = V(0) - L^{-1}[V(i s)/s](t)  (V(0) = 0 for inductive responses)."""
    return V0 - talbot(lambda s: Vfun(1j * s) / s, t, M)


# ---------------------------------------------------------------------------------------------------------------
# Spectra given at explicit frequencies (used by the hill, whose coefficients are computed once for all filter
# abscissae and then summed for several heights).  Re V is the same in
# both time conventions, so these transforms do not depend on it.

def filter_omegas(t):
    """Sorted unique angular frequencies needed for times t (lagged grid -> shared)."""
    w = np.sort(np.concatenate([_F.base / ti for ti in np.atleast_1d(t)]))
    keep = np.concatenate([[True], np.diff(np.log(w)) > 1e-9])
    return w[keep]


def filter_frequencies(t):
    """Frequencies [Hz] of the sorted unique filter angular frequencies needed for the times t."""
    return filter_omegas(t) / (2 * np.pi)


def transient(t, freqs, V, V0=None, kind='off'):
    """V (nf, ...) complex spectrum at freqs [Hz] (must contain all abscissae for t).  Returns (nt, ...).
    kind: 'off' switch-off step, 'on' switch-on step, 'imp' impulse.  V0: DC value (default Re V at lowest f)."""
    w = 2 * np.pi * np.asarray(freqs)
    V = np.asarray(V); re = np.real(V)
    v0 = re[np.argmin(w)] if V0 is None else V0
    out = []
    for ti in np.atleast_1d(t):
        wi = _F.base / ti
        idx = np.array([np.argmin(np.abs(np.log(w / x))) for x in wi])
        assert np.allclose(w[idx], wi, rtol=1e-8), "missing filter abscissa"
        r = re[idx]
        sh = (-1,) + (1,) * (r.ndim - 1)
        if kind == 'imp':
            out.append((2 / np.pi) * np.sum(r * _F.cos.reshape(sh), axis=0) / ti)
        else:
            off = -(2 / np.pi) * np.sum((r - v0) / wi.reshape(sh) * _F.sin.reshape(sh), axis=0) / ti
            out.append(off if kind == 'off' else v0 - off)
    return np.array(out)
