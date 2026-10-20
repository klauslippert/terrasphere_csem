"""Receivers at arbitrary positions ([receivers] points) against empymod, frequency domain: flat three-layer earth
(interfaces at 200 m and 600 m), x-directed grounded wire of 100 m at y = -1100 m, 29 free points (a 5 x 5 grid
of +-400 m and four points off any grid).  The main program (flat field: closed form + layer correction) against
empymod with the settings of tests/validation/1D/run_empymod.py (key_201_2009, 40 source points, source and
receivers 1 mm below the surface, quasi-static).  Measured 2026-10-20: at most 2e-5 of max|V| per component."""
import os, sys, subprocess, tempfile
import numpy as np
import empymod
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

DEPTHS, SIGMA = [0.0, 200.0, 600.0], [0.01, 0.005, 0.03]
FREQS = [0.1, 10.0, 1000.0]
POINTS = [(x, y) for x in (-400.0, -200.0, 0.0, 200.0, 400.0) for y in (-400.0, -200.0, 0.0, 200.0, 400.0)] + \
         [(137.5, -263.0), (-512.0, 311.0), (45.0, 610.0), (-333.0, -77.0)]
CONTROL = f"""name = "free_points_empymod"
[terrain]
width = 127.0
heights = [0.0]
[layers]
depths = {DEPTHS}
conductivity = {SIGMA}
[[source]]
type = "wire"
start = [-50.0, -1100.0]
end = [50.0, -1100.0]
current = 1.0
gauss = 96
[receivers]
points = [{', '.join(f'[{x}, {y}]' for x, y in POINTS)}]
[frequencies]
list = {FREQS}
[output]
formats = ["npz"]
[numerics]
method = "hankel"
"""


def test_free_points_vs_empymod():
    """E_x, E_y, H_x, H_y, H_z at 29 free points against empymod (E_z vanishes on the flat surface)."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'control.toml'), 'w') as fh:
            fh.write(CONTROL)
        r = subprocess.run([sys.executable, '-m', 'terrasphere_csem', tmp], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        s = np.load(os.path.join(tmp, 'results', 'fields.npz'))
        ours = np.concatenate([s['E'][0], s['H'][0]], axis=1)                  # (nf, 6, P)
    px, py = s['px'], s['py']
    assert np.allclose(px, [p[0] for p in POINTS]) and np.allclose(py, [p[1] for p in POINTS])
    assert list(s['profile']) == ['p'] * len(POINTS)
    z = 1e-3
    res = [2e14] + [1 / v for v in SIGMA]
    emp = np.zeros_like(ours)
    for k, (mrec, az, dip) in enumerate([(False, 0, 0), (False, 90, 0), (False, 0, 90),
                                          (True, 0, 0), (True, 90, 0), (True, 0, 90)]):
        v = empymod.bipole([-50.0, 50.0, -1100.0, -1100.0, z, z], [px, py, np.full(len(px), z), az, dip], DEPTHS,
                           res, FREQS, srcpts=40, mrec=mrec, strength=1.0, verb=0, ht='dlf',
                           htarg={'dlf': 'key_201_2009'}, epermH=[0.0] * 4, epermV=[0.0] * 4)
        emp[:, k] = np.asarray(v).reshape(len(FREQS), len(px))
    emp = np.conj(emp); emp[:, [2, 5]] *= -1; emp[:, 3:] *= -1                  # -> z up, e^{-iwt}, H sign of the package
    worst = 0.0
    for i, f in enumerate(FREQS):
        dev = {c: np.abs(ours[i, k] - emp[i, k]).max() / np.abs(emp[i, k]).max()
               for k, c in enumerate(['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']) if c != 'Ez'}
        print(f"f = {f:g} Hz: " + ', '.join(f"{c} {v:.1e}" for c, v in dev.items()))
        worst = max(worst, max(dev.values()))
    assert worst < 1e-4, worst
