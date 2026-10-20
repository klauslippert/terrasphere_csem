"""Receivers at arbitrary positions ([receivers] points, z on the terrain), frequency domain:
(1) control file and checks: valid points pass and come after the profiles, without profiles only the points are
    used; bad lists, points off the FFT grid (dx, and box_dx with the box correction) and beyond the window fail;
(2) Gaussian hill (H = 50 m), three layers: the axis receivers computed as profiles and the same positions given as
    points agree to rounding for both methods (fft, hankel), and the free points off the axes obey the mirror
    symmetry x -> -x of the survey (x-directed wire centred on x = 0)."""
import os, sys, subprocess, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, HERE]
from terrasphere_csem.config import Config
from test_checks import BASE, _run, _has

MODEL = """
[terrain]
width = 127.0
heights = [0.0, 50.0]
[layers]
depths = [0.0, 200.0, 600.0]
conductivity = [0.01, 0.005, 0.03]
[[source]]
type = "wire"
start = [-50.0, {y}]
end = [50.0, {y}]
current = 1.0
gauss = 96
[frequencies]
list = [10.0]
[output]
formats = ["npz"]
"""
AXIS = [(-400.0, 0.0), (-200.0, 0.0), (0.0, 0.0), (200.0, 0.0), (400.0, 0.0),
        (0.0, -400.0), (0.0, -200.0), (0.0, 0.0), (0.0, 200.0), (0.0, 400.0)]
OFF = [(200.0, 200.0), (-200.0, 200.0), (400.0, 200.0), (-400.0, 200.0), (200.0, -200.0), (-200.0, -200.0)]
NUMERICS = {'fft': ('method = "fft"\nn = 256\ndx = 10.0\nbox_correction = false', -1100.0),   # sources >= 1084 m
            'hankel': ('method = "hankel"', -1300.0)}                                    # sources > 1230 m


def test_points_config_and_checks():
    """Parsing and checks of [receivers] points."""
    pts = lambda p, prof=False: (('spacing = 20.0', 'spacing = 20.0\npoints = ' + p +
                                  ('\nprofiles = ["x", "y"]' if prof else '')),)
    err = lambda rep, word: _has(_run(BASE, rep), 'error', word)
    assert not _run(BASE, pts('[[200.0, 200.0], [-400.0, 200.0]]'))
    assert not _run(BASE, pts('[[200.0, 200.0]]', True) + (('method = "fft"', 'method = "hankel"'),))
    assert err(pts('[[200.0]]'), 'list of [x, y] pairs')
    assert err(pts('[[205.0, 0.0]]'), 'not on the FFT grid (dx = 10 m)')
    assert err(pts('[[210.0, 0.0]]'), 'not on the FFT grid (box_dx = 20 m)')
    assert not err(pts('[[210.0, 0.0]]') + (('method = "fft"', 'method = "fft"\nbox_correction = false'),), 'FFT grid')
    assert not err(pts('[[205.0, 3.0]]') + (('method = "fft"', 'method = "hankel"'),), 'FFT grid')
    assert err(pts('[[500.0, 500.0]]'), 'beyond the exact part of the window')
    assert err((('half_length = 600.0', 'half_length = 600.0\nprofiles = []'),), 'no receivers')
    for prof, n in ((True, 2 * 61 + 2), (False, 2)):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'control.toml'), 'w') as fh:
                fh.write(BASE.replace(*pts('[[200.0, 200.0], [-400.0, 200.0]]', prof)[0]))
            px, py, lab = Config(tmp).receivers()
        assert len(px) == n and lab[-2:] == ['p', 'p'], (len(px), lab[-3:])
        assert list(px[-2:]) == [200.0, -400.0] and list(py[-2:]) == [200.0, 200.0]


def _fields(tmp, name, method, receivers):
    """Run the main program on a control file with the hill model and return results/fields.npz."""
    num, y = NUMERICS[method]
    d = os.path.join(tmp, name); os.makedirs(d)
    with open(os.path.join(d, 'control.toml'), 'w') as fh:
        fh.write(f'name = "{name}"\n' + MODEL.format(y=y) + '[receivers]\n' + receivers + '\n[numerics]\n' + num + '\n')
    r = subprocess.run([sys.executable, '-m', 'terrasphere_csem', d], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return dict(np.load(os.path.join(d, 'results', 'fields.npz')))


def test_profiles_vs_points_hill():
    """Profiles and points at the same positions agree; free points obey the mirror symmetry (both methods)."""
    pts = ', '.join(f'[{x}, {y}]' for x, y in AXIS + OFF)
    # mirror x -> -x reverses the x-directed wire (J -> -J); E vector, H pseudovector: Ex, Hy, Hz even in x, Ey, Ez,
    # Hx odd
    sg = {'E': np.array([1, -1, -1]), 'H': np.array([-1, 1, 1])}
    with tempfile.TemporaryDirectory() as tmp:
        for method in NUMERICS:
            a = _fields(tmp, f'{method}_prof', method, 'profiles = ["x", "y"]\nhalf_length = 400.0\nspacing = 200.0')
            b = _fields(tmp, f'{method}_pts', method, f'points = [{pts}]')
            P = len(AXIS)
            assert np.allclose(a['px'], b['px'][:P]) and np.allclose(a['py'], b['py'][:P])
            assert np.allclose(a['pz'], b['pz'][:, :P]) and list(b['profile'][P:]) == ['p'] * len(OFF)
            ih = list(b['heights']).index(50.0)
            zt = 50.0 * np.exp(-(b['px'] ** 2 + b['py'] ** 2) / (2 * 127.0 ** 2))
            assert np.allclose(b['pz'][ih], zt, rtol=0, atol=1e-9)              # z on the terrain
            for F in ('E', 'H'):
                d = np.abs(a[F] - b[F][..., :P]).max() / np.abs(a[F]).max()
                worst = 0.0
                V = b[F][ih, 0]
                for k in range(P, len(b['px'])):
                    j = [i for i in range(P, len(b['px'])) if b['px'][i] == -b['px'][k] and b['py'][i] == b['py'][k]]
                    if j:
                        worst = max(worst, (np.abs(V[:, k] - sg[F] * V[:, j[0]]) / np.abs(V).max(axis=1)).max())
                print(f"{method:6s} {F}: profiles vs points {d:.1e}, mirror symmetry of the free points {worst:.1e}")
                assert d < 1e-10 and worst < 1e-9
            eff = np.abs(b['E'][ih, 0] - b['E'][0, 0])[:, P:].max() / np.abs(b['E'][0, 0]).max()
            assert eff > 1e-2                                                     # the hill acts at the free points
