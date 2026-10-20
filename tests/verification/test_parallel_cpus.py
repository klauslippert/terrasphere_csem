"""Frequency-parallel run ([numerics] cpus > 1, parallel.py): the main program with cpus = 3 gives the fields of
cpus = 1, for the fft series on a hill (to rounding: fewer threads per process change the summation order in the
matrix products and FFTs) and for the flat field (identical), and the log names the processes."""
import os, sys, subprocess, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CONTROL = """name = "cpus"
[terrain]
width = 127.0
heights = {heights}
[layers]
depths = [0.0, 500.0]
conductivity = [0.01, 0.1]
[[source]]
type = "wire"
start = [-200.0, -1100.0]
end = [200.0, -1100.0]
gauss = 9
[receivers]
points = [[-400.0, -400.0], [0.0, 0.0], [200.0, 200.0], [400.0, -200.0]]
[frequencies]
list = [0.3, 30.0, 3000.0]
[output]
formats = ["npz"]
[numerics]
method = "fft"
n = 256
dx = 10.0
orders = 8
box_correction = false
{cpus}
"""


def _run(tmp, heights, cpus):
    """Fields and log of the main program for the heights and number of processes."""
    d = os.path.join(tmp, f'h{len(heights)}_c{cpus}'); os.makedirs(d)   # cpus None: key omitted
    with open(os.path.join(d, 'control.toml'), 'w') as fh:
        fh.write(CONTROL.format(heights=heights, cpus='' if cpus is None else f'cpus = {cpus}'))
    r = subprocess.run([sys.executable, '-m', 'terrasphere_csem', d], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return dict(np.load(os.path.join(d, 'results', 'fields.npz'))), open(os.path.join(d, 'results', 'log.txt')).read()


def test_cpus_series_and_flat():
    """cpus = 3 against cpus = 1: fft series on a 50 m hill to rounding, flat field identical; without the key one
    process and a note in the log."""
    with tempfile.TemporaryDirectory() as tmp:
        a, log1 = _run(tmp, [0.0, 50.0], 1)
        b, log = _run(tmp, [0.0, 50.0], 3)
        c, log0 = _run(tmp, [0.0, 50.0], None)
        assert 'on 3 processes' in log and 'on 1 processes' not in log1, log
        assert 'cpus not given, using cpus = 1' in log0 and 'cpus not given' not in log1 + log
        assert np.array_equal(a['E'], c['E']) and np.array_equal(a['H'], c['H'])
        for F in ('E', 'H'):
            d = np.abs(a[F] - b[F]).max() / np.abs(a[F]).max()
            print(f"hill, {F}: max |cpus 1 - cpus 3| / max|F| = {d:.1e}")
            assert d < 1e-10
        a, _ = _run(tmp, [0.0], 1)
        b, _ = _run(tmp, [0.0], 3)
        assert np.array_equal(a['E'], b['E']) and np.array_equal(a['H'], b['H'])
