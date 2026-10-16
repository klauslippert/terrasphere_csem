"""Sanity checks of the control file (terrasphere_csem/checks.py): a valid model gives no error, and every check
fires on a model built to violate it.  The main program stops on errors unless -f is given."""
import os, sys, subprocess, tempfile, tomllib
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from terrasphere_csem.checks import check

BASE = """
name = "check"
[terrain]
width = 150.0
heights = [0.0, 30.0, -30.0]
[layers]
depths = [0.0, 300.0]
conductivity = [0.01, 0.003]
[sphere]
radius = 60.0
conductivity = 1.0
center = [0.0, 0.0, 140.0]
degree = 10
[[source]]
type = "wire"
start = [-500.0, -1500.0]
end = [500.0, -1500.0]
[receivers]
half_length = 600.0
spacing = 20.0
[frequencies]
list = [3.0, 30.0, 300.0]
[numerics]
method = "fft"
"""


def _run(text, replace=()):
    """Write the control file (BASE with replacements) to a temporary folder and return its issues."""
    for a, b in replace:
        assert a in text, a
        text = text.replace(a, b)
    tomllib.loads(text)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'control.toml'), 'w') as fh:
            fh.write(text)
        return check(tmp)


def _has(issues, level, word):
    """True if an issue of the level contains the word."""
    return any(i.level == level and word in i.message for i in issues)


def test_valid_model():
    """The coupling model of the paper (sphere below a 30 m hill) passes without errors and warnings."""
    iss = _run(BASE)
    assert not iss, [str(i) for i in iss]


def test_errors():
    """Each error check fires on a model built to violate it."""
    E = lambda rep, word: _has(_run(BASE, rep), 'error', word)
    assert E((('method = "fft"', 'method = "fft"\nbox_corection = false'),), "unknown key 'box_corection'")
    assert E((('degree = 10', 'degree = 10\nradius_m = 5'),), "unknown key 'radius_m' in [sphere]")
    assert E((('end = [500.0, -1500.0]', 'end = [500.0, -1500.0]\nlenght = 3'),), "unknown key 'lenght'")
    assert E((('type = "wire"', 'type = "cable"'),), "unknown type")
    assert E((('conductivity = [0.01, 0.003]', 'conductivity = [0.01, -0.003]'),), "conductivities > 0")
    assert E((('[frequencies]\nlist = [3.0, 30.0, 300.0]', '[time]\ntmin = 1.0\ntmax = 0.1'),), "tmin < tmax")
    assert E((('depths = [0.0, 300.0]', 'depths = [0.0, 20.0]'),), "above the bottom of the depression")
    assert E((('center = [0.0, 0.0, 140.0]', 'center = [0.0, 0.0, 270.0]'),), "crosses or touches a layer")
    assert E((('center = [0.0, 0.0, 140.0]', 'center = [0.0, 0.0, 50.0]'),), "entirely below the flat ground")
    assert E((('heights = [0.0, 30.0, -30.0]', 'heights = [0.0, -120.0]'),
              ('center = [0.0, 0.0, 140.0]', 'center = [120.0, 0.0, 140.0]')), "cuts into the sphere")
    assert E((('method = "fft"', 'method = "hankel"'), ('center = [0.0, 0.0, 140.0]', 'center = [50.0, 0.0, 140.0]')),
             "needs the sphere on the hill axis")
    assert E((('start = [-500.0, -1500.0]', 'start = [-500.0, -900.0]'), ('end = [500.0, -1500.0]', 'end = [500.0, -900.0]')),
             "inside the window")
    assert E((('half_length = 600.0', 'half_length = 800.0'),), "beyond the exact part of the window")
    assert E((('spacing = 20.0', 'spacing = 25.0'),), "not a multiple of dx")
    assert E((('heights = [0.0, 30.0, -30.0]', 'heights = [0.0, 260.0]'), ('[sphere]', '[unused]'),
              ('radius = 60.0\nconductivity = 1.0\ncenter = [0.0, 0.0, 140.0]\ndegree = 10\n', '')),
             "unknown key 'unused'")
    iss = _run(BASE, (('[sphere]\nradius = 60.0\nconductivity = 1.0\ncenter = [0.0, 0.0, 140.0]\ndegree = 10\n', ''),
                      ('heights = [0.0, 30.0, -30.0]', 'heights = [0.0, 260.0]'), ('depths = [0.0, 300.0]', 'depths = [0.0]'),
                      ('conductivity = [0.01, 0.003]', 'conductivity = [0.01]')))
    assert _has(iss, 'error', 'beyond the convergence radius'), [str(i) for i in iss]


def test_warnings():
    """Each warning fires on a model built to approach a limit."""
    Wn = lambda rep, word: _has(_run(BASE, rep), 'warning', word)
    assert Wn((('heights = [0.0, 30.0, -30.0]', 'heights = [0.0, 60.0]'),), "converges poorly near the sphere")
    assert Wn((('heights = [0.0, 30.0, -30.0]', 'heights = [0.0, 100.0]'), ('[sphere]\nradius = 60.0\nconductivity = 1.0\n'
               'center = [0.0, 0.0, 140.0]\ndegree = 10\n', '')), "slope above 20 deg")
    assert Wn((('radius = 60.0', 'radius = 125.0'), ('degree = 10', 'degree = 4')), "degree 4 may be too low")
    assert Wn((('center = [0.0, 0.0, 140.0]', 'center = [0.0, 0.0, 235.0]'),), "within 5.0 m of a layer interface")
    assert Wn((('list = [3.0, 30.0, 300.0]', 'list = [3.0, 1e5]'),), "skin depth is not large against H")
    assert Wn((('list = [3.0, 30.0, 300.0]', 'list = [0.01]'), ('method = "fft"', 'method = "fft"\nbox_correction = false')),
              "no box correction")


def test_main_stops_on_errors():
    """python3 -m terrasphere_csem stops with exit code 1 on an error and names -f; nothing is computed."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'control.toml'), 'w') as fh:
            fh.write(BASE.replace('method = "fft"', 'method = "fft"\nbox_corection = false'))
        r = subprocess.run([sys.executable, '-m', 'terrasphere_csem', tmp], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 1 and '-f' in r.stderr and "unknown key 'box_corection'" in r.stdout, (r.stdout, r.stderr)
        assert not os.path.exists(os.path.join(tmp, 'results'))
