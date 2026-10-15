#!/usr/bin/env python3
"""empymod input of one computation folder (flat model): <folder>/empymod/in/empymod_input.toml with the layers,
frequencies, sources (finite bipoles: a wire, or the sides of a loop in the current direction) and receivers of
<folder>/control.toml, and <folder>/empymod/run.sh.  Run it with run_empymod.py (results in <folder>/empymod/out/).
Sources and receivers 1 mm below the surface: exactly on z = 0 empymod puts them into the air, and at 1 um its
magnetic field is NaN; the 1 mm changes the fields by ~2e-5.  Hankel transform: digital filter key_201_2009 (the
default key_401_2009 is not accurate enough for the small inductive E of a closed loop at 0.1 Hz: 10-40 %).
Time domain (block [time] in control.toml): the times of the lagged grid of this package, the signal (switch-off or
on), the Fourier transform as digital filter key_601_2009 in its standard form (pts_per_dec = 0: every time its own
frequencies, no interpolation), Hankel transform the empymod default key_201_2009 (standard); sources and receivers
1e-6 m below the surface, every bipole as an explicit Gauss-Legendre sum of SRCPTS point dipoles (at 1e-6 m empymod
2.6.0 returns NaN for magnetic receivers of a finite bipole, not of point dipoles).
Usage: python3 empymod_input.py <folder>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.abspath(os.path.join(HERE, '..', '..', '..')), os.path.dirname(HERE), HERE]   # package, plotting.py, this folder
from terrasphere_csem.config import Config
from plotting import source_polylines

RES_AIR = 2e14
Z_SRC = Z_RX = 1e-3
SRCPTS = 40
DLF = 'key_201_2009'
Z_TIME = 1e-6                 # time domain: depth of sources and receivers
FT_DLF = 'key_601_2009'       # time domain: Fourier filter, standard (pts_per_dec = 0)

RUN_SH_TIME = """#!/bin/bash
# empymod run of this folder: in/empymod_input.toml -> out/transients.npz, out/transients.csv, out/log.txt.
# Start:  ./run.sh   (PYTHON = Python with empymod, default python3; WORKERS = parallel processes, default 20)
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
"$PYTHON" ../../../../run_empymod_time.py --workers "${WORKERS:-20}" .
"""

RUN_SH = """#!/bin/bash
# empymod run of this folder: in/empymod_input.toml -> out/fields.npz, out/fields.csv, out/log.txt.
# Start:  ./run.sh   (PYTHON = Python with empymod, default python3)
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
"$PYTHON" ../../../../run_empymod.py .
"""


def flist(v, fmt='%.10g'):
    """TOML list of numbers."""
    return '[' + ', '.join(fmt % x for x in v) + ']'


def main():
    """Write the empymod input of the folder (see the module docstring)."""
    st = Config(sys.argv[1])
    time_domain = st.domain == 'time'
    z_src = z_rx = Z_TIME if time_domain else Z_SRC
    px, py, prof = st.receivers()
    run = os.path.join(st.folder, 'empymod'); os.makedirs(os.path.join(run, 'in'), exist_ok=True)
    bip, cur = [], []
    for q, poly in zip(st.source_specs, source_polylines(st)):
        for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
            bip.append([x0, x1, y0, y1]); cur.append(float(q.get('current', 1.0)))
    if time_domain:
        block = ['[time]',
                 f'times = {flist(st.t)}   # s, the lagged grid of this package',
                 f'signal = "{st.signal}"                 # off = switch-off, on = switch-on of the current']
        src_lines = [f'gauss = {SRCPTS}                           # each bipole as an explicit Gauss-Legendre sum of point dipoles']
        num = ['[numerics]', 'ht = "dlf"', f'dlf = "{DLF}"                  # Hankel: empymod default, standard',
               'ft = "dlf"', f'ft_dlf = "{FT_DLF}"               # Fourier: standard, every time its own frequencies',
               'ft_pts_per_dec = 0']
    else:
        block = ['[frequencies]', f'list = {flist(st.freqs)}']
        src_lines = [f'srcpts = {SRCPTS}                          # Gauss points per bipole']
        num = ['[numerics]', 'ht = "dlf"', f'dlf = "{DLF}"']
    lines = [f'# empymod input of {st.name}, flat model (written by tests/validation/1D/empymod_input.py from ../../control.toml).',
             '# empymod frame: z positive DOWN, time dependence e^{+iwt}; the run script converts the results to the',
             '# convention of this package (z up, e^{-iwt}).',
             '[model]',
             f'depth = {flist(st.layers.depths)}          # layer tops (m, down); air above z = 0',
             f'res = {flist([RES_AIR] + [1 / s for s in st.layers.sigma_g])}   # resistivity (Ohm m), air first',
             'quasi_static = true                  # epermH = epermV = 0: no displacement currents, as this package',
             ''] + block + ['',
             '[source]',
             f'depth = {z_src:g}                         # m below the surface'] + src_lines + [
             '# finite bipoles [x0, x1, y0, y1] (m), current from (x0, y0) to (x1, y1); a loop = its sides',
             'bipoles = [' + ',\n           '.join(flist(b) for b in bip) + ']',
             f'current = {flist(cur)}       # A, per bipole',
             '',
             '[receivers]',
             f'depth = {z_rx:g}                         # m below the surface',
             f'x = {flist(px)}',
             f'y = {flist(py)}',
             'profile = [' + ', '.join(f'"{p}"' for p in prof) + ']',
             ''] + num + ['']
    open(os.path.join(run, 'in', 'empymod_input.toml'), 'w').write('\n'.join(lines))
    sh = os.path.join(run, 'run.sh'); open(sh, 'w').write(RUN_SH_TIME if time_domain else RUN_SH); os.chmod(sh, 0o755)
    print(os.path.join(run, 'in', 'empymod_input.toml'))

if __name__ == '__main__':
    main()
