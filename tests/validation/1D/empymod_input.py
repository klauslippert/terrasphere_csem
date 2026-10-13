#!/usr/bin/env python3
"""empymod input of one computation folder (flat model): <folder>/empymod/in/empymod_input.toml with the layers,
frequencies, sources (finite bipoles: a wire, or the sides of a loop in the current direction) and receivers of
<folder>/control.toml, and <folder>/empymod/run.sh.  Run it with run_empymod.py (results in <folder>/empymod/out/).
Sources and receivers 1 mm below the surface: exactly on z = 0 empymod puts them into the air, and at 1 um its
magnetic field is NaN; the 1 mm changes the fields by ~2e-5.  Hankel transform: digital filter key_201_2009 (the
default key_401_2009 is not accurate enough for the small inductive E of a closed loop at 0.1 Hz: 10-40 %).
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
    px, py, prof = st.receivers()
    run = os.path.join(st.folder, 'empymod'); os.makedirs(os.path.join(run, 'in'), exist_ok=True)
    bip, cur = [], []
    for q, poly in zip(st.source_specs, source_polylines(st)):
        for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
            bip.append([x0, x1, y0, y1]); cur.append(float(q.get('current', 1.0)))
    lines = [f'# empymod input of {st.name}, flat model (written by tests/validation/1D/empymod_input.py from ../../control.toml).',
             '# empymod frame: z positive DOWN, time dependence e^{+iwt}; run_empymod.py converts the results to the',
             '# convention of this package (z up, e^{-iwt}).',
             '[model]',
             f'depth = {flist(st.layers.depths)}          # layer tops (m, down); air above z = 0',
             f'res = {flist([RES_AIR] + [1 / s for s in st.layers.sigma_g])}   # resistivity (Ohm m), air first',
             'quasi_static = true                  # epermH = epermV = 0: no displacement currents, as this package',
             '',
             '[frequencies]',
             f'list = {flist(st.freqs)}',
             '',
             '[source]',
             f'depth = {Z_SRC:g}                         # m below the surface',
             f'srcpts = {SRCPTS}                          # Gauss points per bipole',
             '# finite bipoles [x0, x1, y0, y1] (m), current from (x0, y0) to (x1, y1); a loop = its sides',
             'bipoles = [' + ',\n           '.join(flist(b) for b in bip) + ']',
             f'current = {flist(cur)}       # A, per bipole',
             '',
             '[receivers]',
             f'depth = {Z_RX:g}                         # m below the surface',
             f'x = {flist(px)}',
             f'y = {flist(py)}',
             'profile = [' + ', '.join(f'"{p}"' for p in prof) + ']',
             '',
             '[numerics]',
             'ht = "dlf"',
             f'dlf = "{DLF}"',
             '']
    open(os.path.join(run, 'in', 'empymod_input.toml'), 'w').write('\n'.join(lines))
    sh = os.path.join(run, 'run.sh'); open(sh, 'w').write(RUN_SH); os.chmod(sh, 0o755)
    print(os.path.join(run, 'in', 'empymod_input.toml'))


if __name__ == '__main__':
    main()
