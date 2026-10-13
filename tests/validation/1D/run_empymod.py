#!/usr/bin/env python3
"""empymod run of one folder: reads <run>/in/empymod_input.toml (tests/validation/1D/empymod_input.py), writes <run>/out/:
  fields.npz   E, H (nf, 3, P) complex in the convention of this package (z up, e^{-iwt}, V/m and A/m per source
               current), freqs, px, py, profile
  fields.csv   the same as text
  log.txt      empymod version, settings, timing
Conversion (tests/verification/empymod/test_onedim.py): E = conj(E_emp) (1, 1, -1), H = -conj(H_emp) (1, 1, -1).
Usage: python3 run_empymod.py <run folder>"""
import os, sys, time, tomllib
import numpy as np
import empymod

COMPONENTS = [('E', 'x', False, 0.0, 0.0), ('E', 'y', False, 90.0, 0.0), ('E', 'z', False, 0.0, 90.0),
              ('H', 'x', True, 0.0, 0.0), ('H', 'y', True, 90.0, 0.0), ('H', 'z', True, 0.0, 90.0)]


def main():
    """Compute the fields of the run folder (see the module docstring)."""
    run = os.path.abspath(sys.argv[1])
    with open(os.path.join(run, 'in', 'empymod_input.toml'), 'rb') as fh:
        d = tomllib.load(fh)
    m, s, r, nu = d['model'], d['source'], d['receivers'], d['numerics']
    res, dep, fr = m['res'], m['depth'], np.array(d['frequencies']['list'])
    qs = {'epermH': [0.0] * len(res), 'epermV': [0.0] * len(res)} if m.get('quasi_static', True) else {}
    px, py = np.array(r['x']), np.array(r['y']); P = len(px)
    os.makedirs(os.path.join(run, 'out'), exist_ok=True)
    log = open(os.path.join(run, 'out', 'log.txt'), 'w')
    log.write(f'empymod {empymod.__version__}, {time.strftime("%Y-%m-%d %H:%M:%S")}\n{len(s["bipoles"])} bipoles, '
              f'{P} receivers, frequencies {list(fr)}, ht {nu["ht"]} {nu.get("dlf", "")}\n')
    t0 = time.time()
    F = np.zeros((len(fr), 6, P), complex)
    for b, I in zip(s['bipoles'], s['current']):
        src = list(b) + [s['depth'], s['depth']]
        for k, (_, _, mrec, az, dip) in enumerate(COMPONENTS):
            rec = [px, py, np.full(P, r['depth']), az, dip]
            v = empymod.bipole(src, rec, dep, res, fr, srcpts=s['srcpts'], mrec=mrec, strength=I, verb=0,
                               ht=nu['ht'], htarg={'dlf': nu['dlf']} if 'dlf' in nu else {}, **qs)
            F[:, k] += np.asarray(v).reshape(len(fr), P)
    F = np.conj(F); F[:, [2, 5]] *= -1; F[:, 3:] *= -1          # -> z up, e^{-iwt}, H sign of this package
    log.write(f'{time.time() - t0:.0f} s\n'); log.close()
    prof = np.array(r['profile'])
    np.savez_compressed(os.path.join(run, 'out', 'fields.npz'), E=F[:, :3], H=F[:, 3:], freqs=fr, px=px, py=py,
                        profile=prof, time_convention='exp(-iwt)', units='E V/m, H A/m')
    with open(os.path.join(run, 'out', 'fields.csv'), 'w') as fh:
        fh.write('f_Hz,profile,x_m,y_m,' + ','.join(f'{c[0]}{c[1]}_{p}' for c in COMPONENTS for p in ('re', 'im')) + '\n')
        for i, f in enumerate(fr):
            for j in range(P):
                fh.write(f'{f:g},{prof[j]},{px[j]:g},{py[j]:g},' +
                         ','.join(f'{v.real:.9e},{v.imag:.9e}' for v in F[i, :, j]) + '\n')
    print(os.path.join(run, 'out', 'fields.npz'))


if __name__ == '__main__':
    main()
