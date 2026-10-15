#!/usr/bin/env python3
"""empymod run in the time domain, in parallel: reads <run>/in/empymod_input.toml (block [time];
tests/validation/1D/empymod_input.py) of one or more run folders, writes per folder <run>/out/:
  transients.npz  t, E (nt, 3, P) in V/m and dBdt (nt, 3, P) in T/s for a 1 A step of the given signal, in the
                  convention of this package (z up); E_z is not computed (it vanishes on the earth side of the flat
                  surface) and stored as NaN; px, py, profile
  transients.csv  the same as text
  log.txt         empymod version, settings, number of jobs, timing
  parts/          one file per job (kept: an interrupted run continues where it stopped)
Jobs: every bipole x component (E_x, E_y, dB/dt x, y, z) x block of BLOCK times, each a single-threaded empymod call;
WORKERS processes work through all jobs of all folders.  Every bipole is an explicit Gauss-Legendre sum of point
dipoles (moment w_i L/2 I).  empymod loop = 'off' (offsets one after another, same result; all at once needs
> 10 GB per job).  E: empymod signal -1 (switch-off) or +1 (switch-on); dB/dt = +- mu0 x impulse response of
H (signal 0; switch-off = 1 - switch-on).  Frame (tests/verification/empymod/test_onedim.py): E = E_emp (1, 1, -1), B = -B_emp (1, 1, -1).
Usage: python3 run_empymod_time.py [--workers N] <run folder> [<run folder> ...]"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'                                           # one thread per process; parallel over jobs
import sys, time, tomllib, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import empymod

MU0 = 4e-7 * np.pi
BLOCK = 6                                                          # times per job
COMPONENTS = [('E', 'x', False, 0.0, 0.0), ('E', 'y', False, 90.0, 0.0),
              ('dBdt', 'x', True, 0.0, 0.0), ('dBdt', 'y', True, 90.0, 0.0), ('dBdt', 'z', True, 0.0, 90.0)]


def read(run):
    """The input of a run folder."""
    with open(os.path.join(run, 'in', 'empymod_input.toml'), 'rb') as fh:
        return tomllib.load(fh)


def jobs(run, d):
    """All jobs of a run folder: (run, bipole index, component index, block index)."""
    nb, nt = len(d['source']['bipoles']), len(d['time']['times'])
    return [(run, ib, k, j) for ib in range(nb) for k in range(len(COMPONENTS)) for j in range(-(-nt // BLOCK))]


def part_file(run, ib, k, j):
    """Result file of one job."""
    return os.path.join(run, 'out', 'parts', f'b{ib}_{COMPONENTS[k][0]}{COMPONENTS[k][1]}_t{j:03d}.npy')


def compute(job):
    """One job: the component k of bipole ib at the times of block j, in empymod's frame; saved to its part file."""
    run, ib, k, j = job
    d = read(run)
    m, s, r, nu, tm = d['model'], d['source'], d['receivers'], d['numerics'], d['time']
    t = np.array(tm['times'])[j * BLOCK:(j + 1) * BLOCK]
    x0, x1, y0, y1 = s['bipoles'][ib]; I = s['current'][ib]
    L = np.hypot(x1 - x0, y1 - y0); az = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    xg, wg = np.polynomial.legendre.leggauss(int(s['gauss']))
    xs, ys = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * xg, 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * xg
    field, _, mrec, raz, rdip = COMPONENTS[k]
    sgn = 1.0 if tm['signal'] == 'on' else -1.0
    P = len(r['x'])
    qs = {'epermH': [0.0] * len(m['res']), 'epermV': [0.0] * len(m['res'])} if m.get('quasi_static', True) else {}
    v = empymod.bipole(src=[xs, ys, np.full(len(xs), s['depth']), az, 0.0],
                       rec=[np.array(r['x']), np.array(r['y']), np.full(P, r['depth']), raz, rdip],
                       depth=m['depth'], res=m['res'], freqtime=t, signal=int(sgn) if field == 'E' else 0,
                       mrec=mrec, strength=0, verb=0, loop='off',          # one offset at a time: memory
                       ht=nu['ht'], htarg={'dlf': nu['dlf'], 'pts_per_dec': 0},
                       ft=nu['ft'], ftarg={'dlf': nu['ft_dlf'], 'pts_per_dec': nu['ft_pts_per_dec']}, **qs)
    v = np.asarray(v).reshape(len(t), P, len(xs)) @ (wg * 0.5 * L * I)  # Gauss sum over the point dipoles
    if field == 'dBdt':
        v = sgn * MU0 * v                                          # step response d/dt B = +- mu0 h_impulse
    f = part_file(run, ib, k, j); np.save(f + '.tmp.npy', v); os.replace(f + '.tmp.npy', f)
    return job


def merge(run, d, log):
    """Sum the parts of a run folder and write out/transients.npz and .csv in the convention of this package."""
    r, tm = d['receivers'], d['time']
    t = np.array(tm['times']); nt, P = len(t), len(r['x'])
    out = {'E': np.zeros((nt, 3, P)), 'dBdt': np.zeros((nt, 3, P))}
    for (_, ib, k, j) in jobs(run, d):
        field, comp = COMPONENTS[k][:2]
        out[field][j * BLOCK:(j + 1) * BLOCK, 'xyz'.index(comp)] += np.load(part_file(run, ib, k, j))
    out['E'][:, 2] = np.nan                                       # not computed (0 on the earth side)
    out['dBdt'][:, 2] *= -1; out['dBdt'] *= -1                     # z down -> up; axial vector: B = -B_emp (1, 1, -1)
    px, py, prof = np.array(r['x']), np.array(r['y']), np.array(r['profile'])
    np.savez_compressed(os.path.join(run, 'out', 'transients.npz'), t=t, E=out['E'], dBdt=out['dBdt'], px=px, py=py,
                        profile=prof, signal=tm['signal'], units='E V/m, dBdt T/s, per 1 A step')
    with open(os.path.join(run, 'out', 'transients.csv'), 'w') as fh:
        fh.write('t_s,profile,x_m,y_m,Ex,Ey,dBxdt,dBydt,dBzdt\n')
        for i in range(nt):
            for p in range(P):
                fh.write(f'{t[i]:.9e},{prof[p]},{px[p]:g},{py[p]:g},' + ','.join(
                    f'{a:.12e}' for a in (out['E'][i, 0, p], out['E'][i, 1, p], *out['dBdt'][i, :, p])) + '\n')
    log(f'{run}: written out/transients.npz, out/transients.csv')


def main():
    """Run all jobs of the given folders (see the module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    runs = [os.path.abspath(x) for x in a.runs]
    ins = {run: read(run) for run in runs}
    logs = {}
    for run in runs:
        os.makedirs(os.path.join(run, 'out', 'parts'), exist_ok=True)
        logs[run] = open(os.path.join(run, 'out', 'log.txt'), 'a')

    def log(s, run=None):
        """Print a line and append it to the log file of the run (or of all runs)."""
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {s}'
        print(line, flush=True)
        for rr in ([run] if run else runs):
            logs[rr].write(line + '\n'); logs[rr].flush()
    for run, d in ins.items():
        nu, s = d['numerics'], d['source']
        log(f'empymod {empymod.__version__}; {len(s["bipoles"])} bipoles x {s["gauss"]} point dipoles at '
            f'{s["depth"]} m, {len(d["receivers"]["x"])} receivers, {len(d["time"]["times"])} times '
            f'({d["time"]["times"][0]:.3g}..{d["time"]["times"][-1]:.3g} s), signal {d["time"]["signal"]}; Hankel '
            f'{nu["ht"]} {nu["dlf"]} standard, Fourier {nu["ft"]} {nu["ft_dlf"]} pts_per_dec {nu["ft_pts_per_dec"]}', run)
    todo = [jb for run in runs for jb in jobs(run, ins[run]) if not os.path.isfile(part_file(*jb))]
    total = sum(len(jobs(run, ins[run])) for run in runs)
    log(f'{total} jobs, {total - len(todo)} done before, {len(todo)} to run on {a.workers} processes')
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(compute, jb) for jb in todo]
        for n, fu in enumerate(as_completed(futs), 1):
            fu.result()
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                log(f'{n}/{len(todo)} jobs, {el / 60:.1f} min, remaining about {el / n * (len(todo) - n) / 60:.0f} min')
    for run in runs:
        merge(run, ins[run], lambda s_, rr=run: log(s_, rr))
    log(f'finished in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
