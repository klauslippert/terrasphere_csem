# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klaus Lippert
"""Frequency-parallel evaluation for the main program.

The frequencies are independent of each other.  With [numerics] cpus = n > 1, n processes compute interleaved subsets
(f_i, f_{i+n}, f_{i+2n}, ...) of them, each process with the same code as the serial run (one grid, one sphere
coupling, its frequencies one after another) and cpu_count / n threads; the results are put back in the order of the
frequencies.  With cpus = 1 everything runs in the calling process as before.  Every process holds its own grid,
primary field and sphere tables, so the memory grows with n.

Kinds of work: 'fft' (series coefficients on the FFT grid, args = (n, dx, kc, nord, points)), 'hankel' (series
coefficients with the Hankel method, args = nord), 'flat' (flat-earth field at the receivers, args = (px, py))."""
import os
import numpy as np

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'TERRASPHERE_WORKERS')


def run(st, kind, freqs, args):
    """Result of `kind` for all frequencies, first axis = frequency, on [numerics] cpus processes."""
    freqs = np.atleast_1d(freqs)
    n = min(int(st.numerics.get('cpus', 1)), len(freqs))
    if n <= 1:
        return compute(st, kind, freqs, args)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    threads = str(max(1, (os.cpu_count() or n) // n))
    old = {k: os.environ.get(k) for k in THREAD_VARIABLES}
    os.environ.update({k: threads for k in THREAD_VARIABLES})          # inherited by the spawned processes
    try:
        with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context('spawn')) as ex:
            parts = list(ex.map(_worker, [(st.file, kind, freqs[i::n], args) for i in range(n)]))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    out = np.empty((len(freqs),) + parts[0].shape[1:], parts[0].dtype)
    for i, p in enumerate(parts):
        out[i::n] = p
    return out


def _worker(task):
    """One process: read the control file and compute its subset of the frequencies."""
    from .config import Config
    file, kind, freqs, args = task
    return compute(Config(file), kind, freqs, args)


def compute(st, kind, freqs, args):
    """Serial computation of `kind` for the frequencies freqs (see the module docstring)."""
    if kind == 'fft':
        from . import terrain
        n, dx, kc, nord, points = args
        g = st.grid(n=n, dx=dx)
        kg = st.sphere_coupling(g, kc=kc)                               # same low-pass as the series
        idx = g.point_index(*points) if points is not None else None
        CE, CH = terrain.coefficients(freqs, g, st.layers, nord=nord, kc=kc, sphere=kg, points=idx)
        return np.concatenate([CE, CH], axis=2)                          # (nf, nord+1, 6, 2, N) / (nf, nord+1, 6, P)
    if kind == 'hankel':
        from . import terrain_hankel
        g = st.hankel_grid()
        kg = st.sphere_coupling(g)
        CE, CH = terrain_hankel.coefficients(freqs, g, st.layers, nord=args, sphere=kg)
        return np.concatenate([CE, CH], axis=2)                          # (nf, nord+1, 6, P)
    if kind == 'flat':
        return flat_field(st, freqs, *args)
    raise ValueError(kind)


def flat_field(st, freqs, px, py):
    """Flat-earth field at the receivers for all frequencies: (nf, 6, P) (E_z = 0 on the flat surface)."""
    from . import halfspace
    q = st.sources()
    P = halfspace.surface_fields_layered(px, py, freqs, st.layers, q)
    kg = st.sphere_coupling(st.hankel_grid() if st.numerics['method'] == 'hankel' else st.grid()) if st.sphere else None
    out = np.zeros((len(freqs), 6, len(px)), complex)
    for k, c in [(0, 'Ex'), (1, 'Ey'), (3, 'Hx'), (4, 'Hy'), (5, 'Hz')]:
        out[:, k] = P[c]
    if kg is not None:
        for i, f in enumerate(freqs):
            pk = kg.surface_fields(2 * np.pi * f, q, px, py)
            for k, c in [(0, 'Ex'), (1, 'Ey'), (3, 'Hx'), (4, 'Hy'), (5, 'Hz')]:
                out[i, k] += pk[c]
    return out
