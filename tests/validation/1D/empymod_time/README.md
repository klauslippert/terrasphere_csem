# Flat model against empymod in the time domain

The flat models of `../empymod` (four sources: grounded wire in line and broadside, offset loop, large loop; three
layers with interfaces at 200 m and 600 m) in the time domain: switch-off of 1 A, 10 µs to 10 s (the 101 times of the
lagged grid of this package), 5 stations per profile (x and y, -600 to 600 m every 300 m), components E_x, E_y and
dB/dt x, y, z.  E_z vanishes on the earth side of the flat surface and is not compared.

- empymod: https://github.com/emsig/empymod (version: see `cases/<case>/empymod/out/log.txt`)
- Werthmüller, D. (2017). An open-source full 3D electromagnetic modeler for 1D VTI media in Python: empymod.
  *Geophysics*, 82(6), WB9-WB19. https://doi.org/10.1190/geo2016-0626.1

## Layout

One folder per case, `cases/<case>/`:

- `control.toml`: input of this package (block `[time]`, heights 0: only the flat field is computed, no terrain
  series); results in `results/transients.npz` (`python3 -m terrasphere_csem control.toml`).
- `empymod/in/empymod_input.toml`: input of empymod, written by `tests/validation/1D/empymod_input.py <case folder>`;
  results in `empymod/out/transients.npz` (`tests/validation/1D/run_empymod_time.py`, convention of this package:
  z up, E in V/m, dB/dt in T/s per 1 A step); `empymod/out/parts/` holds one file per job.
- `comparison/empymod_out.txt`: deviations (`tests/validation/1D/compare_empymod_time.py <case folder>`).

empymod settings (the most accurate without interpolation): quasi-static (no displacement currents, as this
package); Hankel transform the default digital filter key_201_2009 in its standard form; Fourier transform the
digital filter key_601_2009 in its standard form (`pts_per_dec = 0`: every time its own frequencies, no
interpolation); sources and receivers 1e-6 m below the surface, every bipole as an explicit sum of 40 Gauss-Legendre
point dipoles (at 1e-6 m empymod 2.6.0 returns NaN for magnetic receivers of a finite bipole, not of point dipoles).

Deviation measures (`compare_empymod_time.py`): per receiver max_t |dV| / max_t |V_empymod| (relative to the maximum
of the curve), and pointwise |dV| / |V_empymod| where |V_empymod| exceeds 1e-6 of its maximum.  Receivers where a
component vanishes by symmetry (below 1e-6 of the largest E resp. dB/dt component) are left out of both and reported
with their level.

Report: `python3 tests/validation/1D/report_empymod_time.py tests/validation/1D/empymod_time`, then compile
`report/report.tex` (tectonic or pdflatex).

## Usage

    PYTHON=/path/to/python WORKERS=20 ./run_all.sh

The empymod part runs as 850 single-threaded jobs (bipole x component x block of 6 times) on `WORKERS` processes;
an interrupted run continues with the missing jobs when started again.
