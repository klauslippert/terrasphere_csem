# Flat model against empymod

The flat model (no terrain, no sphere) of the four elfe3D validation cases (`../../3D/elfe3d_topo`: grounded wire in line and
broadside, offset loop, large loop; three layers with interfaces at 200 m and 600 m; 0.1 Hz, 10 Hz, 1 kHz),
computed with this package and with the 1-D code **empymod**.

- empymod: https://github.com/emsig/empymod (used here: version see `empymod/out/log.txt`)
- Werthmüller, D. (2017). An open-source full 3D electromagnetic modeler for 1D VTI media in Python: empymod.
  *Geophysics*, 82(6), WB9-WB19. https://doi.org/10.1190/geo2016-0626.1

## Layout

One folder per case, `cases/<case>/`:

- `control.toml`: input of this package (as `../../3D/elfe3d_topo/cases/<case>/control.toml`, but `heights = [0.0]`);
  results in `results/` (`python3 -m terrasphere_csem control.toml`).
- `empymod/in/empymod_input.toml`: input of empymod (layers, frequencies, sources as finite bipoles, receivers,
  Hankel filter), written by `tests/validation/1D/empymod_input.py <case folder>`; `empymod/run.sh` runs
  `tests/validation/1D/run_empymod.py`, results in `empymod/out/` (`fields.npz`, `fields.csv` in the convention of this package:
  z up, e^{-iwt}).
- `comparison/empymod_out.txt`: deviations (`tests/validation/1D/compare_empymod.py <case folder>`).

empymod settings: quasi-static (no displacement currents, as this package), sources and receivers 1 mm below the
surface (exactly on z = 0 empymod puts them into the air, at 1 um its H is NaN; the 1 mm changes the fields by about
2e-5), digital filter key_201_2009 (the default key_401_2009 is 10-40 % off for the small inductive E of a closed loop
at 0.1 Hz, where the galvanic parts of the four sides cancel), 40 Gauss points per bipole.

## Usage

    PYTHON=/path/to/python ./run_all.sh

Report `report/report.pdf` (per case and profile: setup and fields, series against empymod; table of deviations):
`python3 tests/validation/1D/report_empymod.py tests/validation/1D/empymod`, then compile `report/report.tex` (pdflatex or
tectonic).
