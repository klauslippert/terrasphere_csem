# Validation against elfe3D

Comparison runs of the 3D finite-element code **elfe3D** for the four validation cases of this package (grounded
wire in line and broadside, offset loop, large loop; hill and depression; 0.1 Hz, 10 Hz, 1 kHz).
The flat model is validated against empymod (`../../1D/empymod`) and has no elfe3D run here.
The folder is self-contained for the elfe3D side and can be copied to another machine.

## elfe3D

- Code: https://github.com/emsig/elfe3d (used here: v1.0.1 plus 16 commits, commit 58528bd7, 21 July 2026;
  Zenodo concept DOI [10.5281/zenodo.13309721](https://doi.org/10.5281/zenodo.13309721))
- Rulff, P. (2025). elfe3D v1.0.1: Modelling with the total electric field approach using finite elements in 3D.
  *Journal of Open Source Software*, 10(110), 7949. https://doi.org/10.21105/joss.07949
- Rulff, P., Buntin, L. M., Kalscheuer, T. (2021). Efficient goal-oriented mesh refinement in 3-D finite-element
  modelling adapted for controlled source electromagnetic surveys. *Geophysical Journal International*, 227,
  1624-1645. https://doi.org/10.1093/gji/ggab264

elfe3D needs the mesh generator TetGen (Si, H. (2015). TetGen, a Delaunay-based quality tetrahedral mesh generator.
*ACM Transactions on Mathematical Software*, 41(2), 11. https://doi.org/10.1145/2629697; used here: version 1.5.0)
and the direct solver MUMPS.

## Usage

One folder per run, `cases/<case>/elfe3d/<model>_<band>/`, each with its own `run.sh` (TetGen mesh, receiver check,
elfe3D, logs packed into `logs.tar.gz`).  A single run:

    cd cases/T1_wire_inline/elfe3d/hill30_hf1k
    ELFE3D=/path/to/elfe3d TETGEN=/path/to/tetgen THREADS=5 ./run.sh

All 16 runs, `NPAR` at the same time with `THREADS` threads each (default: cores / NPAR): set `ELFE3D`, `TETGEN`
and `NPAR` at the top of `run_all.sh`, then

    ./run_all.sh

Finished runs are skipped, so `run_all.sh` can be started again after an interruption; progress in `run_all.log`.
Restrict with `CASES="T1_wire_inline"`, `BANDS="nf"` or run names as arguments (`./run_all.sh hill30_nf`).

Memory: 10-16 GB per `nf` run and 35-63 GB per `hf1k` run (2.4-3.5 million unknowns, about 13-18 kB per unknown),
e.g. `NPAR=3` for `hf1k` and `NPAR=5` for `nf` on 20 cores / 188 GB.
Results per run: `out/electric_fields_receiver_line.txt`, `out/magnetic_fields_receiver_line.txt` (frequency, Re/Im of
the three components per receiver, receivers in the order of `in/elfe3D_input.txt`) and `logs.tar.gz`.

Series of this package (all heights of a case in one run, a few minutes per case), from the repository root:

    python3 -m terrasphere_csem tests/validation/3D/elfe3d_topo/cases/<case>/control.toml

giving `cases/<case>/results/` (`fields.npz`, `fields.csv`, `log.txt`).

## Details

| case | source (1 A) |
|---|---|
| `T1_wire_inline` | grounded wire along x, (-1750, 0) to (-1250, 0) |
| `T2_wire_broadside` | grounded wire along y, (-1500, -250) to (-1500, 250) |
| `T3_loop_offset` | square loop 400 m x 400 m, centre (-1500, 0) |
| `T4_loop_large` | square loop 2.6 km x 2.6 km around the hill |

All cases (`cases/<case>/control.toml`): layers 0.01 / 0.005 / 0.03 S/m, interfaces at 200 m and 600 m depth;
Gaussian terrain of width 150 m at the origin, height +30 m (`hill30`), -30 m (`depression30`); receivers on
the x- and y-profile from -600 m to 600 m every 20 m, 1 cm below the surface (122 receivers).

| band | frequencies | domain | mesh | per run |
|---|---|---|---|---|
| `nf` | 0.1 Hz and 10 Hz | +-15 km | fixed | 10-16 GB, 20-35 min (4 threads) |
| `hf1k` | 1 kHz | +-15 km | fixed, finer (about 25 m edges at the source) | 35-63 GB, 1-1.7 h (6-10 threads) |

Maximum tetrahedron volume (m^3) in the boxes of finer mesh around source and receivers (`in/model.poly`):

| band | air medium | air fine | layer 1 medium | layer 1 fine |
|---|---|---|---|---|
| `nf` | 300000 | 60000 | 100000 | 4000 |
| `hf1k` | 50000 | 6000 | 3000 | 1000 |

`hf1k` was first run with a coarser start mesh (300000 / 60000 / 20000 / 6000) and goal-oriented refinement by elfe3D (6 steps, up to 3
million unknowns); that was not accurate enough, because refinement does not repair a source region that is too
coarse.  The fixed finer mesh is used instead (`maxRefSteps 0`).  With refinement elfe3D gives a result for one
frequency per run only, hence a separate band for 1 kHz.

Run folder: `in/elfe3D_input.txt`, `in/regionparameters.txt`, `in/source.txt`, `in/model.poly` (TetGen input: terrain,
layer interfaces, boxes of finer mesh), `in/model.a.node`.  elfe3D has no command-line arguments; it reads `in/` in the
directory it is started in and writes `out/`; `run.sh` does exactly that.

Refined runs write one block of rows per refinement step; only the last block holds the result (here all runs are
unrefined, one block).  Conventions:
E_elfe3D = E, H_elfe3D = -H of this package (both e^{-iwt}).  Meshes (`in/model.<n>.*`, 20-30 GB for all runs) are
not kept.  Comparison with the series: `tests/validation/3D/compare_elfe3d.py <case folder>`, `tests/validation/3D/plot_profiles_freq.py
<case folder>`.  Regenerate the inputs: `python3 tests/validation/3D/elfe3d_input.py <case folder>` (settings per band in `BANDS`; it
reproduces the inputs here exactly).

Report (per case and profile: setup and fields of hill and depression, series against elfe3D), part of
`validation_report.pdf` in the repository root: `python3 tests/validation/3D/report_elfe3d.py tests/validation/3D/elfe3d_topo`,
then compile `report/report.tex` (pdflatex or tectonic).  The report folder is not tracked.
