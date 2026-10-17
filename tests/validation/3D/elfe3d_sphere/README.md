# Validation against elfe3D: sphere under a hill

elfe3D runs for the four validation cases of `../elfe3d_topo` with a sphere added: Gaussian hill of height +90 m and
width 150 m at the origin, layers 0.01 / 0.005 / 0.03 S/m with interfaces at 200 m and 600 m depth, and a
conductive sphere of 10 Ohm m (0.1 S/m), radius 180 m, centre (0, 0) at 400 m depth (in layer 2, under the hill).
Frequencies 0.1 Hz, 10 Hz (band `nf`) and 1 kHz (band `hf1k`).  elfe3D, TetGen and the run scripts: see
`../elfe3d_topo/README.md`.

Choice of the sphere (estimates with the series, anomaly relative to the profile maximum of the component):
this sphere gives 5-21 % at 0.1 Hz and 10 Hz for T1-T3 (about 1 % at 1 kHz; T4, the large loop centred on the
hill, sees almost nothing).  A sphere in layer 1 (r = 70 m, centre 110 m deep) gives larger anomalies, but under the
hill the terrain series does not converge for it (estimated truncation error 40-260 %); this sphere has < 5e-4.

## Degree of the sphere

`degree/check_degree.py` repeats the case `T1_wire_inline` with sphere degrees 12, 15 and 20 (without box correction)
and compares the sphere anomaly; result in `degree/degree.txt`: the change from degree 12 is at most 4e-7 of the
anomaly maximum (1.8e-6 for the weak E_z anomaly at 1 kHz on the y-profile), so degree 12 is converged.

## Runs

Per case `cases/<case>/elfe3d/`:

| run | model |
|---|---|
| `hill90_nf`, `hill90_hf1k` | hill with the sphere |
| `hill90ref_nf`, `hill90ref_hf1k` | the identical mesh, sphere region with the host resistivity (200 Ohm m) |

Written by `tests/validation/3D/elfe3d_input.py cases/<case>` and `tests/validation/3D/elfe3d_input.py cases/<case> --reference`; the
sphere is its own TetGen region (icosphere, level 3, scaled to the true volume) inside a finer box.  The difference
of the two runs is the elfe3D anomaly of the sphere, with the discretisation error of the background largely
cancelled.  Memory: about 15-18 GB per `nf` run and 55-75 GB per `hf1k` run (1.6 / 3.3 million edges).

    ELFE3D=/path/to/elfe3d TETGEN=/path/to/tetgen NPAR=2 THREADS=7 BANDS=hf1k ./run_all.sh
    ELFE3D=/path/to/elfe3d TETGEN=/path/to/tetgen NPAR=2 THREADS=3 BANDS=nf ./run_all.sh

## Series and comparison

`cases/<case>/results/` holds the series with the sphere (`control.toml`, heights [90]), from the repository root:

    python3 -m terrasphere_csem tests/validation/3D/elfe3d_sphere/cases/<case>/control.toml      # about 3 h per case, 7-13 GB

The series without the sphere is taken from `../elfe3d_topo/cases/<case>/results/`.  Comparison and report:

    python3 tests/validation/3D/report_elfe3d_sphere.py tests/validation/3D/elfe3d_sphere
    (cd tests/validation/3D/elfe3d_sphere/report && tectonic report.tex)

giving `cases/<case>/comparison/elfe3d_sphere.txt` (total field and anomaly per frequency and component) and
`report/report.pdf` (not tracked; part of `validation_report.pdf` in the repository root).
