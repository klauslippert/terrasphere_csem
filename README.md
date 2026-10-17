# Semi-analytic reference solutions for 3D controlled-source electromagnetics: a conductive sphere beneath Gaussian topography in a layered earth

[![DOI](https://zenodo.org/badge/1414016660.svg)](https://doi.org/10.5281/zenodo.23434783)

`terrasphere_csem` computes the frequency- and time-domain electromagnetic fields of grounded wires, dipoles and loops
over a layered earth with a Gaussian hill or depression and an optional conductive sphere below it (quasi-static, no
displacement currents).  It is validated against [empymod](https://github.com/emsig/empymod) and
[elfe3D](https://github.com/emsig/elfe3d) (see `validation_report.pdf`).

## Installation

Python >= 3.11 with numpy and scipy:

    pip install .              # the package

## Usage

One control file in, the results out:

    python3 -m terrasphere_csem path/to/control.toml
    python3 -m terrasphere_csem --help          # usage and the description of the output files
    python3 -m terrasphere_csem -f path/to/control.toml   # compute even if the input checks report errors

Before every computation the control file is checked (`terrasphere_csem/checks.py`): unknown keys, ranges, the
layers, the position of the sphere (inside one layer, below the terrain, on the hill axis for `hankel`), sources and
receivers relative to the window of the expansion, slope, frequency limits and the FFT box.  Errors stop the program
(unless `-f`), warnings are printed; both are written to `log.txt`.

The results are written to `results/` next to the control file:

| file | content |
|---|---|
| `log.txt` | the control file, grids, timing, convergence, box correction |
| `fields.npz`, `fields.csv` | frequency domain: E (V/m) and H (A/m) at the receivers for every height and frequency, with the receiver positions and the estimated truncation error |
| `transients.npz`, `transients.csv` | time domain (block `[time]` instead of `[frequencies]`): E, dB/dt and B for a 1 A step |
| `coefficients.npz` | only on request (`formats = [..., "coefficients"]`): the series coefficients |

Which of npz and csv are written is set by `[output] formats`.  Time dependence e^{-iwt}, z points up.

Control files: `examples/control.toml` (a short example) and `examples/control_commented.toml` (all options
explained): terrain (width, heights), layers, optional sphere, sources, receiver profiles, frequencies or time window,
output and numerics.

Three commented examples, one folder each (`_f`: frequency domain, `_t`: time domain), run from the repository root:

    python3 -m terrasphere_csem examples/layered_f               # layered earth, flat ground, grounded wire
    python3 -m terrasphere_csem examples/layered_sphere_t        # sphere in a layered earth, flat ground, loop, transients
    python3 -m terrasphere_csem examples/layered_sphere_hill_f   # sphere below a 30 m hill and depression, grounded wire

The results are written to `results/` in each example folder.

## Tests

    python3 tests/run_tests.py                 # all tests (about 45 min)
    python3 tests/run_tests.py test_mie        # one test file, or a folder such as verification or empymod

- `tests/verification/`: the building blocks and complete fields against analytic solutions, independent reference
  implementations and, in `empymod/`, the 1-D code empymod.
- `tests/validation/`: comparison with independent codes for complete models, as case folders with control files,
  inputs, results and reports: `1D/empymod` (flat model against empymod), `3D/elfe3d_topo` (hills and depressions
  against the 3-D finite-element code elfe3D) and `3D/elfe3d_sphere` (sphere under the hill against elfe3D).  See the
  READMEs there.

The validation report is `validation_report.pdf` in the repository root: the conventions of the comparisons and the
four reports (flat model against empymod in the frequency and time domain, topography and sphere against elfe3D).
Only this PDF is tracked; the reports are written by scripts from the stored results.  To rebuild it:

    python3 tests/validation/1D/report_empymod.py tests/validation/1D/empymod
    python3 tests/validation/1D/report_empymod_time.py tests/validation/1D/empymod_time
    python3 tests/validation/3D/report_elfe3d.py tests/validation/3D/elfe3d_topo
    python3 tests/validation/3D/report_elfe3d_sphere.py tests/validation/3D/elfe3d_sphere
    # compile each <folder>/report/report.tex (tectonic or pdflatex), then the overall report:
    python3 tests/validation/report_overall.py
    (cd tests/validation/report && tectonic report.tex) && cp tests/validation/report/report.pdf validation_report.pdf

## Reproducing the paper figures

The scripts in `paper/` produce every figure and the elfe3D table of the paper.  Extra libraries:

    pip install -r requirements.txt -r paper/requirements.txt     # matplotlib, empymod, libdlf

Every script writes its PDF to the output directory given as its argument (here `out/`) and is run from the
repository root:

    mkdir -p out
    # figures computed directly by the package
    for f in model geometry multipole chain_time poles identities mie onedim reflection dc derivatives latetime \
             convergence terrain_dc; do
        python3 paper/fig_$f.py out
    done
    python3 paper/fig_nulltest.py out          # computes all four panels; --a .. --d recompute single panels,
                                               # --plot redraws from out/nulltest.npz
    # sphere below a hill: first the coefficients (production series), then the figure
    python3 paper/coupling.py out/coupling.npz
    python3 paper/fig_coupling.py out/coupling.npz out
    # comparisons with independent codes, from the results stored in tests/validation/
    python3 paper/fig_elfe_setup.py out
    python3 paper/fig_elfe_topo.py out
    python3 paper/fig_elfe_sphere.py out
    python3 paper/fig_empymod_time.py out
    python3 paper/tab_elfe.py out

`fig_convergence.py` and `fig_nulltest.py` cache their results in `out/` (`convergence.npz`, `nulltest.npz`) and
reuse them on a second run.  `coupling.py` runs the series on the production grid and is the longest step.  The
elfe3D and empymod figures read the stored results of `tests/validation/3D/` and `tests/validation/1D/empymod_time/`;
recomputing those is described in the READMEs there.  `cases.py` holds the model cases and `style.py` the plot style
shared by the scripts.

Third-party data and codes: see `CONTRIBUTIONS.md`.
