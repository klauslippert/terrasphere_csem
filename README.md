# Semi-analytic solutions for a conductive sphere in a layered earth with Gaussian topography in 3D controlled-source electromagnetics

`terrasphere_csem` computes the frequency- and time-domain electromagnetic fields of grounded wires, dipoles and loops
over a layered earth with a Gaussian hill or valley and an optional conductive sphere below it (quasi-static, no
displacement currents).

## Installation

Python >= 3.11 with numpy and scipy:

    pip install .              # the package
    pip install .[test]        # with empymod, for the comparison tests in tests/verification/empymod/

## Usage

One control file in, the results out:

    python3 -m terrasphere_csem path/to/control.toml

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

## Tests

    python3 tests/run_tests.py                 # all tests (about 45 min)
    python3 tests/run_tests.py test_mie        # one test file, or a folder such as verification or empymod

- `tests/verification/`: the building blocks and complete fields against analytic solutions, independent reference
  implementations and, in `empymod/`, the 1-D code empymod.
- `tests/validation/`: comparison with independent codes for complete models, as case folders with control files,
  inputs, results and reports: `1D/empymod` (flat model against empymod), `3D/elfe3d_topo` (hills and valleys against
  the 3-D finite-element code elfe3D) and `3D/elfe3d_sphere` (sphere under the hill against elfe3D); overall report
  in `tests/validation/report/`.  See the READMEs there.

Third-party data and codes: see `CONTRIBUTIONS.md`.
