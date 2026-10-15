#!/bin/bash
# All time-domain flat-model runs: per case the series of this package (results/transients.npz), then empymod for all
# cases together on WORKERS processes (empymod/out/transients.npz), then the comparison (comparison/empymod_out.txt).
# PYTHON = Python with this package's requirements and empymod.  Restrict with CASES="T1_wire_inline ...".
# Progress: run_all.log; empymod progress also in cases/<case>/empymod/out/log.txt.
set -u
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
WORKERS=${WORKERS:-20}
CASES=${CASES:-$(cd cases && ls -d */ | tr -d /)}
LOG=$PWD/run_all.log
export PYTHONPATH="$(cd ../../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"     # the package (repository root)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say() { echo "$(date '+%Y-%m-%d %H:%M:%S %Z')  $*" | tee -a "$LOG"; }
for c in $CASES; do
  ( say "start  series  $c"
    (cd cases/$c && "$PYTHON" -m terrasphere_csem control.toml > series.out 2>&1) && say "done   series  $c" \
      || say "FAILED series  $c" ) &
done
wait
say "start  empymod $CASES"
"$PYTHON" ../run_empymod_time.py --workers "$WORKERS" $(for c in $CASES; do echo cases/$c/empymod; done) \
  > empymod.out 2>&1 && say "done   empymod" || say "FAILED empymod"
for c in $CASES; do
  "$PYTHON" ../compare_empymod_time.py cases/$c > /dev/null && say "done   compare $c" || say "FAILED compare $c"
done
