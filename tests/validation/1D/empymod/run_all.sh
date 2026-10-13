#!/bin/bash
# All flat-model runs: per case the series of this package (results/) and empymod (empymod/out/), then the
# comparison (comparison/empymod_out.txt).  PYTHON = Python with this package's requirements and empymod.
# Restrict with CASES="T1_wire_inline ...".  Progress: run_all.log.
set -u
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
CASES=${CASES:-$(cd cases && ls -d */ | tr -d /)}
LOG=$PWD/run_all.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S %Z')  $*" | tee -a "$LOG"; }
for c in $CASES; do
  ( say "start  $c"
    (cd cases/$c && "$PYTHON" ../../../../../../scripts/run_series.py . > series.out 2>&1) || say "FAILED series  $c"
    PYTHON=$PYTHON cases/$c/empymod/run.sh > cases/$c/empymod/run.out 2>&1 || say "FAILED empymod $c"
    "$PYTHON" ../compare_empymod.py cases/$c > /dev/null && say "done   $c" ) &
done
wait
