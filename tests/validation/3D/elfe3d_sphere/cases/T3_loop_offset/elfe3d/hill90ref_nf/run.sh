#!/bin/bash
# One elfe3D run in this folder: TetGen mesh from in/model.poly, receiver check, elfe3D, logs packed into logs.tar.gz.
# Start:  ./run.sh   (settings: ELFE3D, TETGEN = executables, default from PATH; THREADS = threads, default all cores)
# elfe3D reads in/ and writes out/ relative to this folder; it calls 'tetgen' itself during mesh refinement.
# Leftovers of an earlier attempt (refined meshes, out/, logs) are moved to in/old_<time>/, nothing is deleted.
# Exit code 0 = finished (elfe3D's final line "CPU time to run elfe3D" in the log), 1 = failed.
cd "$(dirname "$0")"
ELFE3D=${ELFE3D:-$(command -v elfe3d)}; TETGEN=${TETGEN:-$(command -v tetgen)}
[ -x "$ELFE3D" ] || { echo "elfe3D executable not found (set ELFE3D)"; exit 1; }
[ -x "$TETGEN" ] || { echo "TetGen executable not found (set TETGEN)"; exit 1; }
ELFE3D=$(readlink -f "$ELFE3D"); TETGEN=$(readlink -f "$TETGEN")
export PATH="$(dirname "$TETGEN"):$PATH"
THREADS=${THREADS:-$(nproc)}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
alt=in/old_$(date +%Y%m%d%H%M%S)
for x in in/model.[0-9]* in/tetgen.log out elfe3d.log time.log check.log logs.tar.gz; do
  [ -e "$x" ] && { mkdir -p "$alt"; mv "$x" "$alt"/; }
done
(cd in && "$TETGEN" -pq1.3kAaeni model.poly > tetgen.log 2>&1) || { echo "TetGen failed (in/tetgen.log)"; exit 1; }
CHECK=../../../../check_receivers.py                    # every receiver strictly inside an earth element
if [ -f "$CHECK" ] && python3 -c "import numpy, scipy" 2>/dev/null; then
  python3 "$CHECK" in > check.log 2>&1 || { echo "receiver check failed (check.log)"; exit 1; }
fi
mkdir -p out
TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"
$TIME "$ELFE3D" > elfe3d.log 2> time.log
tar czf logs.tar.gz elfe3d.log time.log in/tetgen.log $( [ -e check.log ] && echo check.log )
grep -q "CPU time to run elfe3D" elfe3d.log
