#!/bin/bash
# Start all runs: every cases/<case>/elfe3d/<model>_<band>/run.sh, NPAR of them at the same time.
# Settings (block below): ELFE3D, TETGEN (executables), NPAR (runs at the same time, default 4),
#           THREADS (threads per run, default cores / NPAR), BANDS (default "hf100k hf1k nf": longest first),
#           CASES (default all).  Arguments (optional): run names, e.g.  ./run_all.sh flat_nf hill30_hf1k
# A run is skipped when its logs.tar.gz holds a finished elfe3D log; so the script can be started again.
# Progress: run_all.log.
set -u
cd "$(dirname "$0")"
# ---- settings: edit here (or set them on the command line, which takes precedence) ----
ELFE3D=${ELFE3D:-/pfad/elfe3d}                          # elfe3D executable
TETGEN=${TETGEN:-/pfad/tetgen}                          # TetGen executable
NPAR=${NPAR:-4}                                         # runs at the same time
# ----------------------------------------------------------------------------------------
export ELFE3D TETGEN
[ -x "$ELFE3D" ] || { echo "elfe3D executable not found: $ELFE3D (edit ELFE3D at the top of run_all.sh)"; exit 1; }
[ -x "$TETGEN" ] || { echo "TetGen executable not found: $TETGEN (edit TETGEN at the top of run_all.sh)"; exit 1; }
NCORE=$(nproc)
export THREADS=${THREADS:-$(( NCORE / NPAR > 0 ? NCORE / NPAR : 1 ))}
BANDS=${BANDS:-"hf100k hf1k nf"}
CASES=${CASES:-$(cd cases && ls -d */ | tr -d /)}
ONLY=" $* "
LOG=$PWD/run_all.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S %Z')  $*" | tee -a "$LOG"; }
finished() { tar xzOf "$1/logs.tar.gz" elfe3d.log 2>/dev/null | grep -q "CPU time to run elfe3D"; }

one() {
  local r=$1
  if finished "$r"; then say "skip (done)  $r"; return; fi
  say "start        $r"
  if "$r/run.sh" > "$r/run.out" 2>&1; then
    say "done         $r  $(grep -hE 'Elapsed|Maximum resident' "$r/time.log" 2>/dev/null | sed 's/^\s*//' | tr '\n' ' ')"
  else
    say "FAILED       $r  ($(tail -1 "$r/run.out"))"
  fi
}

RUNS=()
for band in $BANDS; do for c in $CASES; do for r in cases/$c/elfe3d/*_$band; do
  [ -x "$r/run.sh" ] || continue
  [ "$ONLY" = "  " ] || [[ "$ONLY" == *" $(basename "$r") "* ]] || continue
  RUNS+=("$r")
done; done; done
say "=== ${#RUNS[@]} runs, $NPAR at a time, $THREADS threads each ($NCORE cores)"
for r in "${RUNS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do wait -n; done
  one "$r" &
  sleep 5
done
wait
say "=== finished"
