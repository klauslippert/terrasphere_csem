#!/bin/bash
# empymod run of this folder: in/empymod_input.toml -> out/transients.npz, out/transients.csv, out/log.txt.
# Start:  ./run.sh   (PYTHON = Python with empymod, default python3; WORKERS = parallel processes, default 20)
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
"$PYTHON" ../../../../run_empymod_time.py --workers "${WORKERS:-20}" .
