#!/bin/bash
# empymod run of this folder: in/empymod_input.toml -> out/fields.npz, out/fields.csv, out/log.txt.
# Start:  ./run.sh   (PYTHON = Python with empymod, default python3)
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
"$PYTHON" ../../../../run_empymod.py .
