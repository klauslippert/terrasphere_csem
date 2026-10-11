#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Runs all test_* functions of all test_*.py files below tests/ without pytest (which may not be installed).

Collected: tests/test_*.py and test_*.py in the subfolders unit/, verification/ and validation/ (recursively);
tests/reference/ holds helper modules only.  Arguments select by test-file name (e.g. test_mie) or by folder
(unit, verification, validation, or a subfolder such as empymod); without arguments everything runs.
The tests in validation/empymod/ need empymod (pip install .[test])."""
import os, sys, time, importlib.util, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))


def collect():
    """(module name, path, folder parts relative to tests/) of every test file."""
    out = []
    for d, dirs, files in os.walk(HERE):
        dirs[:] = sorted(x for x in dirs if x not in ('reference', '__pycache__'))
        rel = os.path.relpath(d, HERE)
        parts = [] if rel == '.' else rel.split(os.sep)
        out += [(f[:-3], os.path.join(d, f), parts) for f in sorted(files) if f.startswith('test_') and f.endswith('.py')]
    return out


tests = collect()
if len(sys.argv) > 1:
    sel = set(sys.argv[1:])
    tests = [t for t in tests if t[0] in sel or sel & set(t[2])]
fails = 0
for mn, path, parts in tests:
    sys.path.insert(0, os.path.dirname(path))
    label = '/'.join(parts + [mn])
    try:
        spec = importlib.util.spec_from_file_location(mn, path); mod = importlib.util.module_from_spec(spec)
        sys.modules[mn] = mod; spec.loader.exec_module(mod)
    except Exception:
        fails += 1; print(f"FAIL {label} (import)"); traceback.print_exc(); continue
    for fn in [getattr(mod, a) for a in dir(mod) if a.startswith('test_') and callable(getattr(mod, a))]:
        t0 = time.time()
        try:
            fn(); print(f"PASS {label}.{fn.__name__} ({time.time()-t0:.0f} s)")
        except Exception:
            fails += 1; print(f"FAIL {label}.{fn.__name__}"); traceback.print_exc()
print(f"{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
