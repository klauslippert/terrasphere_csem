#!/usr/bin/env python3
"""Runs all test_* functions of all tests/test_*.py without pytest (which may not be installed)."""
import os, sys, time, importlib, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
mods = sorted(f[:-3] for f in os.listdir(HERE) if f.startswith('test_') and f.endswith('.py'))
if len(sys.argv) > 1:
    mods = [m for m in mods if m in sys.argv[1:]]
fails = 0
for mn in mods:
    mod = importlib.import_module(mn)
    for fn in [getattr(mod, a) for a in dir(mod) if a.startswith('test_') and callable(getattr(mod, a))]:
        t0 = time.time()
        try:
            fn(); print(f"PASS {mn}.{fn.__name__} ({time.time()-t0:.0f} s)")
        except Exception:
            fails += 1; print(f"FAIL {mn}.{fn.__name__}"); traceback.print_exc()
print(f"{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
