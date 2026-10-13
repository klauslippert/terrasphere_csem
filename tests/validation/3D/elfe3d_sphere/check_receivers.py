#!/usr/bin/env python3
"""Check before an elfe3D run that every receiver lies strictly inside one earth tetrahedron (barycentric
coordinates in (eps, 1-eps)), as elfe3D's receiver search requires.  Usage: python3 check_receivers.py <in-dir>
Exit code 1 if a receiver lies on a face/edge or outside the earth."""
import sys, os
import numpy as np
from scipy.spatial import cKDTree
d = sys.argv[1]
nod = np.loadtxt(os.path.join(d, 'model.1.node'), skiprows=1, comments='#')[:, 1:4]
ele = np.loadtxt(os.path.join(d, 'model.1.ele'), skiprows=1, comments='#').astype(np.int64)
earth = ele[ele[:, 5] == 2][:, 1:5] - 1
lines = open(os.path.join(d, 'elfe3D_input.txt')).read().split('\n')
i = [k for k, l in enumerate(lines) if l.startswith('num_rec')][0]
nr = int(lines[i].split()[1])
rx = np.array([[float(v) for v in lines[i + 1 + k].split()] for k in range(nr)])
tree = cKDTree(nod[earth].mean(1))
bad = 0
for k, p in enumerate(rx):
    _, cand = tree.query(p, k=60)
    best = -1.0
    for c in cand:
        P = nod[earth[c]]
        T = np.column_stack([P[1] - P[0], P[2] - P[0], P[3] - P[0]])
        l = np.linalg.solve(T, p - P[0])
        best = max(best, np.concatenate([[1 - l.sum()], l]).min())
    if best <= 1e-6:
        bad += 1
        print(f"receiver {k + 1} at {p}: min barycentric {best:.2e} -> NOT strictly inside")
print(f"{nr - bad}/{nr} receivers strictly inside an earth element")
sys.exit(1 if bad else 0)
