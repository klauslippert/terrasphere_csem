# SPDX-License-Identifier: MIT
"""Digital linear filter coefficients of Kerry Key, shipped as plain-text tables in this folder.

The tables are third-party data (Copyright Kerry Key, CC BY 4.0, see CONTRIBUTIONS.md and the header of each .txt
file); only this loader is under the licence of the package.  Nothing is fetched from empymod at run time.

    key_201_2009()   201-point Hankel filter (J0, J1): attributes base, j0, j1   (Key 2009, Geophysics 74(2), F9-F20)
    key_401_2009()   401-point Hankel filter (J0, J1): attributes base, j0, j1   (Key 2009)
    key_201_2012()   201-point Fourier filter: attributes base, sin, cos          (Key 2012, Geophysics 77(3), F21-F30)

Use of a Hankel filter:  F(r) = sum_i f(base_i / r) * j_i / r;  of the Fourier filter:  sum_i f(base_i / t) * sin_i / t.
"""
import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=None)
def _load(name, columns):
    a = np.loadtxt(os.path.join(_DIR, name + '.txt'))
    for c in a.T:
        c.flags.writeable = False
    return SimpleNamespace(**dict(zip(columns, a.T)), name=name)


def key_201_2009():
    """201-point Hankel filter of Key (2009): base, j0, j1."""
    return _load('key_201_2009', ('base', 'j0', 'j1'))


def key_401_2009():
    """401-point Hankel filter of Key (2009): base, j0, j1."""
    return _load('key_401_2009', ('base', 'j0', 'j1'))


def key_201_2012():
    """201-point sine/cosine Fourier filter of Key (2012): base, sin, cos."""
    return _load('key_201_2012', ('base', 'sin', 'cos'))
