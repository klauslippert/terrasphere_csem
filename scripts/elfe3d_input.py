#!/usr/bin/env python3
"""elfe3D input of one computation folder: for every height H of <folder>/control.toml and every frequency
band a run folder <folder>/elfe3d/<model>_<band>/in/ with model.poly (TetGen PLC), model.a.node (extra vertices
around the receivers), regionparameters.txt, source.txt and elfe3D_input.txt; <model> = flat | hill<H> | valley<H>.
Bands: 'nf' = frequencies < 100 Hz (domain +-15 km); 'hf1k' (100 Hz..10 kHz) and 'hf100k' (>= 10 kHz): domain
+-4 km, coarser start mesh refined by elfe3D, one frequency per run.
Frame z up, receivers on the terrain 1 cm below the surface.

PLC (layered earth, general sources):
  horizontal planes: ground z = 0 (rings domain | medium box | fine box | terrain patch), every layer interface
  z = -d_j (rings domain | medium box | inner), top/bottom of the boxes; vertical box faces split at every plane they
  cross.  Terrain patch [-P, P]^2, node spacing 20 m, h = 0 on its boundary.  Sources as chains of 10 m segments in the
  ground plane (integer coordinates; elfe3D detects source edges by exact comparison); loops (type 7) with the first
  corner repeated at the end (elfe3D runs over the segments 1..n-1 only).
Usage: python3 elfe3d_input.py <folder>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from terrasphere_csem.config import Config

RHO_AIR = 1e8
P, D_PATCH, FINE, DS = 680.0, 20.0, 760.0, 50.0       # DS 50 m: source chain
MED_MARGIN, MED_ZA = 300.0, 300.0
Z_RX = -0.01
RX_CLUSTER = [(2.1, 0.3, -1.5), (-1.9, -0.4, -1.5), (0.35, 2.05, -1.5), (-0.25, -1.95, -1.5), (0.8, -0.6, -3.7),
              (1.1, 0.7, 2.0), (-0.9, -1.2, 2.0)]
BANDS = {
    # sizes chosen for <= ~1.1 million edges (elfe3D/MUMPS ~ 14 GB per million edges, cap 18 GB)
    'nf': dict(fmax=100.0, D=15000.0, med_z=400.0, ref_steps=0, max_unknowns=3000000,
               maxvol=dict(fine_earth=4000.0, fine_air=60000.0, med_earth=100000.0, med_air=300000.0)),
    # 1 kHz and 100 kHz: coarser start mesh, goal-oriented refinement by elfe3D up to 1.1 million unknowns.
    # elfe3D refines inside its frequency loop and stops when the unknowns do not change, so with refinement
    # only ONE frequency per run gives results: one band per frequency.
    'hf1k': dict(fmin=100.0, fmax=1e4, D=4000.0, med_z=150.0, ref_steps=6, max_unknowns=1100000, beta_ref=0.6, ref_strategy=0,
                 maxvol=dict(fine_earth=6000.0, fine_air=60000.0, med_earth=20000.0, med_air=300000.0)),
    'hf100k': dict(fmin=1e4, D=4000.0, med_z=150.0, ref_steps=6, max_unknowns=1100000, beta_ref=0.6, ref_strategy=0,
                   maxvol=dict(fine_earth=6000.0, fine_air=60000.0, med_earth=20000.0, med_air=300000.0)),
}


class Poly:
    """Piecewise linear complex (PLC, TetGen .poly format) under construction: unique points and facets."""
    def __init__(self):
        """Start with no points and no facets."""
        self.p, self.idx, self.fac = [], {}, []

    def pt(self, x, y, z):
        """Index (1-based) of the point (x, y, z); new points are appended, equal points are shared."""
        key = (round(x, 9), round(y, 9), round(z, 9))
        if key not in self.idx:
            self.p.append(key); self.idx[key] = len(self.p)
        return self.idx[key]

    def facet(self, polys, marker=0, holes=()):
        """Add a facet given by one or more polygons (lists of point indices), a boundary marker and hole points."""
        self.fac.append((polys, marker, list(holes)))

    def write(self, path, regions, head):
        """Write the PLC with its region list (x, y, z, attribute, max. volume, comment) to a .poly file."""
        with open(path, 'w') as fh:
            fh.write(f"# {head}\n{len(self.p)} 3 0 0\n")
            for i, (x, y, z) in enumerate(self.p):
                fh.write(f"{i + 1} {x:.9f} {y:.9f} {z:.9f}\n")
            fh.write(f"{len(self.fac)} 1\n")
            for polys, m, holes in self.fac:
                fh.write(f"{len(polys)} {len(holes)} {m}\n")
                for pl in polys:
                    fh.write(f"{len(pl)} " + " ".join(map(str, pl)) + "\n")
                for k, hp in enumerate(holes):
                    fh.write(f"{k + 1} {hp[0]:.6f} {hp[1]:.6f} {hp[2]:.6f}\n")
            fh.write("0\n")
            fh.write(f"{len(regions)}\n")
            for i, (x, y, z, att, mv, com) in enumerate(regions):
                fh.write(f"{i + 1} {x:.3f} {y:.3f} {z:.3f} {att} {mv:g}  # {com}\n")


def source_paths(st):
    """Source paths in the ground plane: list of (type, corner list) with integer-coordinate corners."""
    out = []
    for q in st.source_specs:
        if q['type'] == 'wire':
            out.append(('wire', [tuple(q['start']), tuple(q['end'])], float(q.get('current', 1.0))))
        elif q['type'] == 'rectangle':
            (x1, y1), (x2, y2) = q['corner1'], q['corner2']
            out.append(('loop', [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], float(q.get('current', 1.0))))
        elif q['type'] == 'polygon':
            out.append(('loop', [tuple(p) for p in q['corners']], float(q.get('current', 1.0))))
        else:
            raise NotImplementedError(f"elfe3D input for source type {q['type']} not implemented")
    assert len(out) == 1, "one source per elfe3D run"
    return out[0]


def receivers(st):
    """Receiver positions [(x, y)] of the control file."""
    px, py, _ = st.receivers()
    return list(zip(px, py))


def _rect(Pl, X, Y, z):
    """Point indices of the corners of the rectangle X x Y at height z."""
    return [Pl.pt(X[0], Y[0], z), Pl.pt(X[1], Y[0], z), Pl.pt(X[1], Y[1], z), Pl.pt(X[0], Y[1], z)]


def _box_facets(Pl, boxes, planes):
    """Top, bottom and side walls of the nested boxes; the side walls are split at every plane they cross."""
    for (X, Y), zb, zt, mark in boxes:
        lev = sorted(set([zb, zt] + [z for z in planes if zb < z < zt]))
        Pl.facet([_rect(Pl, X, Y, zt)], mark); Pl.facet([_rect(Pl, X, Y, zb)], mark)
        for za, zc in zip(lev[:-1], lev[1:]):
            lo, hi = _rect(Pl, X, Y, za), _rect(Pl, X, Y, zc)
            for k in range(4):
                Pl.facet([[lo[k], lo[(k + 1) % 4], hi[(k + 1) % 4], hi[k]]], mark)


def _interface_facets(Pl, zi, boxes):
    """Layer interface planes z = zi: rings between the boxes they cross, the innermost part full."""
    for z in zi:
        crossed = [(XY, zb, zt) for XY, zb, zt, _ in boxes if zb < z < zt]
        for (o, _, _), (inn, _, _) in zip(crossed[:-1], crossed[1:]):
            Pl.facet([_rect(Pl, *o, z), _rect(Pl, *inn, z)], 0,
                     holes=[((inn[0][0] + inn[0][1]) / 2, (inn[1][0] + inn[1][1]) / 2, z)])
        Pl.facet([_rect(Pl, *crossed[-1][0], z)], 0)


def _terrain_patch_points(Pl, hfun):
    """Grid of terrain points on the patch [-P, P]^2 (spacing D_PATCH): (n, point indices (n+1, n+1), boundary ring)."""
    n = int(round(2 * P / D_PATCH)); xs = np.linspace(-P, P, n + 1)
    ids = np.zeros((n + 1, n + 1), int)
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            ids[i, j] = Pl.pt(float(x), float(y), float(hfun(x, y)))
    ring = [ids[i, 0] for i in range(n + 1)] + [ids[n, j] for j in range(1, n + 1)] + \
           [ids[i, n] for i in range(n - 1, -1, -1)] + [ids[0, j] for j in range(n - 1, 0, -1)]
    return n, ids, ring


def _source_chain(Pl, kind, corners):
    """The source path as a chain of segments of about DS metres in the ground plane (list of point-index pairs)."""
    pts = list(corners) + ([corners[0]] if kind == 'loop' else [])
    segs = []
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        nw = int(round(np.hypot(xb - xa, yb - ya) / DS))
        ch = [Pl.pt(xa + (xb - xa) * i / nw, ya + (yb - ya) * i / nw, 0.0) for i in range(nw + 1)]
        segs += [[ch[i], ch[i + 1]] for i in range(nw)]
    return segs


def _ground_plane_facets(Pl, DOM, MED, FIN, segs, ring, n, ids):
    """Facets of the plane z = 0 outside the terrain patch (marker 99) and the triangulated terrain patch."""
    Pl.facet([_rect(Pl, *DOM, 0.0), _rect(Pl, *MED, 0.0)], 99, holes=[(0.0, 0.0, 0.0)])
    Pl.facet([_rect(Pl, *MED, 0.0), _rect(Pl, *FIN, 0.0)] + segs, 99, holes=[(0.0, 0.0, 0.0)])
    Pl.facet([_rect(Pl, *FIN, 0.0), ring], 99, holes=[(0.0, 0.0, 0.0)])
    for i in range(n):
        for j in range(n):
            a, b_, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
            if (i + j) % 2 == 0:
                Pl.facet([[a, b_, c]], 99); Pl.facet([[a, c, d]], 99)
            else:
                Pl.facet([[a, b_, d]], 99); Pl.facet([[b_, c, d]], 99)


def _regions(b, D, MX, MY, mz, z_lo, z_hi, zi):
    """Region seeds (x, y, z, attribute, max. volume, comment): air (attribute 1), layers (2..J+1); outer, medium and
    fine parts."""
    mv = b['maxvol']; e = 1.0
    reg = [(0.0, 0.0, 0.9 * D, 1, -1, 'air outer'),
           (MX[1] - e, MY[1] - e, MED_ZA / 2, 1, mv['med_air'], 'air medium'),
           (FINE - e, FINE - e, z_hi / 2, 1, mv['fine_air'], 'air fine')]
    tops = [0.0] + zi; bots = zi + [-D]
    for j, (zt, zb) in enumerate(zip(tops, bots)):
        att = j + 2
        reg.append((0.9 * D, 0.9 * D, (zt + zb) / 2, att, -1, f'layer {j + 1} outer'))
        zt_m, zb_m = min(zt, z_lo if j == 0 else zt), max(zb, -mz)
        if zt_m > zb_m:
            reg.append((MX[1] - e, MY[1] - e, (zt_m + zb_m) / 2, att, mv['med_earth'], f'layer {j + 1} medium'))
        if j == 0:
            reg.append((FINE - e, FINE - e, z_lo / 2, att, mv['fine_earth'], 'layer 1 fine'))
    return reg


def build(st, H, band):
    """Mesh geometry (Poly, regions, terrain function, source description, domain size) for height H and band."""
    b = BANDS[band]; D = b['D']; mz = b['med_z']
    hfun = lambda x, y: 0.0 if (abs(x) >= P - 1e-9 or abs(y) >= P - 1e-9) else st.height(x, y, H)
    kind, corners, _ = source_paths(st)
    ex = [e[0] for e in corners] + [-FINE, FINE]; ey = [e[1] for e in corners] + [-FINE, FINE]
    MX = (min(ex) - MED_MARGIN, max(ex) + MED_MARGIN); MY = (min(ey) - MED_MARGIN, max(ey) + MED_MARGIN)
    assert MX[0] > -D + 500 and MX[1] < D - 500 and MY[0] > -D + 500 and MY[1] < D - 500, "domain too small"
    z_lo, z_hi = min(0.0, H) - 40.0, max(0.0, H) + 40.0
    zi = [-d for d in st.layers.depths[1:]]                       # interface planes (z < 0)
    assert all(z < z_lo for z in zi), "layer interfaces must lie below the fine box (valley floor - 40 m)"
    Pl = Poly()
    DOM, MED, FIN = ((-D, D), (-D, D)), (MX, MY), ((-FINE, FINE), (-FINE, FINE))
    boxes = [(DOM, -D, D, 1), (MED, -mz, MED_ZA, 0), (FIN, z_lo, z_hi, 0)]    # outer domain, medium box, fine box
    planes = sorted(set([0.0] + [z for z in zi if z > -D]))
    _box_facets(Pl, boxes, planes)
    _interface_facets(Pl, zi, boxes)
    n, ids, ring = _terrain_patch_points(Pl, hfun)
    segs = _source_chain(Pl, kind, corners)
    for x, y in corners:
        assert not (abs(x) < FINE + 100 and abs(y) < FINE + 100), "source too close to the fine box"
    _ground_plane_facets(Pl, DOM, MED, FIN, segs, ring, n, ids)
    reg = _regions(b, D, MX, MY, mz, z_lo, z_hi, zi)
    return Pl, reg, hfun, (kind, corners), D


def input_file(st, freqs, hfun, source, D, band):
    """Text of the elfe3D input file for the frequencies freqs and the given source and band."""
    rx = receivers(st)
    kind, corners = source
    _, _, I = source_paths(st)
    if kind == 'wire':
        stype, s0, s1 = 6, corners[0], corners[1]; length = float(np.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]))
    else:
        stype, s0, s1 = 7, corners[0], corners[1]
        length = float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(corners, corners[1:] + corners[:1])))
    z = [f"{'solver':24s}2", "model_size", f"{-D:.1f} {-D:.1f} {-D:.1f}", f"{D:.1f} {D:.1f} {D:.1f}",
         f"{'num_freq':24s}{len(freqs)}"] + [f"{f:.6g}" for f in freqs] + \
        [f"{'num_rec':24s}{len(rx)}"] + [f"{x:.3f}\t{y:.3f}\t{hfun(x, y) + Z_RX:.6f}" for x, y in rx] + \
        [f"{'output_E_file':24s}out/electric_fields", f"{'output_H_file':24s}out/magnetic_fields",
         f"{'source_type':24s}{stype}", f"{s0[0]:.10f} {s0[1]:.10f} 0.0", f"{s1[0]:.10f} {s1[1]:.10f} 0.0",
         f"{'current_direction':24s}1", f"{'source_moment':24s}{I * length:.10g}",
         f"{'PEC_present':24s}0", f"{'num_PEC':24s}1", f"{D / 2:.1f} 0.0 0.0", f"{D / 2:.1f} 0.0 -500.0",
         f"{'model_file_name':24s}in/model.", f"{'maxRefSteps':24s}{BANDS[band]['ref_steps']}",
         f"{'maxUnknowns':24s}{BANDS[band]['max_unknowns']}",
         f"{'betaRef':24s}{BANDS[band].get('beta_ref', 0.85)}", f"{'accuracyTol':24s}0.00003", f"{'vtk':24s}0", f"{'errorEst_method':24s}4",
         f"{'refStrategy':24s}{BANDS[band].get('ref_strategy', 1)}"]
    return "\n".join(z) + "\n"


def model_name(H):
    """Name of the model of height H [m]: 'flat', 'hill<H>' or 'valley<H>'."""
    return 'flat' if H == 0 else (f'hill{abs(H):g}' if H > 0 else f'valley{abs(H):g}')


def main():
    """Write the elfe3D input of all models and bands of the control folder given on the command line."""
    st = Config(sys.argv[1])
    only = set(sys.argv[2:])                                         # optional: only these run names (e.g. flat_nf)
    for band, b in BANDS.items():
        fr = [f for f in st.freqs if (f < b.get('fmax', np.inf) and f >= b.get('fmin', 0.0))]
        assert band == 'nf' or len(fr) <= 1, 'refinement bands: one frequency per run'
        if not fr:
            continue
        for H in st.heights:
            if only and f'{model_name(H)}_{band}' not in only:
                continue
            d = os.path.join(st.folder, 'elfe3d', f'{model_name(H)}_{band}', 'in'); os.makedirs(d, exist_ok=True)
            Pl, reg, hfun, source, D = build(st, H, band)
            Pl.write(os.path.join(d, 'model.poly'), reg, f"terrasphere-csem {st.name}, H = {H} m, band {band}")
            with open(os.path.join(d, 'regionparameters.txt'), 'w') as fh:
                J = st.layers.J
                fh.write(f"# eleattr\n{J + 1}\n# eleattr rho mu_r epsilon_r\n1 {RHO_AIR:.10g} 1.0 0.0\n")
                for j in range(J):
                    fh.write(f"{j + 2} {1.0 / st.layers.sigma_g[j]:.10g} 1.0 0.0\n")
            kind, corners = source
            with open(os.path.join(d, 'source.txt'), 'w') as fh:
                pts = list(corners) + ([corners[0]] if kind == 'loop' else [])
                fh.write(f"{len(pts)}\n" + "".join(f"{x:.10f} {y:.10f} 0.0\n" for x, y in pts))
            with open(os.path.join(d, 'elfe3D_input.txt'), 'w') as fh:
                fh.write(input_file(st, fr, hfun, source, D, band))
            pts = []
            for x, y in sorted(set(receivers(st))):                  # the summit is on both profiles: once
                for dx, dy, dz in RX_CLUSTER:
                    pts.append((x + dx, y + dy, hfun(x + dx, y + dy) + dz))
            with open(os.path.join(d, 'model.a.node'), 'w') as fh:
                fh.write(f"{len(pts)} 3 0 0\n" + "".join(f"{i + 1} {x:.6f} {y:.6f} {z:.6f}\n" for i, (x, y, z) in enumerate(pts)))
            print(f"{st.name}: {model_name(H)}_{band}: freqs {[float(v) for v in fr]}, nodes {len(Pl.p)}, facets {len(Pl.fac)}, "
                  f"regions {len(reg)}, receivers {len(receivers(st))}")


if __name__ == '__main__':
    main()
