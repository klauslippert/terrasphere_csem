#!/usr/bin/env python3
"""elfe3D input of one computation folder (KH-027): for every height H of <folder>/steuer.toml and every frequency
band a run folder <folder>/elfe3d/<model>_<band>/in/ with model.poly (TetGen PLC), model.a.node (extra vertices
around the receivers), regionparameters.txt, source.txt and elfe3D_input.txt; <model> = flach | huegel<H> | tal<H>.
Bands: 'nf' = frequencies < 100 Hz (domain +-15 km); 'hf1k' (100 Hz..10 kHz) and 'hf100k' (>= 10 kHz): domain
+-4 km, coarser start mesh refined by elfe3D, one frequency per run (KH-029, KH-032).
Frame z up, receivers on the terrain 1 cm below the surface (as in the predecessor comparisons).

PLC (from gausshill/elfe3d/erzeuge_netz.py, extended by layers and general sources):
  horizontal planes: ground z = 0 (rings domain | medium box | fine box | terrain patch), every layer interface
  z = -d_j (rings domain | medium box | inner), top/bottom of the boxes; vertical box faces split at every plane they
  cross.  Terrain patch [-P, P]^2, node spacing 20 m, h = 0 on its boundary.  Sources as chains of 10 m segments in the
  ground plane (integer coordinates; elfe3D detects source edges by exact comparison); loops (type 7) with the first
  corner repeated at the end (elfe3D runs over the segments 1..n-1 only, kugel-semianalytik KSA-067).
Usage: python3 elfe3d_eingabe.py <folder>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from terrasphere_csem.steuerung import Steuerung

RHO_AIR = 1e8
P, D_PATCH, FINE, DS = 680.0, 20.0, 760.0, 50.0       # DS 50 m: source chain (KH-028)
MED_MARGIN, MED_ZA = 300.0, 300.0
Z_RX = -0.01
RX_CLUSTER = [(2.1, 0.3, -1.5), (-1.9, -0.4, -1.5), (0.35, 2.05, -1.5), (-0.25, -1.95, -1.5), (0.8, -0.6, -3.7),
              (1.1, 0.7, 2.0), (-0.9, -1.2, 2.0)]
BAENDER = {
    # sizes chosen for <= ~1.1 million edges (elfe3D/MUMPS ~ 14 GB per million edges, cap 18 GB; KH-028)
    'nf': dict(fmax=100.0, D=15000.0, med_z=400.0, ref_schritte=0, max_unbekannte=3000000,
               maxvol=dict(fine_earth=4000.0, fine_air=60000.0, med_earth=100000.0, med_air=300000.0)),
    # 1 kHz and 100 kHz: coarser start mesh, goal-oriented refinement by elfe3D up to 1.1 million unknowns.
    # elfe3D refines inside its frequency loop and stops when the unknowns do not change, so with refinement
    # only ONE frequency per run gives results (KH-032): one band per frequency.
    'hf1k': dict(fmin=100.0, fmax=1e4, D=4000.0, med_z=150.0, ref_schritte=6, max_unbekannte=1100000, beta_ref=0.6, ref_strategie=0,
                 maxvol=dict(fine_earth=6000.0, fine_air=60000.0, med_earth=20000.0, med_air=300000.0)),
    'hf100k': dict(fmin=1e4, D=4000.0, med_z=150.0, ref_schritte=6, max_unbekannte=1100000, beta_ref=0.6, ref_strategie=0,
                   maxvol=dict(fine_earth=6000.0, fine_air=60000.0, med_earth=20000.0, med_air=300000.0)),
}


class Poly:
    def __init__(self):
        self.p, self.idx, self.fac = [], {}, []

    def pt(self, x, y, z):
        key = (round(x, 9), round(y, 9), round(z, 9))
        if key not in self.idx:
            self.p.append(key); self.idx[key] = len(self.p)
        return self.idx[key]

    def facet(self, polys, marker=0, holes=()):
        self.fac.append((polys, marker, list(holes)))

    def write(self, path, regions, head):
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


def quellwege(st):
    """Source paths in the ground plane: list of (type, corner list) with integer-coordinate corners."""
    out = []
    for q in st.sender:
        if q['typ'] == 'draht':
            out.append(('draht', [tuple(q['von']), tuple(q['nach'])], float(q.get('strom', 1.0))))
        elif q['typ'] == 'rechteck':
            (x1, y1), (x2, y2) = q['ecke1'], q['ecke2']
            out.append(('schleife', [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], float(q.get('strom', 1.0))))
        elif q['typ'] == 'polygon':
            out.append(('schleife', [tuple(p) for p in q['ecken']], float(q.get('strom', 1.0))))
        else:
            raise NotImplementedError(f"elfe3D input for source type {q['typ']} not implemented")
    assert len(out) == 1, "one source per elfe3D run"
    return out[0]


def empfaenger(st):
    px, py, _ = st.empfaenger()
    return list(zip(px, py))


def build(st, H, band):
    b = BAENDER[band]; D = b['D']; mz = b['med_z']
    hfun = lambda x, y: 0.0 if (abs(x) >= P - 1e-9 or abs(y) >= P - 1e-9) else st.h(x, y, H)
    typ, ecken, _ = quellwege(st)
    ex = [e[0] for e in ecken] + [-FINE, FINE]; ey = [e[1] for e in ecken] + [-FINE, FINE]
    MX = (min(ex) - MED_MARGIN, max(ex) + MED_MARGIN); MY = (min(ey) - MED_MARGIN, max(ey) + MED_MARGIN)
    assert MX[0] > -D + 500 and MX[1] < D - 500 and MY[0] > -D + 500 and MY[1] < D - 500, "domain too small"
    z_lo, z_hi = min(0.0, H) - 40.0, max(0.0, H) + 40.0
    zi = [-d for d in st.schichten.tiefen[1:]]                       # interface planes (z < 0)
    assert all(z < z_lo for z in zi), "layer interfaces must lie below the fine box (valley floor - 40 m)"
    Pl = Poly()
    rect = lambda X, Y, z: [Pl.pt(X[0], Y[0], z), Pl.pt(X[1], Y[0], z), Pl.pt(X[1], Y[1], z), Pl.pt(X[0], Y[1], z)]
    DOM, MED, FIN = ((-D, D), (-D, D)), (MX, MY), ((-FINE, FINE), (-FINE, FINE))
    # vertical extents
    boxes = [(DOM, -D, D, 1), (MED, -mz, MED_ZA, 0), (FIN, z_lo, z_hi, 0)]
    planes = sorted(set([0.0] + [z for z in zi if z > -D]))
    for (X, Y), zb, zt, mark in boxes:
        lev = sorted(set([zb, zt] + [z for z in planes if zb < z < zt]))
        Pl.facet([rect(X, Y, zt)], mark); Pl.facet([rect(X, Y, zb)], mark)
        for za, zc in zip(lev[:-1], lev[1:]):
            lo, hi = rect(X, Y, za), rect(X, Y, zc)
            for k in range(4):
                Pl.facet([[lo[k], lo[(k + 1) % 4], hi[(k + 1) % 4], hi[k]]], mark)
    # interface planes: rings between the boxes they cross, inner part full
    for z in zi:
        crossed = [(XY, zb, zt) for XY, zb, zt, _ in boxes if zb < z < zt]
        for (o, _, _), (inn, _, _) in zip(crossed[:-1], crossed[1:]):
            Pl.facet([rect(*o, z), rect(*inn, z)], 0, holes=[((inn[0][0] + inn[0][1]) / 2, (inn[1][0] + inn[1][1]) / 2, z)])
        Pl.facet([rect(*crossed[-1][0], z)], 0)
    # terrain patch
    n = int(round(2 * P / D_PATCH)); xs = np.linspace(-P, P, n + 1)
    ids = np.zeros((n + 1, n + 1), int)
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            ids[i, j] = Pl.pt(float(x), float(y), float(hfun(x, y)))
    ring = [ids[i, 0] for i in range(n + 1)] + [ids[n, j] for j in range(1, n + 1)] + \
           [ids[i, n] for i in range(n - 1, -1, -1)] + [ids[0, j] for j in range(n - 1, 0, -1)]
    # source chain in the medium ring of the ground plane
    pts = list(ecken) + ([ecken[0]] if typ == 'schleife' else [])
    segs = []
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        nw = int(round(np.hypot(xb - xa, yb - ya) / DS))
        ch = [Pl.pt(xa + (xb - xa) * i / nw, ya + (yb - ya) * i / nw, 0.0) for i in range(nw + 1)]
        segs += [[ch[i], ch[i + 1]] for i in range(nw)]
    for x, y in ecken:
        assert not (abs(x) < FINE + 100 and abs(y) < FINE + 100), "source too close to the fine box"
    Pl.facet([rect(*DOM, 0.0), rect(*MED, 0.0)], 99, holes=[(0.0, 0.0, 0.0)])
    Pl.facet([rect(*MED, 0.0), rect(*FIN, 0.0)] + segs, 99, holes=[(0.0, 0.0, 0.0)])
    Pl.facet([rect(*FIN, 0.0), ring], 99, holes=[(0.0, 0.0, 0.0)])
    for i in range(n):
        for j in range(n):
            a, b_, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
            if (i + j) % 2 == 0:
                Pl.facet([[a, b_, c]], 99); Pl.facet([[a, c, d]], 99)
            else:
                Pl.facet([[a, b_, d]], 99); Pl.facet([[b_, c, d]], 99)
    # regions: air (1), layers (2..J+1); outer / medium / fine
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
    return Pl, reg, hfun, (typ, ecken), D


def eingabe(st, freqs, hfun, quelle, D, band):
    rx = empfaenger(st)
    typ, ecken = quelle
    _, _, I = quellwege(st)
    if typ == 'draht':
        stype, s0, s1 = 6, ecken[0], ecken[1]; laenge = float(np.hypot(ecken[1][0] - ecken[0][0], ecken[1][1] - ecken[0][1]))
    else:
        stype, s0, s1 = 7, ecken[0], ecken[1]
        laenge = float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ecken, ecken[1:] + ecken[:1])))
    z = [f"{'solver':24s}2", "model_size", f"{-D:.1f} {-D:.1f} {-D:.1f}", f"{D:.1f} {D:.1f} {D:.1f}",
         f"{'num_freq':24s}{len(freqs)}"] + [f"{f:.6g}" for f in freqs] + \
        [f"{'num_rec':24s}{len(rx)}"] + [f"{x:.3f}\t{y:.3f}\t{hfun(x, y) + Z_RX:.6f}" for x, y in rx] + \
        [f"{'output_E_file':24s}out/electric_fields", f"{'output_H_file':24s}out/magnetic_fields",
         f"{'source_type':24s}{stype}", f"{s0[0]:.10f} {s0[1]:.10f} 0.0", f"{s1[0]:.10f} {s1[1]:.10f} 0.0",
         f"{'current_direction':24s}1", f"{'source_moment':24s}{I * laenge:.10g}",
         f"{'PEC_present':24s}0", f"{'num_PEC':24s}1", f"{D / 2:.1f} 0.0 0.0", f"{D / 2:.1f} 0.0 -500.0",
         f"{'model_file_name':24s}in/model.", f"{'maxRefSteps':24s}{BAENDER[band]['ref_schritte']}",
         f"{'maxUnknowns':24s}{BAENDER[band]['max_unbekannte']}",
         f"{'betaRef':24s}{BAENDER[band].get('beta_ref', 0.85)}", f"{'accuracyTol':24s}0.00003", f"{'vtk':24s}0", f"{'errorEst_method':24s}4",
         f"{'refStrategy':24s}{BAENDER[band].get('ref_strategie', 1)}"]
    return "\n".join(z) + "\n"


def modellname(H):
    return 'flach' if H == 0 else (f'huegel{abs(H):g}' if H > 0 else f'tal{abs(H):g}')


def main():
    st = Steuerung(sys.argv[1])
    nur = set(sys.argv[2:])                                         # optional: only these run names (e.g. flach_nf)
    for band, b in BAENDER.items():
        fr = [f for f in st.freqs if (f < b.get('fmax', np.inf) and f >= b.get('fmin', 0.0))]
        assert band == 'nf' or len(fr) <= 1, 'refinement bands: one frequency per run'
        if not fr:
            continue
        for H in st.hoehen:
            if nur and f'{modellname(H)}_{band}' not in nur:
                continue
            d = os.path.join(st.ordner, 'elfe3d', f'{modellname(H)}_{band}', 'in'); os.makedirs(d, exist_ok=True)
            Pl, reg, hfun, quelle, D = build(st, H, band)
            Pl.write(os.path.join(d, 'model.poly'), reg, f"terrasphere-csem {st.name}, H = {H} m, band {band}")
            with open(os.path.join(d, 'regionparameters.txt'), 'w') as fh:
                J = st.schichten.J
                fh.write(f"# eleattr\n{J + 1}\n# eleattr rho mu_r epsilon_r\n1 {RHO_AIR:.10g} 1.0 0.0\n")
                for j in range(J):
                    fh.write(f"{j + 2} {1.0 / st.schichten.sigma_g[j]:.10g} 1.0 0.0\n")
            typ, ecken = quelle
            with open(os.path.join(d, 'source.txt'), 'w') as fh:
                pts = list(ecken) + ([ecken[0]] if typ == 'schleife' else [])
                fh.write(f"{len(pts)}\n" + "".join(f"{x:.10f} {y:.10f} 0.0\n" for x, y in pts))
            with open(os.path.join(d, 'elfe3D_input.txt'), 'w') as fh:
                fh.write(eingabe(st, fr, hfun, quelle, D, band))
            pts = []
            for x, y in sorted(set(empfaenger(st))):                  # the summit is on both profiles: once
                for dx, dy, dz in RX_CLUSTER:
                    pts.append((x + dx, y + dy, hfun(x + dx, y + dy) + dz))
            with open(os.path.join(d, 'model.a.node'), 'w') as fh:
                fh.write(f"{len(pts)} 3 0 0\n" + "".join(f"{i + 1} {x:.6f} {y:.6f} {z:.6f}\n" for i, (x, y, z) in enumerate(pts)))
            print(f"{st.name}: {modellname(H)}_{band}: freqs {[float(v) for v in fr]}, nodes {len(Pl.p)}, facets {len(Pl.fac)}, "
                  f"regions {len(reg)}, receivers {len(empfaenger(st))}")


if __name__ == '__main__':
    main()
