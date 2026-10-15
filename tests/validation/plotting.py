"""Plotting helpers shared by the comparisons in 1D/ (empymod) and 3D/ (elfe3D): source paths of a control file,
the setup figure (plan view and xz-section through y = 0 with source, receivers, terrain, layers and sphere), the
profile figure (Re/Im of E and H, all three components, series against a reference, all frequencies) and LaTeX
escaping."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

COLORS = ['#1f5fa8', '#2a9d5c', '#d9731a', '#8e44ad', '#c0392b']        # per frequency
COMPONENTS = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
MODEL_COLORS = {0.0: 'k', 1.0: '#c0392b', -1.0: '#1f5fa8'}


def factor_into_labels(fig):
    """Every linear y-axis of fig: no separate factor or offset above the axis; a power of ten of the tick values
    goes into the axis label instead, 'E (mV/m)' -> 'E ($10^{-11}$ mV/m)' (a label without a unit in parentheses
    gets '($\\times10^{n}$)' appended).  Tick labels show the values divided by that power, with the fewest decimals
    that show them exactly.  Call after plotting, before tight_layout."""
    fig.canvas.draw()
    for a in fig.axes:
        if a.get_yscale() != 'linear' or not a.get_visible() or not a.axison:
            continue
        lo, hi = a.get_ylim()
        ticks = [v for v in a.get_yticks() if min(lo, hi) - 1e-12 * abs(hi - lo) <= v <= max(lo, hi) + 1e-12 * abs(hi - lo)]
        m = max((abs(v) for v in ticks), default=0.0)
        oom = int(np.floor(np.log10(m))) if m > 0 and not 1e-2 <= m < 1e4 else 0
        sc = 10.0 ** -oom
        for nd in range(0, 16):                                      # fewest decimals that keep the ticks distinct
            labs = [f'{v * sc + 0.0:.{nd}f}' for v in ticks]
            if len(set(labs)) == len(labs) and all(abs(float(q) - v * sc) <= 1e-9 * max(m * sc, 1e-300)
                                                    for q, v in zip(labs, ticks)):
                break
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _, sc=sc, nd=nd: f'{v * sc + 0.0:.{nd}f}'.replace('-0' + ('.' + '0' * nd if nd else ''),
                                                                       '0' + ('.' + '0' * nd if nd else ''))))
        a.yaxis.offsetText.set_visible(False)
        if oom:
            lab = a.get_ylabel()
            if lab.endswith(')') and '(' in lab:
                i = lab.rindex('(')
                a.set_ylabel(lab[:i] + '($10^{%d}$ ' % oom + lab[i + 1:], fontsize=a.yaxis.label.get_fontsize())
            else:
                a.set_ylabel((lab + ' ' if lab else '') + r'($\times10^{%d}$)' % oom, fontsize=a.yaxis.label.get_fontsize())


def source_polylines(st):
    """Source paths as lists of (x, y) corners (closed for loops)."""
    out = []
    for q in st.source_specs:
        if q['type'] == 'wire':
            out.append([tuple(q['start']), tuple(q['end'])])
        elif q['type'] == 'rectangle':
            (x0, y0), (x1, y1) = q['corner1'], q['corner2']
            out.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
        elif 'corners' in q:
            c = [tuple(p) for p in q['corners']]
            out.append(c + [c[0]])
    return out


def section_crossings(path, y=0.0):
    """Points (x) and pieces [x0, x1] where a polyline meets the plane y = const."""
    pts, segs = [], []
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        if y0 == y and y1 == y:
            segs.append((min(x0, x1), max(x0, x1)))
        elif (y0 - y) * (y1 - y) <= 0 and y0 != y1:
            pts.append(x0 + (y - y0) / (y1 - y0) * (x1 - x0))
    return sorted(set(pts)), segs


def setup_figure(st, d, prof, H, path):
    """Plan view and xz-section (y = 0) with source, receivers of profile prof, terrain of height H and layers."""
    srcs = source_polylines(st)
    allx = [p[0] for s in srcs for p in s] + [-600.0, 600.0]
    xl = (min(allx) - 250, max(allx) + 250)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw=dict(width_ratios=[1, 1.35]))
    # plan view
    for k, s in enumerate(srcs):
        s = np.array(s); a1.plot(s[:, 0], s[:, 1], '-', color='#d9731a', lw=2.2, label='source (Tx)' if k == 0 else None)
    for p, ls in (('x', '-'), ('y', '-')):
        sel = d['profile'] == p
        on = p == prof
        a1.plot(d['px'][sel], d['py'][sel], 'o' if on else '.', ms=2.5 if on else 1.5,
                color='#2a9d5c' if on else '0.7', label=f'{p}-profile (this section)' if on else f'{p}-profile')
    for r, ls in ((st.s, '-'), (2 * st.s, '--')):
        a1.add_patch(plt.Circle((0, 0), r, fill=False, color='0.4', ls=ls, lw=0.8))
    a1.plot([], [], '-', color='0.4', lw=0.8, label=r'terrain: $s$ and $2s$ ($s$ = %g m)' % st.s)
    a1.axhline(0, color='0.85', lw=0.6, zorder=0)
    if st.sphere:                                                    # sphere: outline in plan view and section
        xc, yc, dc = (float(v) for v in st.sphere['center']); rs = float(st.sphere['radius'])
        lab = r'sphere %.3g $\Omega$m, $r$ = %g m' % (1 / float(st.sphere['conductivity']), rs)
        a1.add_patch(plt.Circle((xc, yc), rs, fill=False, color='#7b3fa0', lw=1.3))
        a1.plot([], [], '-', color='#7b3fa0', lw=1.3, label=lab)
        if abs(yc) < rs:
            a2.add_patch(plt.Circle((xc, -dc), np.sqrt(rs ** 2 - yc ** 2), color='#7b3fa0', alpha=0.35, lw=0))
            a2.plot([], [], 's', color='#7b3fa0', alpha=0.35, label=lab + ' (cut at y = 0)')
    a1.set_aspect('equal'); a1.set_xlim(*xl); a1.set_ylim(-(xl[1] - xl[0]) / 2.6, (xl[1] - xl[0]) / 2.6)
    if 'T4' in st.name:
        a1.set_ylim(-1550, 1550)
    a1.set_xlabel('x (m)'); a1.set_ylabel('y (m)'); a1.set_title('plan view', fontsize=10)
    a1.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    # xz-section through y = 0
    x = np.linspace(*xl, 800)
    zmin = -st.layers.depths[-1] - 250
    a2.plot(x, st.height(x, 0.0 * x, H), '-', color=MODEL_COLORS[float(np.sign(H))], lw=1.2,
            label=(f'hill {H:+g} m' if H > 0 else f'depression {H:+g} m' if H < 0 else 'flat'))
    dep = list(st.layers.depths[1:]); sig = list(st.layers.sigma_g)
    for z in dep:
        a2.axhline(-z, color='0.5', lw=0.8, ls='--')
    tops = [0.0] + dep; bots = dep + [-zmin]
    for j, s in enumerate(sig):
        zc = -(tops[j] + min(bots[j], -zmin)) / 2
        a2.text(xl[0] + 60, zc, r'$\rho_%d$ = %.3g $\Omega$m' % (j + 1, 1 / s), fontsize=7.5,
                va='center')
    a2.text(xl[0] + 60, 60, 'air', fontsize=7.5, va='center')
    for k, s in enumerate(srcs):
        pts, segs = section_crossings(s)
        for (x0, x1) in segs:
            a2.plot([x0, x1], [0, 0], '-', color='#d9731a', lw=3.5, solid_capstyle='butt',
                    label='source (Tx)' if k == 0 else None)
        if pts:
            a2.plot(pts, np.zeros(len(pts)), 'v', color='#d9731a', ms=7,
                    label='source crossing y = 0' if not segs and k == 0 else None)
    sel = d['profile'] == prof
    if prof == 'x':
        a2.plot(d['px'][sel], st.height(d['px'][sel], 0.0, H), '-', color='#2a9d5c', lw=2.5, label='x-profile (on the terrain)')
    else:
        a2.plot([0], [st.height(0.0, 0.0, H)], 'o', color='#2a9d5c', ms=6, label='y-profile (crosses y = 0 at x = 0)')
    a2.set_xlim(*xl); a2.set_ylim(zmin, 120)
    a2.set_xlabel('x (m)'); a2.set_ylabel('z (m, up)')
    a2.set_title('xz-section at y = 0 (vertical exaggeration)', fontsize=10)
    a2.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    for a in (a1, a2):
        a.tick_params(labelsize=8)
    factor_into_labels(fig)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def profile_figure(st, d, ser, ih, prof, path, P, ref, comps=COMPONENTS):
    """Fields on profile prof of model ih: rows Re E, Im E, Re H, Im H; columns x, y, z; all frequencies.  ser: series
    (lines); ref(f) -> (fields (6, P) or None, origin): reference (circles), e.g. the elfe3D run or empymod at that
    frequency.  Only the components in comps are drawn."""
    fig, ax = plt.subplots(4, 3, figsize=(10, 6.3), sharex=True)
    sel = d['profile'] == prof
    pos = (d['px'] if prof == 'x' else d['py'])[sel]
    origins = []
    for jf, f in enumerate(d['freqs']):
        el, origin = ref(f)
        if el is None:
            continue
        origins.append(f'{f:g} Hz: {origin}')
        for k, c in enumerate(COMPONENTS):
            if c not in comps:
                continue
            sc = 1e3 if k < 3 else 1e6
            for ir, (part, fn) in enumerate([('Re', np.real), ('Im', np.imag)]):
                a = ax[2 * (k // 3) + ir, k % 3]
                a.plot(pos, sc * fn(ser[ih, jf, k, sel]), '-', color=COLORS[jf], lw=1.1,
                       label=f'{f:g} Hz' if (k == 0 and ir == 0) else None)
                a.plot(pos[::2], sc * fn(el[k, sel][::2]), 'o', ms=2.8, mfc='none', color=COLORS[jf])
                a.set_title(f'{part} {c}', fontsize=9)
                a.tick_params(labelsize=7.5)
    for a in ax[-1]:
        a.set_xlabel(f'{prof} (m)', fontsize=8.5)
    for r in range(4):
        for j in range(3):
            ax[r, j].set_ylabel('E (mV/m)' if r < 2 else r'H ($\mu$A/m)', fontsize=8.5 if j == 0 else 7.5)
    ax[0, 0].legend(fontsize=7.5)
    factor_into_labels(fig)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return origins


def tex_escape(s):
    return s.replace('_', r'\_')
