#!/usr/bin/env python3
"""LaTeX report of the elfe3D validation: one section per case and profile (x, y), one subsection (page) per model
with terrain (hill, valley; the flat model is validated against empymod elsewhere): setup figure (plan view and
xz-section through y = 0 with source, profile, terrain and layers), then the fields on that profile: Re/Im of E and H,
all three components, series (lines) against elfe3D (circles), all frequencies with an elfe3D run.  Physical units (E in nV/m, H in uA/m, per 1 A), H with the elfe3D sign.
Writes <validation folder>/report/{report.tex, fig/*.pdf}.
Usage: python3 report_elfe3d.py <validation folder (with cases/)>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from terrasphere_csem.config import Config
from compare_elfe3d import model_name
from plot_profiles_freq import elfe_field, COLORS

COMPONENTS = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
MODEL_COLORS = {0.0: 'k', 1.0: '#c0392b', -1.0: '#1f5fa8'}


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
    a1.set_aspect('equal'); a1.set_xlim(*xl); a1.set_ylim(-(xl[1] - xl[0]) / 2.6, (xl[1] - xl[0]) / 2.6)
    if 'T4' in st.name:
        a1.set_ylim(-1550, 1550)
    a1.set_xlabel('x (m)'); a1.set_ylabel('y (m)'); a1.set_title('plan view', fontsize=10)
    a1.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    # xz-section through y = 0
    x = np.linspace(*xl, 800)
    zmin = -st.layers.depths[-1] - 250
    a2.plot(x, st.height(x, 0.0 * x, H), '-', color=MODEL_COLORS[float(np.sign(H))], lw=1.2,
            label=(f'hill {H:+g} m' if H > 0 else f'valley {H:+g} m'))
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
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def profile_figure(st, d, ser, ih, prof, path, P):
    """Fields on profile prof of model ih: rows Re E, Im E, Re H, Im H; columns x, y, z; all frequencies."""
    H = d['heights'][ih]
    fig, ax = plt.subplots(4, 3, figsize=(10, 6.3), sharex=True)
    sel = d['profile'] == prof
    pos = (d['px'] if prof == 'x' else d['py'])[sel]
    origins = []
    for jf, f in enumerate(d['freqs']):
        el, origin = elfe_field(st, H, f, P, 'out')
        if el is None:
            continue
        origins.append(f'{f:g} Hz: {origin}')
        for k, c in enumerate(COMPONENTS):
            sc = 1e9 if k < 3 else 1e6
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
        ax[r, 0].set_ylabel('E (nV/m)' if r < 2 else r'H ($\mu$A/m)', fontsize=8.5)
    ax[0, 0].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return origins


def tex_escape(s):
    return s.replace('_', r'\_')


def main():
    """Write the figures and report.tex for all cases (see the module docstring)."""
    root = os.path.abspath(sys.argv[1])
    out = os.path.join(root, 'report'); os.makedirs(os.path.join(out, 'fig'), exist_ok=True)
    cases = sorted(c for c in os.listdir(os.path.join(root, 'cases'))
                   if os.path.isfile(os.path.join(root, 'cases', c, 'results', 'fields.npz')))
    body = []
    for c in cases:
        st = Config(os.path.join(root, 'cases', c))
        d = np.load(os.path.join(st.folder, 'results', 'fields.npz'))
        ser = np.concatenate([d['E'], -d['H']], axis=2)               # (nh, nf, 6, P), elfe3D convention
        P = len(d['px'])
        desc = open(os.path.join(st.folder, 'control.toml')).readline().lstrip('# ').strip()
        for prof in d['profile'][np.sort(np.unique(d['profile'], return_index=True)[1])]:
            body.append(rf'\section{{{tex_escape(c)}, {prof}-profile}}' + '\n' + tex_escape(desc) + '\n')
            for ih, H in enumerate(d['heights']):
                if H == 0:                                             # flat: validated against empymod
                    continue
                m = model_name(H)
                lab = f'hill {H:g} m' if H > 0 else f'valley {-H:g} m'
                setup, fp = f'fig/{c}_{prof}_{m}_setup.pdf', f'fig/{c}_{prof}_{m}.pdf'
                setup_figure(st, d, prof, H, os.path.join(out, setup))
                orig = profile_figure(st, d, ser, ih, prof, os.path.join(out, fp), P)
                body.append(rf'\subsection{{{lab}}}' + '\n')
                body.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{%s}' % setup +
                            r'\caption{Setup of %s, %s-profile, %s: plan view (left) and xz-section at $y = 0$ (right).}'
                            % (tex_escape(c), prof, lab) + r'\end{figure}' + '\n')
                body.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{%s}' % fp +
                            r'\caption{%s, %s-profile, %s: series (lines) and elfe3D (circles, every second receiver). '
                            r'elfe3D runs: %s.}' % (tex_escape(c), prof, lab, tex_escape('; '.join(orig))) +
                            r'\end{figure}' + '\n' + r'\clearpage' + '\n')
            print(c, prof)
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[hmargin=2cm,vmargin=1.5cm]{geometry}
\usepackage[font=small,skip=2pt]{caption}
\setlength{\textfloatsep}{6pt}\setlength{\intextsep}{6pt}
\usepackage{graphicx,float}
\title{terrasphere\_csem: validation against elfe3D}
\date{\today}
\begin{document}
\maketitle
Series solution of this package (lines) against the 3D finite-element code elfe3D (circles) for four sources,
a Gaussian hill and a Gaussian valley (height $\pm30$ m, width $s = 150$ m); the flat model is validated against
empymod.  Three layers with
interfaces at 200 m and 600 m depth.  Receivers 1 cm below the surface on the x- and y-profile from $-600$ m to
600 m every 20 m.  Fields per 1 A source current, E in nV/m, H in $\mu$A/m, time dependence $e^{-i\omega t}$,
H with the sign convention of elfe3D ($H_\mathrm{elfe3D} = -H$).
\tableofcontents
\clearpage
""" + '\n'.join(body) + r"""
\end{document}
"""
    open(os.path.join(out, 'report.tex'), 'w').write(tex)
    print(os.path.join(out, 'report.tex'))


if __name__ == '__main__':
    main()
