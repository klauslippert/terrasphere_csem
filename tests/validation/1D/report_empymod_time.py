#!/usr/bin/env python3
"""LaTeX report of the time-domain flat-model comparison with empymod: one section per case and profile (x, y) with
the setup figure (plan view and xz-section through y = 0) and the transients on that profile (|V| of E_x, E_y and
dB/dt x, y, z against time, series (lines) against empymod (circles), every station), then the tables of the
deviations
(compare_empymod_time.py).  Physical units (E in mV/m, dB/dt in V/m^2 = T/s, per 1 A step), signs of this package.
Writes <validation folder>/report/{report.tex, fig/*.pdf}.
Usage: python3 report_empymod_time.py <validation folder (with cases/)>"""
import os, sys, tomllib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.abspath(os.path.join(HERE, '..', '..', '..')), os.path.dirname(HERE), HERE]   # package, plotting.py, this folder
import numpy as np
from terrasphere_csem.config import Config
from plotting import plt, setup_figure, tex_escape, COLORS, factor_into_labels
from compare_empymod_time import load, deviations, COMPONENTS, ZERO

LABELS = {'Ex': '$E_x$', 'Ey': '$E_y$', 'dBx/dt': r'$\partial_t B_x$', 'dBy/dt': r'$\partial_t B_y$',
          'dBz/dt': r'$\partial_t B_z$'}


def transient_figure(ser, emp, t, d, prof, path):
    """Transients on profile prof, E left (E_x, E_y; the legend below), dB/dt right (x, y, z): |V| against time
    (solid/open circles where V > 0, dashed/filled circles where V < 0).  Stations where the component vanishes by
    symmetry (compare_empymod_time.ZERO) are left out; a component that vanishes on the whole profile is shown as it
    comes out of both codes: V itself, linear (the rounding noise, or a note where both codes give exactly 0)."""
    sel = np.where(d['profile'] == prof)[0]
    pos = (d['px'] if prof == 'x' else d['py'])[sel]
    fig, ax = plt.subplots(3, 2, figsize=(10, 6.6), sharex=True)
    place = {'Ex': ax[0, 0], 'Ey': ax[1, 0], 'dBx/dt': ax[0, 1], 'dBy/dt': ax[1, 1], 'dBz/dt': ax[2, 1]}
    left = []
    for field, k, name in COMPONENTS:
        a1 = place[name]
        sc = 1e3 if field == 'E' else 1.0
        ref = np.abs(emp[field][:, :2 if field == 'E' else 3]).max()
        unit = 'mV/m' if field == 'E' else r'V/m$^2$'
        zero = np.abs(emp[field][:, k, sel]).max(0) < ZERO * ref
        for js, (p, x) in enumerate(zip(sel, pos)):
            a, b = sc * ser[field][:, k, p], sc * emp[field][:, k, p]
            c = COLORS[js % len(COLORS)]
            m = np.arange(len(t)) % 4 == js % 4                       # every fourth time, staggered per station
            if zero.all():                                            # vanishes on the whole profile: as computed
                a1.semilogx(t, a, '-', color=c, lw=1.0, label=f'{prof} = {x:g} m')
                a1.semilogx(t[m], b[m], 'o', ms=3, mfc='none', color=c)
                continue
            if zero[js]:
                left.append((name, x)); continue
            a1.loglog(t, np.where(a > 0, a, np.nan), '-', color=c, lw=1.1, label=f'{prof} = {x:g} m')
            a1.loglog(t, np.where(a < 0, -a, np.nan), '--', color=c, lw=1.1)
            a1.loglog(t[m & (b > 0)], b[m & (b > 0)], 'o', ms=3, mfc='none', color=c)
            a1.loglog(t[m & (b < 0)], -b[m & (b < 0)], 'o', ms=3, color=c)
        if zero.all():
            a1.set_ylabel(f'{LABELS[name]} ({unit})', fontsize=8.5)
            a1.set_title(f'{LABELS[name]}: zero by symmetry, as computed (linear)', fontsize=9)
            if not (np.any(ser[field][:, k, sel]) or np.any(emp[field][:, k, sel])):
                a1.set_ylim(-1, 1)
                a1.text(0.5, 0.75, 'exactly 0 in both codes', transform=a1.transAxes, ha='center', va='center',
                        fontsize=8.5, color='0.3')
        else:
            a1.set_ylabel(f'|{LABELS[name]}| ({unit})', fontsize=8.5)
            a1.set_title(f'{LABELS[name]}', fontsize=9)
        a1.tick_params(labelsize=7.5); a1.grid(True, which='major', color='0.9', lw=0.5)
    ax[1, 0].tick_params(labelbottom=True); ax[1, 0].set_xlabel('t (s)', fontsize=8.5)
    ax[2, 1].set_xlabel('t (s)', fontsize=8.5)
    h, l = place['dBz/dt'].get_legend_handles_labels()                 # dB/dt z never vanishes on both profiles
    if not l:
        h, l = place['Ex'].get_legend_handles_labels()
    lg = ax[2, 0]; lg.axis('off')
    h += [plt.Line2D([], [], color='k', ls='-'), plt.Line2D([], [], color='k', ls='--'),
          plt.Line2D([], [], color='k', marker='o', ls='', ms=3, mfc='none'), plt.Line2D([], [], color='k', marker='o', ls='', ms=3)]
    l += ['series, V > 0', 'series, V < 0', 'empymod, V > 0', 'empymod, V < 0']
    lg.legend(h, l, loc='center', ncol=2, fontsize=8, frameon=False)
    factor_into_labels(fig)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return left


def main():
    """Write the figures and report.tex for all cases (see the module docstring)."""
    root = os.path.abspath(sys.argv[1])
    out = os.path.join(root, 'report'); os.makedirs(os.path.join(out, 'fig'), exist_ok=True)
    cases = sorted(c for c in os.listdir(os.path.join(root, 'cases'))
                   if os.path.isfile(os.path.join(root, 'cases', c, 'empymod', 'out', 'transients.npz')))
    body, rows, drows = [], [], []
    for c in cases:
        st = Config(os.path.join(root, 'cases', c))
        ser, emp, t = load(st.folder)
        d = np.load(os.path.join(st.folder, 'results', 'transients.npz'))
        desc = open(os.path.join(st.folder, 'control.toml')).readline().lstrip('# ').strip()
        for prof in d['profile'][np.sort(np.unique(d['profile'], return_index=True)[1])]:
            setup, fp = f'fig/{c}_{prof}_setup.pdf', f'fig/{c}_{prof}.pdf'
            setup_figure(st, d, prof, 0.0, os.path.join(out, setup))
            left = transient_figure(ser, emp, t, d, prof, os.path.join(out, fp))
            gone = {}
            for name, x in left:
                gone.setdefault(name, []).append(f'{x:g}')
            note = ('; single stations left out (zero by symmetry): ' + ', '.join(f'{LABELS[n]} at {prof} = {", ".join(v)} m'
                                                                   for n, v in gone.items())) if gone else ''
            body.append(rf'\section{{{tex_escape(c)}, {prof}-profile}}' + '\n' + tex_escape(desc) + '\n')
            body.append(r'\begin{figure}[H]\centering\includegraphics[width=0.72\textwidth]{%s}' % setup +
                        r'\caption{Setup of %s, %s-profile, flat model: plan view (left) and xz-section at $y = 0$ '
                        r'(right).}' % (tex_escape(c), prof) + r'\end{figure}' + '\n')
            body.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{%s}' % fp +
                        r'\caption{%s, %s-profile, flat model, switch-off of 1 A: series (lines) and empymod (circles, '
                        r'every fourth time), one colour per station; left $E$, right $\partial_t B$.  A component that vanishes on the whole '
                        r'profile by symmetry is shown as computed: $V$ itself, linear%s.  Deviations: '
                        r'section~\ref{sec:dev}.}'
                        % (tex_escape(c), prof, note) + r'\end{figure}' + '\n' + r'\clearpage' + '\n')
            print(c, prof)
        dv = deviations(ser, emp, t)
        decs = sorted(next(iter(dv.values()))['decades'])
        for name, v in dv.items():
            z = '--' if v['zero'] is None else '%.0e' % v['zero']
            rows.append(f'{tex_escape(c)} & {LABELS[name]} & {v["n"]} & %.1e & %.1e & %.1e & {z}' %
                        (np.median(v['curve']), v['curve'].max(), v['pointwise'].max()) + r' \\')
            drows.append(f'{tex_escape(c)} & {LABELS[name]} & ' + ' & '.join('%.0e' % v['decades'][q] for q in decs) + r' \\')
        rows.append(r'\hline'); drows.append(r'\hline')
    table = (r'\section{Deviations}\label{sec:dev}' + '\n' +
             r'Per receiver: curve $= \max_t|V_\mathrm{series} - V_\mathrm{empymod}| / \max_t|V_\mathrm{empymod}|$, '
             r'median and maximum over the $n$ receivers of both profiles; pointwise $= \max|\Delta V| / '
             r'|V_\mathrm{empymod}|$ over the times where $|V_\mathrm{empymod}|$ exceeds $10^{-6}$ of its maximum; '
             r'zero: largest $\max_t|V|$ of both codes on the receivers where the component vanishes by symmetry '
             r'(below $10^{-6}$ of the largest $E$ resp.\ $\partial_t B$ component), relative to that component; '
             r'these receivers are left out of the other measures.' + '\n' +
             r'\begin{center}\small\begin{tabular}{llrcccc}' + '\n' +
             r'case & component & $n$ & curve median & curve max & pointwise & zero \\ \hline' + '\n' +
             '\n'.join(rows) + '\n' + r'\end{tabular}\end{center}' + '\n' +
             r'Curve measure, maximum over the receivers, per decade of time:' + '\n' +
             r'\begin{center}\small\begin{tabular}{ll' + 'c' * len(decs) + '}' + '\n' +
             r'case & component & ' + ' & '.join(r'$10^{%d}$\,s' % q for q in decs) + r' \\ \hline' + '\n' +
             '\n'.join(drows) + '\n' + r'\end{tabular}\end{center}' + '\n')
    inp = tomllib.load(open(os.path.join(root, 'cases', cases[0], 'empymod', 'in', 'empymod_input.toml'), 'rb'))
    tm, nu = inp['time']['times'], inp['numerics']
    ver = [l for l in open(os.path.join(root, 'cases', cases[0], 'empymod', 'out', 'log.txt')) if 'empymod' in l][0]
    ver = ver.split('empymod ')[1].split(';')[0]
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[hmargin=2cm,vmargin=1.5cm]{geometry}
\usepackage[font=small,skip=2pt]{caption}
\setlength{\textfloatsep}{6pt}\setlength{\intextsep}{6pt}
\usepackage{graphicx,float}
\title{terrasphere\_csem: flat model against empymod in the time domain}
\date{\today}
\begin{document}
\maketitle
Transients of the series solution of this package for the flat model (lines) against the 1-D code empymod
(circles) for the four sources of the elfe3D comparison.  Three layers with interfaces at 200 m and 600 m depth.
Switch-off of 1 A, %d times from %.3g s to %.3g s (the lagged time grid of this package).  Receivers on the x- and
y-profile from $-600$ m to 600 m every 300 m: series on the surface (earth side), empymod sources and receivers
$10^{-6}$ m below it.  Components $E_x$, $E_y$, $\partial_t B_x$, $\partial_t B_y$, $\partial_t B_z$; $E_z$
vanishes on the earth side of the flat surface and is not compared.  empymod %s, quasi-static (no displacement
currents); Hankel transform the digital filter %s in its standard form, Fourier transform the digital filter %s
in its standard form (every time its own frequencies, no interpolation); every bipole an explicit sum of %d
Gauss--Legendre point dipoles, a loop as its four sides.  $E$ in mV/m, $\partial_t B$ in V/m$^2$ (= T/s), all per 1 A.
\tableofcontents
\clearpage
""" % (len(tm), tm[0], tm[-1], ver, tex_escape(nu['dlf']), tex_escape(nu['ft_dlf']), inp['source']['gauss'])
    tex += '\n'.join(body) + table + r"""
\end{document}
"""
    open(os.path.join(out, 'report.tex'), 'w').write(tex)
    print(os.path.join(out, 'report.tex'))


if __name__ == '__main__':
    main()
