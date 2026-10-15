#!/usr/bin/env python3
"""LaTeX report of the flat-model comparison with empymod: one section per case and profile (x, y) with the setup
figure (plan view and xz-section through y = 0) and the fields on that profile (Re/Im of E and H, all three
components, series (lines) against empymod (circles), all frequencies), then a table of the deviations
(compare_empymod.py).  Physical units (E in mV/m, H in uA/m, per 1 A), H with the sign of this package.
Writes <validation folder>/report/{report.tex, fig/*.pdf}.
Usage: python3 report_empymod.py <validation folder (with cases/)>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.abspath(os.path.join(HERE, '..', '..', '..')), os.path.dirname(HERE), HERE]   # package, plotting.py, this folder
import numpy as np
from terrasphere_csem.config import Config
from plotting import setup_figure, profile_figure, tex_escape
from compare_empymod import load, deviations, COMPONENTS


def main():
    """Write the figures and report.tex for all cases (see the module docstring)."""
    root = os.path.abspath(sys.argv[1])
    out = os.path.join(root, 'report'); os.makedirs(os.path.join(out, 'fig'), exist_ok=True)
    cases = sorted(c for c in os.listdir(os.path.join(root, 'cases'))
                   if os.path.isfile(os.path.join(root, 'cases', c, 'empymod', 'out', 'fields.npz')))
    body, rows = [], []
    for c in cases:
        st = Config(os.path.join(root, 'cases', c))
        ser, emp, fr, d = load(st.folder)
        ih = list(d['heights']).index(0.0)
        P = len(d['px'])
        S = np.zeros((len(d['heights']),) + ser.shape, complex); S[ih] = ser
        desc = open(os.path.join(st.folder, 'control.toml')).readline().lstrip('# ').strip()
        for prof in d['profile'][np.sort(np.unique(d['profile'], return_index=True)[1])]:
            setup, fp = f'fig/{c}_{prof}_setup.pdf', f'fig/{c}_{prof}.pdf'
            setup_figure(st, d, prof, 0.0, os.path.join(out, setup))
            profile_figure(st, d, S, ih, prof, os.path.join(out, fp), P,
                           ref=lambda f: (emp[list(fr).index(f)], 'empymod'))
            body.append(rf'\section{{{tex_escape(c)}, {prof}-profile}}' + '\n' + tex_escape(desc) + '\n')
            body.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{%s}' % setup +
                        r'\caption{Setup of %s, %s-profile, flat model: plan view (left) and xz-section at $y = 0$ '
                        r'(right).}' % (tex_escape(c), prof) + r'\end{figure}' + '\n')
            body.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{%s}' % fp +
                        r'\caption{%s, %s-profile, flat model: series (lines) and empymod (circles, every second '
                        r'receiver).}' % (tex_escape(c), prof) + r'\end{figure}' + '\n' + r'\clearpage' + '\n')
            print(c, prof)
        dv = deviations(ser, emp)
        for i, f in enumerate(fr):
            rows.append(f'{tex_escape(c)} & {f:g} & ' + ' & '.join(
                ('%.0e' % dv[(i, k)][0]) if k == 'Ez' else ('%.0e / %.0e' % (dv[(i, k)][0], dv[(i, k)][2]))
                for k in COMPONENTS) + r' \\')
    table = (r'\section{Deviations}' + '\n' +
             r'$|V_\mathrm{series} - V_\mathrm{empymod}| / \max|V_\mathrm{empymod}|$ over both profiles, median / '
             r'maximum; $E_z$: $\max|E_{z,\mathrm{empymod}}| / \max|E|$ (zero on the flat surface; empymod 1 mm '
             r'below it).' + '\n' + r'\begin{center}\small\begin{tabular}{llcccccc}' + '\n' +
             r'case & $f$ (Hz) & ' + ' & '.join(f'${k[0]}_{k[1]}$' for k in COMPONENTS) + r' \\ \hline' + '\n' +
             '\n'.join(rows) + '\n' + r'\end{tabular}\end{center}' + '\n')
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[hmargin=2cm,vmargin=1.5cm]{geometry}
\usepackage[font=small,skip=2pt]{caption}
\setlength{\textfloatsep}{6pt}\setlength{\intextsep}{6pt}
\usepackage{graphicx,float}
\title{terrasphere\_csem: flat model against empymod}
\date{\today}
\begin{document}
\maketitle
Series solution of this package for the flat model (lines) against the 1-D code empymod (circles) for the four
sources of the elfe3D comparison.  Three layers with interfaces at 200 m and 600 m depth.  Receivers on the x- and
y-profile from $-600$ m to 600 m every 20 m: series on the surface (earth side), empymod sources and receivers 1 mm
below it.  empymod quasi-static (no displacement currents), digital filter key\_201\_2009, finite bipoles (40 Gauss
points each; a loop as its four sides).  Fields per 1 A source current, E in mV/m, H in $\mu$A/m, time dependence
$e^{-i\omega t}$.  Components that vanish on a profile by symmetry are plotted as they come out (noise
below $10^{-10}$ of the field maximum in both codes; a power of ten of an axis is part of its label).
\tableofcontents
\clearpage
""" + '\n'.join(body) + table + r"""
\end{document}
"""
    open(os.path.join(out, 'report.tex'), 'w').write(tex)
    print(os.path.join(out, 'report.tex'))


if __name__ == '__main__':
    main()
