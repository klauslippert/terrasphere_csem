#!/usr/bin/env python3
"""LaTeX report of the elfe3D validation: one section per case and profile (x, y), one subsection (page) per model
with terrain (hill, depression; the flat model is validated against empymod elsewhere): setup figure (plan view and
xz-section through y = 0 with source, profile, terrain and layers), then the fields on that profile: Re/Im of E and H,
all three components, series (lines) against elfe3D (circles), all frequencies with an elfe3D run.  Physical units (E in mV/m, H in uA/m, per 1 A); series exactly as written by this package, elfe3D converted to its
convention (H = -H_elfe3D).
Writes <validation folder>/report/{report.tex, fig/*.pdf}, or report_<H>m/ when only the heights +-H are wanted.
Usage: python3 report_elfe3d.py <validation folder (with cases/)> [H]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.abspath(os.path.join(HERE, '..', '..', '..')), os.path.dirname(HERE), HERE]   # package, plotting.py, this folder
import numpy as np
from terrasphere_csem.config import Config
from compare_elfe3d import model_name
from plot_profiles_freq import elfe_field
from plotting import setup_figure, profile_figure, tex_escape


def main():
    """Write the figures and report.tex for all cases (see the module docstring)."""
    root = os.path.abspath(sys.argv[1])
    only = float(sys.argv[2]) if len(sys.argv) > 2 else None           # only the heights +-only
    out = os.path.join(root, 'report' if only is None else f'report_{only:g}m'); os.makedirs(os.path.join(out, 'fig'), exist_ok=True)
    cases = sorted(c for c in os.listdir(os.path.join(root, 'cases'))
                   if os.path.isfile(os.path.join(root, 'cases', c, 'results', 'fields.npz')))
    body = []
    for c in cases:
        st = Config(os.path.join(root, 'cases', c))
        d = np.load(os.path.join(st.folder, 'results', 'fields.npz'))
        ser = np.concatenate([d['E'], d['H']], axis=2)                # (nh, nf, 6, P), as written by terrasphere_csem
        P = len(d['px'])
        desc = open(os.path.join(st.folder, 'control.toml')).readline().lstrip('# ').strip()
        for prof in d['profile'][np.sort(np.unique(d['profile'], return_index=True)[1])]:
            body.append(rf'\section{{{tex_escape(c)}, {prof}-profile}}' + '\n' + tex_escape(desc) + '\n')
            for ih, H in enumerate(d['heights']):
                if H == 0 or (only is not None and abs(H) != only):    # flat: validated against empymod
                    continue
                m = model_name(H)
                lab = f'hill {H:g} m' if H > 0 else f'depression {-H:g} m'
                setup, fp = f'fig/{c}_{prof}_{m}_setup.pdf', f'fig/{c}_{prof}_{m}.pdf'
                setup_figure(st, d, prof, H, os.path.join(out, setup))
                orig = profile_figure(st, d, ser, ih, prof, os.path.join(out, fp), P,
                                      ref=lambda f: elfe_field(st, H, f, P, 'out'))
                body.append(rf'\subsection{{{lab}}}' + '\n')
                body.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{%s}' % setup +
                            r'\caption{Setup of %s, %s-profile, %s: plan view (left) and xz-section at $y = 0$ (right).}'
                            % (tex_escape(c), prof, lab) + r'\end{figure}' + '\n')
                body.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{%s}' % fp +
                            r'\caption{%s, %s-profile, %s: series (lines) and elfe3D (circles, every second receiver). '
                            r'elfe3D runs: %s.}' % (tex_escape(c), prof, lab, tex_escape('; '.join(orig))) +
                            r'\end{figure}' + '\n' + r'\clearpage' + '\n')
            print(c, prof)
    hs = sorted({abs(float(h)) for h in d['heights'] if h != 0 and (only is None or abs(h) == only)})
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
Gaussian hills and depressions (height """ + ' m and '.join(r'$\pm%g$' % h for h in hs) + r""" m, width $s = 150$ m); the flat model is validated against
empymod.  Three layers with
interfaces at 200 m and 600 m depth.  Receivers 1 cm below the surface on the x- and y-profile from $-600$ m to
600 m every 20 m.  Fields per 1 A source current, E in mV/m, H in $\mu$A/m, time dependence $e^{-i\omega t}$,
in the convention of this package: the series exactly as it writes them, elfe3D converted ($H = -H_\mathrm{elfe3D}$).
\tableofcontents
\clearpage
""" + '\n'.join(body) + r"""
\end{document}
"""
    open(os.path.join(out, 'report.tex'), 'w').write(tex)
    print(os.path.join(out, 'report.tex'))


if __name__ == '__main__':
    main()
