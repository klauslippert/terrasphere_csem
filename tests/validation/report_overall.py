#!/usr/bin/env python3
"""Overall validation report: cover page, table of contents and the references of the codes compared with (empymod;
elfe3D with TetGen and MUMPS), followed by the three reports as they are: 1D/empymod/report/report.pdf (flat model
against empymod), 3D/elfe3d_topo/report/report.pdf (topography against elfe3D), 3D/elfe3d_sphere/report/report.pdf
(sphere under the hill against elfe3D); build them first.
Writes report/report.tex next to this script; compile it there (tectonic or pdflatex, needs pdfpages).
Usage: python3 report_overall.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = [('Flat model against empymod (1-D)', '1D/empymod'),
           ('Topography against elfe3D', '3D/elfe3d_topo'),
           ('Sphere under the hill against elfe3D', '3D/elfe3d_sphere')]

TEX = r"""\documentclass[11pt,a4paper]{article}
\usepackage[hmargin=2cm,vmargin=1.5cm]{geometry}
\usepackage{pdfpages}\usepackage[hidelinks]{hyperref}
\title{terrasphere\_csem: validation}
\date{\today}
\begin{document}
\maketitle
\thispagestyle{empty}
Validation of the series solution of this package against two independent codes: the 1-D code empymod for the flat
model, and the 3-D finite-element code elfe3D for Gaussian topography (hills and valleys of $\pm30$ m and $\pm90$ m,
width 150 m) and for a conductive sphere under the 90 m hill.  Four sources (grounded wire in line and broadside,
offset loop, large loop around the hill), three layers with interfaces at 200 m and 600 m depth, receivers on the
x- and y-profile from $-600$ m to 600 m every 20 m, frequencies 0.1 Hz, 10 Hz and 1 kHz.  The three reports follow
in full; each ends with its table of deviations.

\tableofcontents

\section*{References}
\addcontentsline{toc}{section}{References}
\paragraph{empymod} (flat model; version 2.6.0, quasi-static, digital filter key\_201\_2009)
\begin{itemize}
\item Werthm\"uller, D. (2017). An open-source full 3D electromagnetic modeler for 1D VTI media in Python: empymod.
  \emph{Geophysics}, 82(6), WB9--WB19. \href{https://doi.org/10.1190/geo2016-0626.1}{doi:10.1190/geo2016-0626.1}
\item Key, K. (2009). 1D inversion of multicomponent, multifrequency marine CSEM data: Methodology and synthetic
  studies for resolving thin resistive layers. \emph{Geophysics}, 74(2), F9--F20.
  \href{https://doi.org/10.1190/1.3058434}{doi:10.1190/1.3058434} (Hankel filters)
\end{itemize}
\paragraph{elfe3D} (topography and sphere; \url{https://github.com/emsig/elfe3d}, v1.0.1 plus 16 commits, commit
58528bd7 of 21 July 2026; Zenodo concept DOI \href{https://doi.org/10.5281/zenodo.13309721}{10.5281/zenodo.13309721})
\begin{itemize}
\item Rulff, P. (2025). elfe3D v1.0.1: Modelling with the total electric field approach using finite elements in
  3D. \emph{Journal of Open Source Software}, 10(110), 7949.
  \href{https://doi.org/10.21105/joss.07949}{doi:10.21105/joss.07949}
\item Rulff, P., Buntin, L. M., Kalscheuer, T. (2021). Efficient goal-oriented mesh refinement in 3-D
  finite-element modelling adapted for controlled source electromagnetic surveys. \emph{Geophysical Journal
  International}, 227, 1624--1645. \href{https://doi.org/10.1093/gji/ggab264}{doi:10.1093/gji/ggab264}
\end{itemize}
\paragraph{TetGen} (mesh generator used by elfe3D; version 1.5.0)
\begin{itemize}
\item Si, H. (2015). TetGen, a Delaunay-based quality tetrahedral mesh generator. \emph{ACM Transactions on
  Mathematical Software}, 41(2), 11:1--11:36. \href{https://doi.org/10.1145/2629697}{doi:10.1145/2629697}
\end{itemize}
\paragraph{MUMPS} (direct solver used by elfe3D; version 5.8.2, sequential, with the orderings METIS 5.1.0 and
SCOTCH 7.0.11 and OpenBLAS 0.3.34)
\begin{itemize}
\item Amestoy, P. R., Duff, I. S., L'Excellent, J.-Y., Koster, J. (2001). A fully asynchronous multifrontal solver
  using distributed dynamic scheduling. \emph{SIAM Journal on Matrix Analysis and Applications}, 23(1), 15--41.
  \href{https://doi.org/10.1137/S0895479899358194}{doi:10.1137/S0895479899358194}
\item Amestoy, P. R., Buttari, A., L'Excellent, J.-Y., Mary, T. (2019). Performance and scalability of the block
  low-rank multifrontal factorization on multicore architectures. \emph{ACM Transactions on Mathematical
  Software}, 45(1), 2:1--2:26. \href{https://doi.org/10.1145/3242094}{doi:10.1145/3242094}
\end{itemize}
\clearpage
"""


def main():
    """Write report/report.tex (see the module docstring)."""
    out = os.path.join(HERE, 'report'); os.makedirs(out, exist_ok=True)
    for _, f in REPORTS:
        assert os.path.isfile(os.path.join(HERE, f, 'report', 'report.pdf')), f'build {f}/report first'
    inc = ''.join(r'\includepdf[pages=-,addtotoc={1,section,1,%s,%s}]{../%s/report/report.pdf}' % (t, f.replace('/', '_'), f)
                  + '\n' for t, f in REPORTS)
    open(os.path.join(out, 'report.tex'), 'w').write(TEX + inc + r'\end{document}' + '\n')
    print(os.path.join(out, 'report.tex'))


if __name__ == '__main__':
    main()
