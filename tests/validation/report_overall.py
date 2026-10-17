#!/usr/bin/env python3
"""Overall validation report: cover page, table of contents, the conventions of the comparisons (time convention,
frame and source normalisation of empymod and elfe3D against this package) and the references of the codes compared
with (empymod; elfe3D with TetGen and MUMPS), followed by the four reports as they are: 1D/empymod/report/report.pdf (flat model
against empymod), 1D/empymod_time/report/report.pdf (the same in the time domain), 3D/elfe3d_topo/report/report.pdf
(topography against elfe3D), 3D/elfe3d_sphere/report/report.pdf (sphere under the hill against elfe3D); build them
first.
Writes report/report.tex next to this script; compile it there (tectonic or pdflatex, needs pdfpages) and copy
report/report.pdf to validation_report.pdf in the repository root, the only tracked copy (the report folders are not
tracked).
Usage: python3 report_overall.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = [('Flat model against empymod (1-D)', '1D/empymod'),
           ('Flat model against empymod in the time domain (1-D)', '1D/empymod_time'),
           ('Topography against elfe3D', '3D/elfe3d_topo'),
           ('Sphere under the hill against elfe3D', '3D/elfe3d_sphere')]

TEX = r"""\documentclass[11pt,a4paper]{article}
\usepackage[hmargin=2cm,vmargin=1.5cm]{geometry}
\usepackage{amsmath}\usepackage{pdfpages}\usepackage[hidelinks]{hyperref}
\title{terrasphere\_csem: validation}
\date{\today}
\begin{document}
\maketitle
\thispagestyle{empty}
Validation of the series solution of this package against two independent codes: the 1-D code empymod for the flat
model, and the 3-D finite-element code elfe3D for Gaussian topography (hills and depressions of $\pm30$ m and $\pm90$ m,
width 150 m) and for a conductive sphere under the 90 m hill.  Four sources (grounded wire in line and broadside,
offset loop, large loop around the hill), three layers with interfaces at 200 m and 600 m depth, receivers on the
x- and y-profile from $-600$ m to 600 m every 20 m, frequencies 0.1 Hz, 10 Hz and 1 kHz; the flat model also in
the time domain (switch-off, 10\,$\mu$s to 10 s, every 300 m).  The four reports follow in full; each ends with its
table of deviations.

\tableofcontents

\section*{Conventions}
\addcontentsline{toc}{section}{Conventions}
Time dependence of this package: $e^{-i\omega t}$; frame $(x,y,z)$ right-handed with $z$ up; quasi-static (no
displacement currents).  The conversions below are applied in all comparisons of this report and of the paper.

\subsection*{empymod}
empymod includes displacement currents by default (relative permittivity 1 in every layer), whereas this package is
quasi-static; the two then differ by about $\omega\epsilon_0/\sigma$ ($10^{-5}$ at 1\,kHz in 0.005\,S/m), which first
showed up as a frequency-proportional residual of the first-order null test.  All comparisons use empymod with
\texttt{epermH = epermV = 0}.  In that mode its air layer is a near-perfect insulator, and a source or receiver
lying exactly on an interface, which empymod assigns to the layer above (for $z=0$: the air), is moved 1\,$\mu$m into
the layer below, where this package places it; the displacement itself changes the fields by about $10^{-8}$, which
sets the level of the 1-D comparisons with surface sources (fields in the ground from a source in the air are
unaffected, so the Born test uses no displacement).  Further, empymod uses $e^{+i\omega t}$ and a depth frame ($z$
down) that is left-handed with respect to our $z$-up frame, follows Hunziker et al.\ (2015) in normalising magnetic
sources to a unit magnetic current moment\footnote{Hunziker, Thorbecke \& Slob (2015), Geophysics 80(1), F1--F18,
p.~F2: ``The configuration has an electric or magnetic current type source located in a layer of finite
thickness''; the Green's functions are the fields of a unit point source of either type.}, and defines its vertical
magnetic dipole (\texttt{dip=90}) pointing down.  Established against closed-form full-space dipole fields and used
in all tests:
\begin{equation*}
\mathbf E_{\mathrm{ours}}=f\,\overline{\mathbf E}_{\mathrm{emp}}\circ(1,1,-1),\qquad
\mathbf H_{\mathrm{ours}}=-f\,\overline{\mathbf H}_{\mathrm{emp}}\circ(1,1,-1),\qquad
f=\begin{cases}1&\text{HED}\\ i\omega\mu_0&\text{HMD}\\-i\omega\mu_0&\text{VMD (our moment up)}\end{cases}
\end{equation*}
where the bar is the complex conjugate and $\circ$ the component-wise product (polar vectors flip their
$z$-component under the frame change, axial vectors their horizontal components, which gives the extra sign of
$\mathbf H$).  For loop sources at the surface empymod places the wire in the air; for a horizontal electric dipole
exactly on the surface we place it in the ground, and the two agree to $10^{-6}$.  A receiver on the wire of a
coincident loop cannot be evaluated by empymod (the segment ends produce a static near field), and its Fourier route
loses accuracy at late times; Key's 401-point Hankel filter with exact frequencies was used for the comparisons at
the $10^{-4}$ level.  With its default settings (displacement currents, grounded wire in the air layer) empymod
describes a different problem from the quasi-static grounded wire used here: its galvanic electric field differs
below 0.1\,Hz (by a factor 20 at $10^{-4}$\,Hz) and its switch-off response is not comparable; quasi-static and with
the wire in the ground it agrees with ours to $10^{-6}$ down to $10^{-4}$\,Hz.

\subsection*{elfe3D}
elfe3D was used in the version v1.0.1 plus 16 commits, commit 58528bd7 of its public repository (\texttt{git
describe}: v1.0.1-16-g58528bd), compiled with gfortran against MUMPS~5.8.2 and meshed with TetGen~1.5.0, the version
recommended by its authors.  The build reproduces the example shipped with the repository (line source over a
half-space with a conductive block, three frequencies, 73 receivers) to $4\times10^{-6}$ of the profile maximum.  The
repository example uses ``$z$ positive upwards''\footnote{elfe3D, \texttt{elfe3D/in/README.md} (commit 58528bd7):
``modelling domain: 30 x 30 x 30 km / z positive upwards''.}, which coincides with our component frame; receiver
positions are therefore entered as $z=-d$.  The time convention was determined by running the same example with the
block set to the host resistivity, a homogeneous half-space that this package computes exactly: the ratio of the two
electric fields is real and independent of frequency (0.965 at 1, 10 and 100\,Hz on the unrefined example mesh),
whereas the ratio to the complex conjugate acquires a phase that grows with frequency ($19^\circ$ at 100\,Hz).  The
output of elfe3D thus follows $e^{-i\omega t}$, as this package.  This agrees with the system matrix in the code,
$\mu^{-1}K-i\omega\rho^{-1}M$ with right-hand side $+i\omega$ times the source term\footnote{elfe3D commit 58528bd7,
\texttt{elfe3d.f90}, line 594: \texttt{AA = cmplx((1/mu(l))*K, -w*(1/rho(l))*MM, kind=dp)}; line 659:
\texttt{Bg = cmplx(0.0\_dp, (edl*source*w*p)/(source\_length),kind=dp)}.}, i.e.\
$\nabla\times\mu^{-1}\nabla\times\mathbf E-i\omega\sigma\mathbf E=i\omega\mathbf J$, although the thesis of the main
author formulates the method with $e^{+i\omega t}$\footnote{Rulff (2023), PhD thesis, Uppsala University, p.~27:
``Note, that the BVP and all subsequent equations are formulated with a time dependency of e$^{-i\omega t}$ in
Paper I, whereas the implementation had to be changed to a time dependency of e$^{i\omega t}$ at a later stage
(Papers III \& IV) for compatibility reasons.''}.  The magnetic field, however, is formed with the $e^{+i\omega t}$
form of Faraday's law, $\mathbf H=-(i\omega\mu)^{-1}\nabla\times\mathbf E$\footnote{elfe3D commit 58528bd7,
\texttt{mod\_calculate\_tf.f90}, line 129: \texttt{factor = cmplx(0.0,(1/(w*mu(rec1\_el))), kind=dp)}, applied to
$\sum_j E_j\nabla\times\mathbf N_j$; the comment reads ``H\_rec1 = -1/(iwmu)sum(Ej curl (Nj)''.}, so that the output
$\mathbf H$ carries the opposite sign of the field belonging to the output $\mathbf E$; in the half-space test all
three magnetic components have ratio $-1$ and all electric components $+1$, which rules out a frame reflection (that
would flip $E_z$ and $H_z$ against the horizontal components).  Our own $\mathbf H$ satisfies
$\nabla\times\mathbf E=i\omega\mu_0\mathbf H$ to $10^{-4}$ in a finite-difference check.  Finally, elfe3D distributes
the source current as source moment divided by the total length of the source edges; with the moment set to current
times wire length (or loop perimeter) both codes carry the same current.  Hence
\begin{equation*}
\mathbf E_{\mathrm{elfe3D}}=\mathbf E_{\mathrm{ours}},\qquad \mathbf H_{\mathrm{elfe3D}}=-\mathbf H_{\mathrm{ours}},
\end{equation*}
without complex conjugation and without reflection.

Two properties of the source definition matter for loops.  elfe3D marks a mesh edge as a source edge if both of its
end points lie on a segment of the source polygon to within the machine epsilon, $2.2\times10^{-16}$\,m in absolute
terms\footnote{elfe3D commit 58528bd7, \texttt{mod\_calculate\_global\_source.f90}, function \texttt{dist2segment}
(distance via a cross product compared with \texttt{eps\_dp}, then the exact test \texttt{p1p2 .ne. p1cp + p2cp}) and
\texttt{mod\_constant.f90}, line 50: \texttt{eps\_dp = abs(epsilon(1.0\_dp))}.}.  With the optimisation switches of the
distributed makefile (\texttt{-Ofast -mfma}) this test is not exact for oblique segments: a copy of the routine
compiled with these switches finds only 52 of the 64 sides of a 64-gon, compiled with \texttt{-O2} all of them.  Our
sources are therefore axis-parallel chains of 10\,m segments with integer corner coordinates, for which all products
are exact: the wires and the two rectangular loops of the validation cases.  Further, for the horizontal segmented
loop (source type 7) the code runs over the segments 1 to $n-1$ only; the closing segment from the last to the first
corner is added for the vertical loop (type 9) alone\footnote{elfe3D commit 58528bd7,
\texttt{mod\_calculate\_global\_source.f90}, subroutine \texttt{segmented\_source}: case(7) loops \texttt{do i =
1,num\_segments-1} with no closing segment; case(9) ends with ``! last segment to close the loop''.}.  We therefore
repeat the first corner at the end of the corner list; an open loop would carry charges at its gap whose galvanic
field dominates the horizontal components.  With both measures elfe3D reports source lengths of 1600\,m and
10\,400\,m in all runs, the perimeters of the two loops.

\section*{References}
\addcontentsline{toc}{section}{References}
\paragraph{empymod} (flat model; version 2.6.0, quasi-static, digital filter key\_201\_2009; time domain: Fourier
filter key\_601\_2009)
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
