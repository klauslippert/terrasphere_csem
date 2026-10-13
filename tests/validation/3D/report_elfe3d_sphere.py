#!/usr/bin/env python3
"""Sphere comparison elfe3D <-> series (tests/validation/3D/elfe3d_sphere): per case the elfe3D run with the sphere
(<model>_<band>) and the run on the same mesh with the host resistivity in the sphere (<model>ref_<band>, written by
elfe3d_input.py --reference) against the series with the sphere (<case>/results) and without it (the same case in
<elfe3d folder>/cases/<case>/results).  Measures per frequency and component (both profiles together):
  total field  |V_elfe - V_series| / max|V_series|                    median / 90th percentile
  anomaly      A = V(sphere) - V(no sphere); elfe3D: sphere run - reference run (the mesh error largely cancels)
               max|A_elfe - A_series| / max|A_series| and the size of the anomaly max|A_series| / max|V_series|
Writes <folder>/cases/<case>/comparison/elfe3d_sphere.txt and the report <folder>/report/{report.tex, fig/*.pdf}
(setup, total field and anomaly per case and profile; E in nV/m, H in uA/m; series exactly as written by this package, elfe3D converted to its convention, H = -H_elfe3D).
Usage: python3 report_elfe3d_sphere.py <folder (with cases/)> [<elfe3d folder without sphere>, default ../elfe3d_topo]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.abspath(os.path.join(HERE, '..', '..', '..')), os.path.dirname(HERE), HERE]   # package, plotting.py, this folder
import numpy as np
from terrasphere_csem.config import Config
from compare_elfe3d import model_name, is_finished, read_run, COMPONENTS, TO_OURS
from plotting import setup_figure, profile_figure, tex_escape

MIN_ANOMALY = 0.01          # anomalies below 1 % of max|V| are listed but not rated
ZERO = 1e-6                 # a component below 1e-6 of the largest E (H) component vanishes in the series: not rated


def elfe_run(st, name, f, P):
    """elfe3D fields (6, P) of run <name>_<band> at frequency f (nf below 100 Hz, else hf1k) in the convention of this
    package (H = -H_elfe3D), or None."""
    run = os.path.join(st.folder, 'elfe3d', f'{name}_{"nf" if f < 100 else "hf1k"}', 'out')
    fn = os.path.join(run, 'magnetic_fields_receiver_line.txt')
    if not (os.path.isfile(fn) and os.path.getsize(fn) > 0 and is_finished(run)):
        return None
    fe, V = read_run(run, P)
    ok = [i for i in range(len(fe)) if abs(fe[i] / f - 1) < 1e-6 and np.any(V[i])]
    return TO_OURS[:, None] * V[ok[-1]].T if ok else None


def main():
    """Compare and write the report (see the module docstring)."""
    root = os.path.abspath(sys.argv[1])
    base = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(os.path.dirname(root), 'elfe3d_topo')
    out = os.path.join(root, 'report'); os.makedirs(os.path.join(out, 'fig'), exist_ok=True)
    cases = sorted(c for c in os.listdir(os.path.join(root, 'cases'))
                   if os.path.isfile(os.path.join(root, 'cases', c, 'results', 'fields.npz')))
    body, rows = [], []
    for c in cases:
        st = Config(os.path.join(root, 'cases', c))
        d = np.load(os.path.join(st.folder, 'results', 'fields.npz'))
        b = np.load(os.path.join(base, 'cases', c, 'results', 'fields.npz'))
        H = float(d['heights'][0]); jb = list(b['heights']).index(H)
        assert len(d['heights']) == 1 and np.allclose(d['freqs'], b['freqs']) and np.allclose(d['px'], b['px'])
        P, fr, m = len(d['px']), d['freqs'], model_name(H)
        ser = np.concatenate([d['E'], d['H']], axis=2)                  # (1, nf, 6, P), as written by terrasphere_csem
        ser0 = np.concatenate([b['E'][jb:jb + 1], b['H'][jb:jb + 1]], axis=2)
        ans = ser - ser0                                                # series anomaly
        el = {f: (elfe_run(st, m, f, P), elfe_run(st, m + 'ref', f, P)) for f in fr}
        lines = [f'# {c}: elfe3D with sphere ({m}) and reference ({m}ref, same mesh) vs series with / without sphere',
                 '# total: |V_elfe - V_series| / max|V_series|, median/90th percentile (%);  anomaly: '
                 'max|A_elfe - A_series| / max|A_series| (%) [max|A_series| / max|V_series| (%)]']
        for jf, f in enumerate(fr):
            e, e0 = el[f]
            if e is None:
                continue
            tot, an = [], []
            for k, cc in enumerate(COMPONENTS):
                s, a = ser[0, jf, k], ans[0, jf, k]
                if np.abs(s).max() < ZERO * np.abs(ser[0, jf, 3 * (k // 3):3 * (k // 3) + 3]).max():
                    tot.append(f'{cc} zero'); an.append(f'{cc} zero')      # vanishes in the series (T4 E_z)
                    rows.append((c, f, cc, np.nan, np.nan, np.nan)); continue
                dev = np.abs(e[k] - s) / np.abs(s).max()
                tot.append(f'{cc} {100 * np.median(dev):5.2f}/{100 * np.percentile(dev, 90):5.1f}')
                size = np.abs(a).max() / np.abs(s).max()
                err = np.abs((e[k] - e0[k]) - a).max() / np.abs(a).max() if e0 is not None else np.nan
                an.append(f'{cc} {100 * err:5.1f} [{100 * size:4.1f}]')
                rows.append((c, f, cc, np.median(dev), size, err))
            lines += [f'total    f = {f:8g} Hz  ' + '  '.join(tot), f'anomaly  f = {f:8g} Hz  ' + '  '.join(an)]
        os.makedirs(os.path.join(st.folder, 'comparison'), exist_ok=True)
        open(os.path.join(st.folder, 'comparison', 'elfe3d_sphere.txt'), 'w').write('\n'.join(lines) + '\n')
        print('\n'.join(lines))
        desc = open(os.path.join(st.folder, 'control.toml')).readline().lstrip('# ').strip()
        for prof in d['profile'][np.sort(np.unique(d['profile'], return_index=True)[1])]:
            setup, ft, fa = (f'fig/{c}_{prof}_{x}.pdf' for x in ('setup', 'total', 'anomaly'))
            setup_figure(st, d, prof, H, os.path.join(out, setup))
            profile_figure(st, d, ser, 0, prof, os.path.join(out, ft), P,
                           ref=lambda f: (el[f][0], m))
            profile_figure(st, d, ans, 0, prof, os.path.join(out, fa), P,
                           ref=lambda f: ((el[f][0] - el[f][1]) if el[f][1] is not None else None, m + ' - ' + m + 'ref'))
            body.append(rf'\section{{{tex_escape(c)}, {prof}-profile}}' + '\n' + tex_escape(desc) + '\n')
            body.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{%s}' % setup +
                        r'\caption{Setup of %s, %s-profile, hill %g m with sphere: plan view (left) and xz-section at '
                        r'$y = 0$ (right).}' % (tex_escape(c), prof, H) + r'\end{figure}' + '\n')
            body.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{%s}' % ft +
                        r'\caption{%s, %s-profile, total field with sphere: series (lines) and elfe3D (circles, every '
                        r'second receiver).}' % (tex_escape(c), prof) + r'\end{figure}' + '\n' + r'\clearpage' + '\n')
            body.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{%s}' % fa +
                        r'\caption{%s, %s-profile, sphere anomaly $V(\mathrm{sphere}) - V(\mathrm{no\ sphere})$: '
                        r'series (lines) and elfe3D run with sphere minus reference run on the same mesh (circles).}'
                        % (tex_escape(c), prof) + r'\end{figure}' + '\n' + r'\clearpage' + '\n')
            print(c, prof)
    fmt = lambda v: '--' if not np.isfinite(v) else f'{100 * v:.2g}'
    tab = []
    for c in cases:
        for f in sorted({r[1] for r in rows if r[0] == c}):
            rr = {r[2]: r for r in rows if r[0] == c and r[1] == f}
            tab.append(f'{tex_escape(c)} & {f:g} & ' + ' & '.join(
                f'{fmt(rr[k][3])} & ' + (f'{fmt(rr[k][5])} ({fmt(rr[k][4])})' if rr[k][4] >= MIN_ANOMALY else
                                         f'({fmt(rr[k][4])})') for k in COMPONENTS) + r' \\')
    st0 = Config(os.path.join(root, 'cases', cases[0])); sp = st0.sphere
    table = (r'\section{Deviations}' + '\n' +
             r'Per component two columns, in per cent: total field $|V_\mathrm{elfe} - V_\mathrm{series}| / '
             r'\max|V_\mathrm{series}|$ (median over both profiles), and anomaly $\max|A_\mathrm{elfe} - '
             r'A_\mathrm{series}| / \max|A_\mathrm{series}|$ with the size of the anomaly $\max|A_\mathrm{series}| / '
             r'\max|V_\mathrm{series}|$ in brackets (anomalies below %g\,\%% of the field: size only; --: the component vanishes in the series, $E_z$ of T4).' % (100 * MIN_ANOMALY)
             + '\n' + r'\begin{center}\scriptsize\setlength{\tabcolsep}{2.5pt}\begin{tabular}{ll' + 'rl' * 6 + '}\n' +
             r'case & $f$ & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{${k[0]}_{k[1]}$}}' for k in COMPONENTS) +
             r' \\ \hline' + '\n' + '\n'.join(tab) + '\n' + r'\end{tabular}\end{center}' + '\n')
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[hmargin=2cm,vmargin=1.5cm]{geometry}
\usepackage[font=small,skip=2pt]{caption}
\setlength{\textfloatsep}{6pt}\setlength{\intextsep}{6pt}
\usepackage{graphicx,float}
\title{terrasphere\_csem: sphere under a hill against elfe3D}
\date{\today}
\begin{document}
\maketitle
Series solution of this package (lines) against the 3D finite-element code elfe3D (circles) for the four sources of
the elfe3D comparison with a sphere: Gaussian hill of height %g m and width 150 m, three layers (interfaces at 200 m
and 600 m depth), sphere of %g $\Omega$m, radius %g m, centre (%g, %g) m at %g m depth (in layer 2, under the hill).
For each source two figures: the total field with the sphere, and the anomaly of the sphere.  The elfe3D anomaly is
the difference of the run with the sphere and a run on the identical mesh with the host resistivity in the sphere, so
that the discretisation error of the background largely cancels; the series anomaly is the series with the sphere
minus the series without it.  Receivers 1 cm below the surface on the x- and y-profile from $-600$ m to 600 m every
20 m.  Fields per 1 A source current, E in nV/m, H in $\mu$A/m, time dependence $e^{-i\omega t}$, in the
convention of this package: the series exactly as it writes them, elfe3D converted ($H = -H_\mathrm{elfe3D}$).
\tableofcontents
\clearpage
""" % (H, 1 / float(sp['conductivity']), float(sp['radius']), *(float(v) for v in sp['center'])) + \
        '\n'.join(body) + table + r"""
\end{document}
"""
    open(os.path.join(out, 'report.tex'), 'w').write(tex)
    print(os.path.join(out, 'report.tex'))


if __name__ == '__main__':
    main()
