#!/usr/bin/env python3
"""Kreisschleife in empymod als 1-D-Referenz (Owner 06.10.2026: "empymod als 1-D-Vergleich").
Polygon aus N geraden Segmenten (empymod.bipole, srcpts Gauss-Punkte je Segment), Umkreisradius so skaliert,
dass die Polygonflaeche gleich pi a^2 ist; eine Sehnenmitte liegt auf der +x-Achse.
Koordinaten: empymod z nach UNTEN (z_emp = Tiefe), unsere z nach oben. Empfaenger auf der +x-Achse mit
Azimut 90 Grad (= +y = phi-Richtung) -> E_phi.
Zeitkonvention: empymod rechnet mit e^{+iwt}, wir mit e^{-iwt}; die Frequenzwerte werden daher KONJUGIERT
zurueckgegeben (Probe 06.10.2026: VMD-Vollraum ours/conj(emp) reell auf 1e-7)."""
import numpy as np
import empymod
# Befund 06.10.2026: empymods Standard-Hankelfilter (key_201_2012) liefert fuer das feine Schleifenpolygon bei
# Empfaengern im Nahfeld 2-3 % Fehler; key_401_2009 + exakte Frequenzen (pts_per_dec -1) bringen 1e-4.
STD = dict(htarg={'dlf': 'key_401_2009'}, ftarg={'pts_per_dec': -1})


def polygon(a, N=256, z=0.0, flaechengleich=True):
    r = a * np.sqrt(np.pi / ((N / 2) * np.sin(2 * np.pi / N))) if flaechengleich else a
    ph = (np.arange(N + 1) + 0.5) * 2 * np.pi / N
    x, y = r * np.cos(ph), r * np.sin(ph)
    return [x[:-1], x[1:], y[:-1], y[1:], np.full(N, z), np.full(N, z)]


def E_phi_freq(a, rho, z_emp, f, depth, res, N=256, srcpts=5, **kw):
    """E_phi [V/m] bei 1 A an Punkten (rho_i, 0, z_emp_i), Frequenzen f [Hz]; Form (nf, npts); konjugiert."""
    rho, z_emp, f = (np.atleast_1d(v).astype(float) for v in (rho, z_emp, f))
    rec = [rho, np.zeros_like(rho), z_emp, 90.0, 0.0]
    out = empymod.bipole(src=polygon(a, N), rec=rec, depth=depth, res=res, freqtime=f,
                         srcpts=srcpts, strength=1.0, verb=0, **{'htarg': STD['htarg'], **kw})
    return np.conj(np.asarray(out).reshape(len(f), len(rho), N).sum(-1))


def V_off_time(a, t, depth, res, a_rx=None, z_rec=0.02, N=256, srcpts=5, **kw):
    """Ausschaltspannung [V] der koaxialen Empfangsschleife (Radius a_rx, Standard = a) bei 1 A:
    V = 2 pi a_rx E_phi(rho = a_rx, z_rec).  ACHTUNG (Befund 06.10.2026): bei a_rx = a sitzt der Empfaenger am Draht,
    die log-singulaere Selbstinduktion (rein imaginaer) verdirbt empymods Zeittransformation (dlf ab ~3 ms grob
    falsch, fftlog voellig). Fuer Zeitbereichsvergleiche a_rx deutlich != a waehlen."""
    t = np.atleast_1d(t).astype(float)
    a_rx = a if a_rx is None else a_rx
    rec = [a_rx, 0.0, z_rec, 90.0, 0.0]
    out = empymod.bipole(src=polygon(a, N), rec=rec, depth=depth, res=res, freqtime=t, signal=-1,
                         srcpts=srcpts, strength=1.0, verb=0, **{**STD, **kw})
    return 2 * np.pi * a_rx * np.asarray(out).reshape(len(t), N).sum(-1)


def V_freq(a, f, depth, res, a_rx=None, z_rec=0.02, N=256, srcpts=5, **kw):
    """V(w) [V/A] der koaxialen Empfangsschleife, konjugiert. Re V ist auch bei a_rx = a endlich und brauchbar."""
    a_rx = a if a_rx is None else a_rx
    return 2 * np.pi * a_rx * E_phi_freq(a, a_rx, z_rec, f, depth, res, N=N, srcpts=srcpts, **kw)[:, 0]


def E_vmd(x, z_emp, z_src, f, depth, res):
    """E_phi (= E_y auf der x-Achse) eines vertikalen Magnetdipols bei (0,0,z_src), Normierung empymod (strength 0);
    Empfaenger in verschiedenen Tiefen -> bipole mit msrc; konjugiert."""
    x, z_emp = np.atleast_1d(x).astype(float), np.atleast_1d(z_emp).astype(float)
    out = empymod.bipole(src=[0.0, 0.0, z_src, 0.0, 90.0], rec=[x, np.zeros_like(x), z_emp, 90.0, 0.0], depth=depth,
                         res=res, freqtime=f, msrc=True, strength=0.0, verb=0, htarg=STD['htarg'])
    return np.conj(np.asarray(out).reshape(-1))
