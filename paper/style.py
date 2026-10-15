"""Common plot style of the figure scripts (English labels) and the helper that saves a figure as PDF and PNG."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'lines.linewidth': 1.2, 'savefig.dpi': 200})
C = dict(hill='#1f5fa8', depression='#d9731a', flat='0.45', sphere='#2a9d5c', third='#8e44ad', layer='#b7a27a')
# style of the figures taken over from the sphere paper: larger font, grid, colour cycle COL
LS_RC = {'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
         'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.2}
# style of the figures taken over from the terrain paper: its colours GH_C and settings GH_RC
GH_C = dict(hill='#2a78d6', depression='#eb6834', third='#1baf7a', fourth='#eda100', flat='#5f5e5a', ref='#000000')
GH_RC = {'font.size': 8.5, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'axes.grid': True, 'grid.color': '#dddddd',
         'grid.linewidth': 0.5, 'legend.frameon': False, 'legend.fontsize': 7.5, 'xtick.major.width': 0.6,
         'ytick.major.width': 0.6, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.labelsize': 8.5,
         'axes.titlesize': 8.5}
COL = ['#1f4e79', '#c0392b', '#2e8b57', '#d68910', '#6c3483', '#17a2b8', '#7f8c8d']


def save(fig, name, outdir, png=True):
    """Save the figure as `<outdir>/<name>.pdf` and, if `png`, `<outdir>/<name>.png` (the folder is created if
    necessary)."""
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + '.pdf'), bbox_inches='tight')
    if png:
        fig.savefig(os.path.join(outdir, name + '.png'), bbox_inches='tight')
    print('saved', name)
