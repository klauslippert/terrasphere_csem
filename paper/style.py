"""Common plot style of the figure scripts (English labels) and the helper that saves a figure as PDF and PNG."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'lines.linewidth': 1.2, 'savefig.dpi': 200})
C = dict(hill='#1f5fa8', valley='#d9731a', flat='0.45', sphere='#2a9d5c', third='#8e44ad', layer='#b7a27a')


def save(fig, name, outdir):
    """Save the figure as `<outdir>/<name>.pdf` and `<outdir>/<name>.png` (the folder is created if necessary)."""
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, name + '.png'), bbox_inches='tight')
    print('saved', name)
