"""Shared style for paper figures, so all notebooks in run/ use the same colours, fonts and sizes.

    from py_files import plot_style as ps
    ps.apply()                                   # once per notebook, before plotting
    fig, ax = plt.subplots(figsize=(ps.COLUMN, 3))
    ax.imshow(img, cmap=ps.DIVERGING)            # signed quantities (offsets, residuals), symmetric limits
    ax.hexbin(x, y, cmap=ps.DENSITY)             # point density behind coloured overlays
    ax.plot(x, y, color=ps.SERIES[0])            # lines/markers: SERIES in this fixed order, never cycled

Colours are colour-blind safe: the categorical order passes CVD separation checks for adjacent series (protan /
deutan / tritan); the diverging map runs blue -> neutral grey -> red with equal arms; sequential and density maps
are single-hue, light -> dark. Three of the lighter series colours (aqua, yellow, magenta) are low-contrast on
white, so multi-series plots must keep a legend and distinct marker shapes (MARKERS).

Figures are made at their printed size (MNRAS column / page width) so the fonts below are the fonts in the paper:
don't make a figure 2x too big and let LaTeX shrink it, which is what made the old labels unreadable.
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

COLUMN = 3.32    # inches, MNRAS \columnwidth (84 mm)
PAGE = 6.97      # inches, MNRAS \textwidth (177 mm)

# categorical series, fixed order: blue, orange, aqua, yellow, magenta, green, violet, red
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

# diverging: blue <-> red, neutral grey midpoint, equal arms
DIVERGING = LinearSegmentedColormap.from_list(
    'lpm_diverging', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f3b1aa', '#e34948', '#8c1c1b'])
DIVERGING.set_bad('#bdbdbd')      # no data
# sequential magnitude: one hue (blue), light -> dark
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'lpm_sequential', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
# point density behind coloured overlays: one hue (grey), light -> dark
DENSITY = 'Greys'


def apply():
    """Set matplotlib defaults for paper figures (fonts sized for print at COLUMN / PAGE width)."""
    mpl.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'figure.titlesize': 9,
        'axes.prop_cycle': mpl.cycler(color=SERIES),
        'lines.linewidth': 1.2, 'lines.markersize': 4,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'legend.frameon': False,
        'image.cmap': 'lpm_sequential',
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
    })


for _cm in (DIVERGING, SEQUENTIAL):
    if _cm.name not in mpl.colormaps:
        mpl.colormaps.register(_cm)
