"""
matplotlib figures typeset by LaTeX in the README figures' engraving style.

For the parts of a README figure that TikZ and pgfplots handle badly: heatmaps, contour
maps, images and scatters of more than a few thousand points. The PGF backend writes LaTeX
drawing commands, not an image, so every label is set in Libertinus by LuaLaTeX and can use
the notation macros of latentfig.sty; large raster parts go to a PNG beside the .pgf.

A generator script in figures/src/ runs from figures/ (`just figures` does this):

  import sys; sys.path.insert(0, 'tex')
  import figstyle
  figstyle.setup()
  fig, ax = figstyle.figure(width=0.5, aspect=0.8)   # fraction of the 6.5 in canvas
  ax.imshow(grid, cmap=figstyle.GREYS, rasterized=True)
  figstyle.save(fig, 'build/units')                  # build/units.pgf (+ PNG parts)

and the figure's .tex includes it with `import{build/}{units.pgf}` (the import package's
command), inside its figurecanvas, so the raster part's path resolves.
"""

import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('pgf')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FIGWIDTH_PT = 6.5 * 72.27  # \figwidth of latentfig.sty
ENGINE = 'lualatex'

# Palette (matches latentfig.sty): ink, lighter inks, and one spot colour.
INK = '#111111'
SOFT = '#555555'
FAINT = '#8A8A8A'
SPOT = '#C8372D'

# Colour maps, for the rare figure that needs one: greys for magnitudes, and ink through
# white to spot for signed values (centre it on 0).
GREYS = LinearSegmentedColormap.from_list('greys', ['#FFFFFF', '#BDBDBD', '#5A5A5A', INK])
SIGNED = LinearSegmentedColormap.from_list('signed', [INK, '#9A9A9A', '#FFFFFF', '#E3A49E', SPOT])

_STYLE_DIR = Path(__file__).resolve().parent  # the folder holding latentfig.sty


def setup(font_size: float = 9.0) -> None:
  """Hand all text to LuaLaTeX with latentfig's fonts, and set the engraving look."""
  # Let the LaTeX run inside matplotlib find latentfig.sty; the trailing separator keeps
  # TeX's default search path too.
  existing = os.environ.get('TEXINPUTS', '').strip(os.pathsep)
  os.environ['TEXINPUTS'] = os.pathsep.join([str(_STYLE_DIR), *([existing] if existing else [])]) + os.pathsep
  plt.rcParams.update({
    'pgf.texsystem': ENGINE,
    'pgf.rcfonts': False,
    'pgf.preamble': r'\usepackage[text]{latentfig}',
    'text.usetex': False,  # the pgf backend handles text itself
    'font.family': 'serif',
    'font.size': font_size,
    'text.color': INK,
    'axes.labelsize': font_size - 1,
    'axes.labelcolor': INK,
    'axes.edgecolor': INK,
    'axes.linewidth': 0.5,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.prop_cycle': matplotlib.cycler(color=[INK]),
    'xtick.labelsize': font_size - 1,
    'ytick.labelsize': font_size - 1,
    'xtick.color': INK,
    'ytick.color': INK,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'lines.linewidth': 0.7,
    'hatch.color': '#8C8C8C',
    'hatch.linewidth': 0.3,
    'legend.frameon': False,
    'legend.fontsize': font_size - 1,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    # no bbox='tight': it would crop the figure narrower than requested
  })


def figure(width: float = 1.0, aspect: float = 0.4, nrows: int = 1, ncols: int = 1, **kwargs: Any) -> Any:
  """A figure `width` × the canvas wide and `aspect` × that tall, laid out to fit exactly."""
  kwargs.setdefault('layout', 'constrained')
  w_in = FIGWIDTH_PT / 72.27 * width
  return plt.subplots(nrows, ncols, figsize=(w_in, w_in * aspect), **kwargs)


def title(ax: Any, text: str) -> None:
  """An italic run-in panel title, as `head` in latentfig.sty, e.g. '(a) Layer 1.'."""
  ax.set_title(rf'\textit{{{text}}}', loc='left', fontsize=plt.rcParams['font.size'], color=INK, pad=6)


def save(fig: Any, path: str | Path) -> Path:
  """Write the figure as .pgf at the given path stem."""
  out = Path(path).with_suffix('.pgf')
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out)
  plt.close(fig)
  return out
