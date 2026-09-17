"""CMS publication-style conventions for every figure the ABCD pipeline draws.

All visual choices live in this module so the physics code stays free of styling
and every plot in the note/thesis looks like it came from the same analysis:

- mplhep's CMS style (fonts, in-pointing ticks, four-sided frame) as the base.
- The CMS-recommended colour schemes (Petroff, arXiv:2107.02270): colour-blind
  safe and still readable in greyscale print.
- One ``CMS  Preliminary`` header per figure carrying lumi and sqrt(s),
  configured once per run with :func:`configure` (or the ``ABCD_CMS_LABEL`` /
  ``ABCD_LUMI`` / ``ABCD_COM`` environment variables for the standalone
  scripts).
- Every figure written as both PDF (vector, for the document) and PNG (for
  browsing), via :func:`save`.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

# hep.style.CMS is tuned for a 10x10 figure at font.size 26. Only the few knobs
# that CMS leaves open are set here; everything else stays at the style default
# so upgrading mplhep keeps us current.
plt.rcParams.update({
    "figure.autolayout": False,       # we call tight_layout / bbox_inches ourselves
    "axes.labelsize": "medium",
    "legend.fontsize": 20,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# --------------------------------------------------------------------- #
# Colours (Petroff, arXiv:2107.02270)
# --------------------------------------------------------------------- #
# 10-colour scheme, for categorical series with many entries.
PETROFF_10 = [
    "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
    "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd",
]
# 6-colour high-contrast scheme, for the few-series plots.
PETROFF_6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]

# Semantic assignments: a given quantity keeps its colour across every figure.
SIGNAL_COLOR = "#e42536"      # red
BACKGROUND_COLOR = "#5790fc"  # blue
DATA_COLOR = "black"
TRAIN_COLOR = "#5790fc"
VAL_COLOR = "#f89c20"
NEUTRAL_COLOR = "#9c9ca1"

LINESTYLES = ["-", "--", "-.", ":"]

# Single-hue sequential ramps for the 2D occupancy maps. Single-hue (rather than
# viridis-like) ramps keep a monotonic lightness in greyscale print, and they let
# background-MC and data planes be told apart at a glance.
CMAP_MC = "Blues"
CMAP_DATA = "Greens"

# Profile-marker colours for the data plane. Both clear a 3:1 contrast ratio against
# the light end of the Greens ramp (violet 8.2:1, orange 3.1:1), where a plain yellow
# scores 2.08:1, and they stay 95 dE apart under simulated protan/deutan/tritan vision.
PROFILE_COLOR = "#4a3aa7"
PROFILE_COLOR_WEIGHTED = "#eb6834"

# Figure sizes, all at the CMS style's native font scale.
FIG_SINGLE = (10, 9)      # one panel, no colorbar
FIG_PLANE = (11, 9)       # one 2D panel plus colorbar
FIG_RATIO = (10, 11)      # main panel plus ratio panel

SAVE_FORMATS = ("pdf", "png")
# Extensions save() will strip off a caller-supplied path before appending its own.
_IMAGE_EXTS = {".pdf", ".png", ".svg", ".eps", ".ps", ".jpg", ".jpeg"}

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Run-wide plot context
# --------------------------------------------------------------------- #
@dataclass
class CMSContext:
    """Everything the CMS header needs, shared by every figure in one run."""

    label: str = "Preliminary"   # italic text after "CMS"; "" for publication
    lumi: float = None           # integrated luminosity in fb^-1 (None: omitted)
    com: float = 13.6            # sqrt(s) in TeV
    extra: str = ""              # in-frame annotation, e.g. the channel name


def _env_float(name):
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: not a number", name, raw)
        return None


CONTEXT = CMSContext(
    label=os.environ.get("ABCD_CMS_LABEL", "Preliminary"),
    lumi=_env_float("ABCD_LUMI"),
    com=_env_float("ABCD_COM") or 13.6,
    extra=os.environ.get("ABCD_CHANNEL", ""),
)


def configure(label=None, lumi=None, com=None, extra=None):
    """Set the run-wide CMS header context. Unset arguments keep their value."""
    if label is not None:
        CONTEXT.label = label
    if lumi is not None:
        CONTEXT.lumi = float(lumi)
    if com is not None:
        CONTEXT.com = float(com)
    if extra is not None:
        CONTEXT.extra = extra


def add_cli_args(parser):
    """Add --cms-label / --lumi / --com to a standalone script's parser."""
    group = parser.add_argument_group("plot style")
    group.add_argument("--cms-label", default=None,
                       help='Italic text after "CMS" (e.g. "Preliminary", "Simulation", "" for publication)')
    group.add_argument("--lumi", type=float, default=None, help="Integrated luminosity in fb^-1 for the header")
    group.add_argument("--com", type=float, default=None, help="Centre-of-mass energy in TeV (default: 13.6)")
    return parser


def configure_from_args(args, extra=None):
    """Apply the :func:`add_cli_args` options to the run-wide context."""
    configure(label=getattr(args, "cms_label", None), lumi=getattr(args, "lumi", None),
              com=getattr(args, "com", None), extra=extra)


# --------------------------------------------------------------------- #
# Axis labels
# --------------------------------------------------------------------- #
# Branch names that get a typeset label instead of the underscore-stripped one.
AXIS_LABELS = {
    "dnn_score": "DNN score",
    "dnn_0_score": "DNN score 0",
    "dnn_1_score": "DNN score 1",
    "bdt_score": "BDT score",
}


def axis_label(name):
    """Human-readable axis label for a branch name."""
    return AXIS_LABELS.get(name, str(name).replace("_", " "))


# --------------------------------------------------------------------- #
# Figure furniture
# --------------------------------------------------------------------- #
def cms_header(ax, data=False, lumi=None, com=None, label=None, loc=0):
    """Draw the ``CMS Preliminary ... fb^-1 (13.6 TeV)`` header above ``ax``.

    ``data=False`` makes mplhep prefix the label with "Simulation", which is the
    right header for every MC-only figure.
    """
    hep.cms.label(
        label if label is not None else CONTEXT.label,
        data=data,
        lumi=lumi if lumi is not None else (CONTEXT.lumi if data else None),
        com=com if com is not None else CONTEXT.com,
        ax=ax,
        loc=loc,
    )


def annotate(ax, lines, x=0.04, y=0.94, fontsize=20, ha="left", va="top", **kwargs):
    """In-frame annotation (channel, selection, region) under the CMS header."""
    if not lines:
        return None
    if isinstance(lines, str):
        lines = [lines]
    lines = [ln for ln in lines if ln]
    if not lines:
        return None
    return ax.text(
        x, y, "\n".join(lines), transform=ax.transAxes,
        ha=ha, va=va, fontsize=fontsize, linespacing=1.4, **kwargs,
    )


def legend(ax, loc="best", ncols=1, **kwargs):
    """CMS-style legend: no frame, no shadow, sensible default size."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    kwargs.setdefault("fontsize", 18)
    return ax.legend(handles, labels, loc=loc, frameon=False, ncols=ncols, **kwargs)


def headroom(ax, factor=1.5, log=False, ymin=None):
    """Leave room above the tallest bin so the legend never sits on the data."""
    top = ax.get_ylim()[1]
    if log:
        ax.set_yscale("log")
        bottom = ymin if ymin is not None else max(ax.get_ylim()[0], 1e-3)
        ax.set_ylim(bottom, top * (factor ** 4))
    else:
        ax.set_ylim(0 if ymin is None else ymin, top * factor)


def colorbar(fig, mappable, ax, label="Events"):
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    cbar.set_label(label, fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    return cbar


def save(fig, path, close=True):
    """Write one figure as PDF (for the document) and PNG (for browsing).

    ``path`` may carry an image extension or none at all. Extensions are appended
    rather than substituted with ``with_suffix``, because the plot names embed
    checkpoint stems such as ``single-abcdisco-042-0.1234`` whose dot would
    otherwise be mistaken for a suffix. Returns the list of files written.
    """
    path = Path(path)
    if path.suffix.lower() in _IMAGE_EXTS:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in SAVE_FORMATS:
        out = path.with_name(f"{path.name}.{ext}")
        fig.savefig(out, bbox_inches="tight")
        written.append(out)
    if close:
        plt.close(fig)
    logger.info("Saved %s.{%s}", path, ",".join(SAVE_FORMATS))
    return written
