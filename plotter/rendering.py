#!/usr/bin/env python
"""CMS analysis-note style rendering for filled ``hist.Hist`` histograms.

All visual conventions live here, separate from the I/O and histogram
filling in ``plotter.py``:

- mplhep CMS style with the CMS-recommended colour schemes
  (Petroff, arXiv:2107.02270; colour-blind safe).
- Stacked backgrounds (largest at the bottom) with a hatched MC
  statistical-uncertainty band.
- Data as black points, signal overlays as coloured step lines.
- A Data / Pred. ratio panel with the MC uncertainty band around unity.
- Plots saved as both PNG (for browsing) and PDF (for the note).
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist

logger = logging.getLogger(__name__)

hep.style.use(hep.style.CMS)

# CMS-recommended 10-colour scheme (Petroff, arXiv:2107.02270) for the
# background stack. Colours are assigned by category order in the config,
# so a given process keeps its colour across every plot.
STACK_COLORS = [
    "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
    "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd",
]
# High-contrast line colours (from the 6-colour scheme) for signal overlays;
# distinct linestyles keep them apart in print / greyscale.
SIGNAL_COLORS = ["#e42536", "#7a21dd", "#f89c20"]
SIGNAL_LINESTYLES = ["-", "--", "-.", ":"]

UNC_LABEL = "Stat. unc."
SAVE_FORMATS = ["png"]
PNG_DPI = 300
LINEAR_HEADROOM = 1.45   # y-axis headroom factors so the legend never
LOG_HEADROOM = 100.0     # overlaps the histograms


@dataclass
class PlotStyle:
    """Plot-wide presentation context (everything but the histograms)."""
    cms_label: str = "Preliminary"   # text next to "CMS"; "" for publication
    lumi: Optional[float] = None     # integrated luminosity in fb^-1
    com: float = 13.6                # sqrt(s) in TeV
    ratio_ylim: Tuple[float, float] = (0.0, 2.0)
    logy_ymin: float = 0.1


def scale_hist(h: Hist, scale: float) -> Hist:
    """Return a copy of a Weight-storage hist scaled by a constant."""
    if scale == 1.0:
        return h
    h2 = h.copy()
    view = h2.view()
    view["value"] *= scale
    view["variance"] *= scale * scale
    return h2


def sum_hists(hists: List[Hist]) -> Hist:
    total = hists[0].copy()
    for h in hists[1:]:
        total += h
    return total


def save_figure(fig, name: str, save_path: str) -> None:
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    for ext in SAVE_FORMATS:
        dpi = PNG_DPI if ext == "png" else None
        fig.savefig(save_dir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved {save_dir / name}.{{{','.join(SAVE_FORMATS)}}}")


# --------------------------------------------------------------------- #
# 1D
# --------------------------------------------------------------------- #
def render_1d(
    cfg,
    style: PlotStyle,
    bkg_labels: Optional[List[str]] = None,
    sig_labels: Optional[List[str]] = None,
    density: bool = False,
    save: bool = True,
    save_path: str = "plots",
) -> None:
    """Draw one Hist1D config (already filled) in CMS style."""
    data_h = cfg.hist_data[0] if cfg.hist_data else None
    bkg_hs = list(cfg.hist_bkg or [])
    sig_hs = list(cfg.hist_sig or [])
    if data_h is None and not bkg_hs and not sig_hs:
        logger.warning(f"No histograms to plot for {cfg.var}")
        return

    # Stack entries keep their colour by config position (identity), but are
    # stacked largest-first so the biggest background sits at the bottom.
    if bkg_labels is None:
        bkg_labels = [f"Background {i + 1}" for i in range(len(bkg_hs))]
    entries = [
        (h, lab, STACK_COLORS[i % len(STACK_COLORS)])
        for i, (h, lab) in enumerate(zip(bkg_hs, bkg_labels))
        if h.sum().value > 0
    ]
    entries.sort(key=lambda e: e[0].sum().value, reverse=True)
    total = sum_hists([e[0] for e in entries]) if entries else None

    has_ratio = data_h is not None and total is not None
    fig, ax, rax = _make_figure(has_ratio)
    ymax = 0.0

    if entries:
        hep.histplot(
            [e[0] for e in entries], ax=ax, histtype="fill", stack=True,
            color=[e[2] for e in entries], label=[e[1] for e in entries],
            density=density, flow="none",
        )
        ymax = max(ymax, _peak(total, density))
        if not density:
            _stat_band(ax, total, label=UNC_LABEL)

    for i, sig_h in enumerate(sig_hs):
        scale = 1.0 if density else cfg.scale
        drawn = scale_hist(sig_h, scale)
        label = (sig_labels[i] if sig_labels and i < len(sig_labels)
                 else f"Signal {i + 1}" if len(sig_hs) > 1 else "Signal")
        hep.histplot(
            drawn, ax=ax, histtype="step", linewidth=2.5, yerr=False,
            color=SIGNAL_COLORS[i % len(SIGNAL_COLORS)],
            linestyle=SIGNAL_LINESTYLES[i % len(SIGNAL_LINESTYLES)],
            label=label + _scale_suffix(scale), density=density, flow="none",
        )
        ymax = max(ymax, _peak(drawn, density))

    if data_h is not None:
        # Empty data bins (e.g. a blinded region) are masked, not drawn at 0.
        drawn = data_h if density else _mask_empty(data_h)
        hep.histplot(drawn, ax=ax, histtype="errorbar", color="black",
                     markersize=6, label="Data", density=density, flow="none")
        ymax = max(ymax, _peak(data_h, density, with_err=True))

    if has_ratio:
        _draw_ratio(rax, data_h, total, style)
        ax.set_xlabel("")  # histplot labels the shared x-axis; only rax keeps it
        rax.set_xlabel(cfg.xlabel)
    else:
        ax.set_xlabel(cfg.xlabel)

    ax.set_ylabel(_events_label(cfg, density))
    ax.set_xlim(cfg.binning[1], cfg.binning[2])
    _set_headroom(ax, ymax, cfg.logy, density, style)
    _legend(ax)
    hep.cms.label(style.cms_label, data=data_h is not None,
                  lumi=style.lumi, com=style.com, ax=ax)

    if save:
        name = f"{cfg.var}_logy" if cfg.logy else cfg.var
        save_figure(fig, name, save_path)
    plt.close(fig)


def _make_figure(has_ratio: bool):
    if has_ratio:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(10, 11), sharex=True,
            gridspec_kw={"height_ratios": (3.5, 1), "hspace": 0.06},
        )
        return fig, ax, rax
    fig, ax = plt.subplots(figsize=(10, 10))
    return fig, ax, None


def _peak(h: Hist, density: bool, with_err: bool = False) -> float:
    """Maximum drawn bin height of a histogram."""
    if density:
        vals = h.density()
    else:
        vals = h.values()
        if with_err:
            vals = vals + np.sqrt(np.abs(h.variances()))
    return float(np.nanmax(vals)) if len(vals) else 0.0


def _mask_empty(h: Hist) -> Hist:
    """Copy of a hist with empty bins set to NaN so they are not drawn."""
    h2 = h.copy()
    view = h2.view()
    empty = view["value"] == 0
    view["value"][empty] = np.nan
    view["variance"][empty] = np.nan
    return h2


def _stat_band(ax, total: Hist, label: Optional[str] = None, around_one: bool = False):
    """Hatched MC statistical-uncertainty band (absolute, or relative to 1)."""
    vals = total.values()
    errs = np.sqrt(np.abs(total.variances()))
    if around_one:
        rel = np.divide(errs, vals, out=np.zeros_like(errs), where=vals > 0)
        lo, hi = 1.0 - rel, 1.0 + rel
    else:
        lo, hi = vals - errs, vals + errs
    ax.stairs(hi, total.axes[0].edges, baseline=lo, fill=True,
              facecolor="none", edgecolor="black", alpha=0.5,
              hatch="////", linewidth=0, label=label)


def _draw_ratio(rax, data_h: Hist, total: Hist, style: PlotStyle) -> None:
    vals = total.values()
    centers = total.axes[0].centers
    good = (vals > 0) & (data_h.values() > 0)
    ratio = data_h.values()[good] / vals[good]
    yerr = np.sqrt(np.abs(data_h.variances()[good])) / vals[good]
    _stat_band(rax, total, around_one=True)
    rax.errorbar(centers[good], ratio, yerr=yerr, fmt="o",
                 color="black", markersize=5)
    rax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    rax.set_ylabel("Data / Pred.")
    rax.set_ylim(*style.ratio_ylim)
    rax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="both"))


def _events_label(cfg, density: bool) -> str:
    if density:
        return "A. U."
    nbins, xmin, xmax = cfg.binning
    width = round((xmax - xmin) / nbins, 2)
    unit = re.search(r"\[(.+?)\]", cfg.xlabel)
    unit = f" {unit.group(1)}" if unit else ""
    return f"Events / {width:g}{unit}"


def _scale_suffix(scale: float) -> str:
    if scale == 1.0:
        return ""
    exp = np.log10(scale)
    if exp == int(exp):
        return rf" ($\times 10^{{{int(exp)}}}$)"
    return rf" ($\times$ {scale:g})"


def _set_headroom(ax, ymax: float, logy: bool, density: bool, style: PlotStyle) -> None:
    if ymax <= 0:
        return
    if logy:
        ax.set_yscale("log")
        if density:
            ax.set_ylim(top=ymax * LOG_HEADROOM)
        else:
            ax.set_ylim(style.logy_ymin, ymax * LOG_HEADROOM)
    else:
        ax.set_ylim(0, ymax * LINEAR_HEADROOM)


def _legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    # Data first, then the stack (largest first), uncertainty band, signals.
    if "Data" in labels:
        i = labels.index("Data")
        handles.insert(0, handles.pop(i))
        labels.insert(0, labels.pop(i))
    ax.legend(handles, labels, loc="upper right", frameon=False,
              fontsize=18, ncols=2 if len(handles) > 4 else 1)


# --------------------------------------------------------------------- #
# 2D
# --------------------------------------------------------------------- #
def render_2d(cfg, style: PlotStyle, save: bool = True, save_path: str = "plots") -> None:
    panels = []
    if cfg.hist_data:
        panels.append(("Data", cfg.hist_data[0]))
    if cfg.hist_bkg:
        panels.append(("Background", sum_hists(cfg.hist_bkg)))
    if cfg.hist_sig:
        panels.append(("Signal", cfg.hist_sig[0]))
    if not panels:
        logger.warning(f"No histograms to plot for {cfg.xvar} vs {cfg.yvar}")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, h) in zip(axes, panels):
        vals = h.values().T
        vals = np.where(vals > 0, vals, np.nan)  # empty bins stay white
        im = ax.pcolormesh(h.axes[0].edges, h.axes[1].edges, vals,
                           cmap="viridis", rasterized=True)
        ax.set_xlabel(cfg.xlabel)
        ax.set_ylabel(cfg.ylabel)
        ax.set_title(title, fontsize=20)
        fig.colorbar(im, ax=ax, label="Events")

    if save:
        save_figure(fig, f"{cfg.xvar}_vs_{cfg.yvar}", save_path)
    plt.close(fig)
