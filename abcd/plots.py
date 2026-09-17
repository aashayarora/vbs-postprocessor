"""Diagnostic plots for training inputs, the ABCD plane, and model performance.

Every figure is drawn in the CMS style defined in ``style.py`` and written as
both PDF and PNG, so the same code produces the thesis/note figures and the
browsing plots.
"""

import logging
import re
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.patheffects
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import style
from common import data_length, score_column, to_flat_float_column
from style import axis_label as _pretty

# How the train/val/all subsets are named on the plots.
SUBSET_LABELS = {"train": "Training set", "val": "Validation set", "all": ""}


def _subset_label(name):
    return SUBSET_LABELS.get(name, name.capitalize() if name else "")


def save_tensorboard_plots(log_dir, output_dir):
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        fig, ax = plt.subplots(figsize=style.FIG_SINGLE)
        ax.plot(steps, values, linewidth=2, color=style.BACKGROUND_COLOR)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(_pretty(tag))
        style.annotate(ax, tag)
        style.cms_header(ax, data=False)
        style.save(fig, output_dir / tag.replace("/", "_"))


def plot_input_feature_distributions(
    raw_data,
    training_features,
    train_idx,
    val_idx,
    output_dir,
    feature_transforms=None,
    skip_cols=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(raw_data["label"])
    weights = np.asarray(raw_data["weight"]) if "weight" in raw_data else np.ones(data_length(raw_data), dtype=np.float64)
    weights = np.where(weights < 1e4, weights, 0.0)  # safeguard against extreme weights that can break the weighted density calculation

    train_mask = np.zeros(data_length(raw_data), dtype=bool)
    val_mask = np.zeros(data_length(raw_data), dtype=bool)
    train_mask[np.asarray(train_idx, dtype=np.int64)] = True
    val_mask[np.asarray(val_idx, dtype=np.int64)] = True

    training_feature_set = set(training_features)
    feature_transforms = feature_transforms or {}
    skip_cols = set(skip_cols or [])
    skip_cols |= {
        "label", "weight", "dataset_idx", "sample_idx", "split",
        "dnn_score", "dnn_0_score", "dnn_1_score",
    }

    def _safe_filename(name):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

    def _weighted_density(values, weights, bins):
        counts, edges = np.histogram(values, bins=bins, weights=weights, density=False)
        total = np.sum(counts)
        if total > 0:
            counts = counts / total
        return counts, edges

    def _numeric_full_column(arr):
        """Full-length float64 column, or None for non-numeric columns."""
        arr = np.asarray(arr)
        if arr.dtype != object and not np.issubdtype(arr.dtype, np.number):
            return None
        try:
            return to_flat_float_column(arr)
        except Exception:
            return None

    # Signal/background carry their analysis colours; train/val are told apart by
    # linestyle rather than a second pair of hues, so the plot stays readable in print.
    series_style = {
        ("bkg", "train"): (style.BACKGROUND_COLOR, "-"),
        ("bkg", "val"): (style.BACKGROUND_COLOR, "--"),
        ("sig", "train"): (style.SIGNAL_COLOR, "-"),
        ("sig", "val"): (style.SIGNAL_COLOR, "--"),
    }

    for feat in sorted(k for k in raw_data.keys() if k not in skip_cols):
        full_arr = _numeric_full_column(raw_data[feat])
        if full_arr is None:
            logging.info("Skipping non-numeric feature '%s' for input plotting", feat)
            continue

        valid_mask = np.isfinite(full_arr)
        if valid_mask.sum() < 2:
            logging.info("Skipping feature '%s' because it has fewer than 2 finite values", feat)
            continue

        plot_vals = full_arr[valid_mask]
        plot_labels = labels[valid_mask]
        plot_weights = weights[valid_mask]
        plot_train_mask = train_mask[valid_mask]
        plot_val_mask = val_mask[valid_mask]

        vmin = np.min(plot_vals)
        vmax = np.max(plot_vals)
        if vmin == vmax:
            eps = 0.5 if vmin == 0 else 0.05 * abs(vmin)
            vmin -= eps
            vmax += eps

        bins = np.linspace(vmin, vmax, 51)

        fig, ax = plt.subplots(figsize=style.FIG_SINGLE)

        series = [
            ("sig", "train", (plot_labels == 1) & plot_train_mask, "Signal (train)"),
            ("sig", "val",   (plot_labels == 1) & plot_val_mask,   "Signal (val)"),
            ("bkg", "train", (plot_labels == 0) & plot_train_mask, "Background (train)"),
            ("bkg", "val",   (plot_labels == 0) & plot_val_mask,   "Background (val)"),
        ]

        drew_any = False
        for cls_name, split_name, mask, legend_label in series:
            vals = plot_vals[mask]
            wts = plot_weights[mask]
            if len(vals) == 0 or np.sum(wts) <= 0:
                continue

            counts, edges = _weighted_density(vals, wts, bins=bins)
            color, linestyle = series_style[(cls_name, split_name)]
            ax.stairs(counts, edges, linewidth=2, color=color, linestyle=linestyle,
                      label=legend_label)
            drew_any = True

        if not drew_any:
            plt.close(fig)
            logging.info("Skipping feature '%s' because no drawable backgrounds were found", feat)
            continue

        ax.set_xlabel(_pretty(feat))
        ax.set_ylabel("Fraction of events")
        ax.set_xlim(bins[0], bins[-1])
        style.headroom(ax, factor=1.55)

        note = [style.CONTEXT.extra]
        if feat in training_feature_set:
            transform = feature_transforms.get(feat, "none")
            note.append("Training input" + (f" (transform: {transform})" if transform != "none" else ""))
        style.annotate(ax, note)
        style.legend(ax, loc="upper right")
        style.cms_header(ax, data=False)

        style.save(fig, output_dir / f"inputs_{_safe_filename(feat)}")


def plot_constraint_var_distribution(data, constraint_var, train_idx, val_idx, output_path):
    labels = np.asarray(data["label"])
    constraint = np.asarray(data[constraint_var])
    weights = np.asarray(data["weight"]) if "weight" in data else None

    fig, axes = plt.subplots(
        2, 2, figsize=(19, 12), sharex="col",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07, "wspace": 0.22},
    )

    for col, (mask, title) in enumerate([
        (labels == 1, "Signal"),
        (labels == 0, "Background"),
    ]):
        ax_main = axes[0, col]
        ax_ratio = axes[1, col]

        all_idx = np.where(mask)[0]
        train_mask_idx = np.intersect1d(all_idx, train_idx)
        val_mask_idx = np.intersect1d(all_idx, val_idx)

        all_vals = constraint[all_idx]
        train_vals = constraint[train_mask_idx]
        val_vals = constraint[val_mask_idx]

        all_w = weights[all_idx] if weights is not None else None
        train_w = weights[train_mask_idx] if weights is not None else None
        val_w = weights[val_mask_idx] if weights is not None else None

        bins = np.linspace(np.min(all_vals), np.max(all_vals), 51)

        all_counts, _ = np.histogram(all_vals, bins=bins, weights=all_w, density=False)
        train_counts, _ = np.histogram(train_vals, bins=bins, weights=train_w, density=False)
        val_counts, _ = np.histogram(val_vals, bins=bins, weights=val_w, density=False)

        all_counts_norm = all_counts / (all_counts.sum() + 1e-12)
        train_counts_norm = train_counts / (train_counts.sum() + 1e-12)
        val_counts_norm = val_counts / (val_counts.sum() + 1e-12)

        for counts, color, linestyle, label in [
            (all_counts_norm, style.DATA_COLOR, "-", "All"),
            (train_counts_norm, style.TRAIN_COLOR, "--", "Train"),
            (val_counts_norm, style.VAL_COLOR, "-.", "Validation"),
        ]:
            ax_main.stairs(counts, bins, linewidth=2, color=color, linestyle=linestyle, label=label)

        ax_main.set_ylabel("Fraction of events")
        ax_main.set_xlim(bins[0], bins[-1])
        style.headroom(ax_main, factor=1.5)
        style.annotate(ax_main, [title, style.CONTEXT.extra])
        style.legend(ax_main, loc="upper right")
        style.cms_header(ax_main, data=False)

        ratio = np.where(train_counts_norm > 1e-12, val_counts_norm / train_counts_norm, np.nan)
        ax_ratio.stairs(ratio, bins, linewidth=2, color=style.DATA_COLOR)
        ax_ratio.axhline(1.0, color=style.NEUTRAL_COLOR, linewidth=1.5, linestyle="--")
        ax_ratio.set_xlabel(_pretty(constraint_var))
        ax_ratio.set_ylabel("Val. / train", fontsize=20)
        ax_ratio.set_xlim(bins[0], bins[-1])
        ax_ratio.set_ylim(0, 2)
        ax_ratio.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="both"))

    style.save(fig, output_path)


def plot_weight_distributions(sig_data, bkg_data, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={"wspace": 0.22})

    for ax, data, title, color in [
        (axes[0], sig_data, "Signal", style.SIGNAL_COLOR),
        (axes[1], bkg_data, "Background", style.BACKGROUND_COLOR),
    ]:
        ax.hist(np.asarray(data["weight"]), bins=50, histtype="step", linewidth=2, color=color)
        ax.set_xlabel("Event weight")
        ax.set_ylabel("Events")
        ax.set_yscale("log")
        style.annotate(ax, [title, style.CONTEXT.extra])
        style.cms_header(ax, data=False)

    style.save(fig, output_path)


def _profile_overlay(ax, x, y, bins, xrange, weights=None, color=style.PROFILE_COLOR,
                     label="Profile mean"):
    # Markers sit on a light->dark colormap, so no single colour contrasts with every
    # cell underneath. A white outline separates them from the dark end; the marker
    # colour itself carries the light end.
    ring = [matplotlib.patheffects.withStroke(linewidth=2.5, foreground="white")]

    bin_edges = np.linspace(xrange[0], xrange[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    means, errors = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (x >= lo) & (x < hi)
        vals = y[mask]
        w = weights[mask] if weights is not None else None
        if len(vals) > 0:
            mean = np.average(vals, weights=w)
            variance = np.average((vals - mean) ** 2, weights=w)
            n_eff = (np.sum(w) ** 2 / np.sum(w ** 2)) if w is not None else len(vals)
            errors.append(np.sqrt(variance / n_eff))
            means.append(mean)
        else:
            means.append(np.nan)
            errors.append(np.nan)
    means = np.array(means)
    errors = np.array(errors)
    container = ax.errorbar(bin_centers, means, yerr=errors, color=color,
                            fmt="o", markersize=5, linewidth=1.5, label=label, zorder=5)
    for artist in (container.lines[0], *container.lines[2]):
        if artist is not None:
            artist.set_path_effects(ring)


def plot_abcd_plane(data, flavor, constraint_var, output_path, title_suffix=""):
    labels = np.asarray(data["label"])
    background = {k: np.asarray(v)[labels == 0] for k, v in data.items()}

    score_col = score_column(flavor)

    bins = [50, 50]
    dnn_range = (0, 1)
    constrain_var_range = (0, max(np.asarray(data[constraint_var])))

    fig, ax = plt.subplots(figsize=style.FIG_PLANE)

    h = ax.hist2d(background[score_col], background[constraint_var], bins=bins,
                  range=[dnn_range, constrain_var_range], cmap=style.CMAP_MC,
                  norm=matplotlib.colors.LogNorm(), rasterized=True)
    style.colorbar(fig, h[3], ax, label="Events")

    _profile_overlay(ax, background[score_col], background[constraint_var], bins[0], dnn_range)
    _profile_overlay(ax, background[score_col], background[constraint_var], bins[0], dnn_range,
                     weights=background["weight"] if "weight" in background else None,
                     color=style.PROFILE_COLOR_WEIGHTED, label="Profile mean (weighted)")

    ax.set_xlabel(_pretty(score_col))
    ax.set_ylabel(_pretty(constraint_var))
    ax.set_xlim(dnn_range)
    ax.set_ylim(constrain_var_range)
    style.annotate(ax, ["Background MC", style.CONTEXT.extra, _subset_label(title_suffix)])
    style.legend(ax, loc="lower right")
    style.cms_header(ax, data=False)

    style.save(fig, output_path)


def plot_abcd_plane_data(data, flavor, constraint_var, output_path, blind_threshold=0.8, title_suffix=""):
    score = np.asarray(data[score_column(flavor)])
    constraint = np.asarray(data[constraint_var])
    weights = np.asarray(data["weight"]) if "weight" in data else None

    bins = [50, 50]
    dnn_range = (0, 1)
    constrain_var_range = (0, max(constraint))

    counts, xedges, yedges = np.histogram2d(
        score, constraint, bins=bins,
        range=[dnn_range, constrain_var_range], weights=weights,
    )

    # Blank any cell that reaches past the threshold on either axis, not just those fully
    # inside it, so a bin straddling the boundary cannot leak signal-region yield.
    blind_x = xedges[1:] > blind_threshold
    blind_y = yedges[1:] > blind_threshold
    counts[blind_x, :] = np.nan
    counts[:, blind_y] = np.nan

    fig, ax = plt.subplots(figsize=style.FIG_PLANE)
    mesh = ax.pcolormesh(
        xedges, yedges,
        np.ma.masked_where(~np.isfinite(counts) | (counts <= 0), counts).T,
        cmap=style.CMAP_DATA, norm=matplotlib.colors.LogNorm(), rasterized=True,
    )
    style.colorbar(fig, mesh, ax, label="Events")

    n_open = int(np.argmax(blind_x)) if blind_x.any() else bins[0]
    y_open = yedges[int(np.argmax(blind_y))] if blind_y.any() else yedges[-1]
    if n_open > 0:
        open_range = (xedges[0], xedges[n_open])
        # Cut on both axes rather than leaning on the binning range to drop the high-DNN
        # events, so nothing blinded is handed to the profile in the first place.
        visible = (score < xedges[n_open]) & (constraint < y_open)
        _profile_overlay(ax, score[visible], constraint[visible], n_open, open_range,
                         color=style.PROFILE_COLOR)
        _profile_overlay(ax, score[visible], constraint[visible], n_open, open_range,
                         weights=weights[visible] if weights is not None else None,
                         color=style.PROFILE_COLOR_WEIGHTED, label="Profile mean (weighted)")

    # Blinding on either axis leaves an L: the full high-DNN strip, plus the
    # high-constraint strip beside it.
    x0 = xedges[int(np.argmax(blind_x))] if blind_x.any() else xedges[-1]
    for rx, ry, rw, rh in (
        (x0, yedges[0], xedges[-1] - x0, yedges[-1] - yedges[0]),
        (xedges[0], y_open, x0 - xedges[0], yedges[-1] - y_open),
    ):
        if rw > 0 and rh > 0:
            ax.add_patch(plt.Rectangle(
                (rx, ry), rw, rh,
                facecolor="none", edgecolor=style.NEUTRAL_COLOR, hatch="//", linewidth=1.0, zorder=3,
            ))
    if blind_x.any() and blind_y.any():
        ax.text(0.5 * (x0 + xedges[-1]), 0.5 * (y_open + yedges[-1]), "Blinded",
                ha="center", va="center", color=style.NEUTRAL_COLOR, fontsize=20,
                rotation=90, zorder=4)

    ax.set_xlim(dnn_range)
    ax.set_ylim(constrain_var_range)
    ax.set_xlabel(_pretty(score_column(flavor)))
    ax.set_ylabel(_pretty(constraint_var))
    style.annotate(ax, [style.CONTEXT.extra, _subset_label(title_suffix)])
    style.legend(ax, loc="lower right")
    style.cms_header(ax, data=True)

    style.save(fig, output_path)


def plot_decorrelation_check(data, flavor, constraint_var, output_path, title_suffix=""):
    labels = np.asarray(data["label"])
    background = {k: np.asarray(v)[labels == 0] for k, v in data.items()}
    score_col = score_column(flavor)

    score_bins = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, ax = plt.subplots(figsize=style.FIG_SINGLE)
    for i in range(len(score_bins) - 1):
        mask = (background[score_col] >= score_bins[i]) & (background[score_col] < score_bins[i + 1])
        ax.hist(background[constraint_var][mask], bins=50, density=True,
                histtype="step", linewidth=2,
                color=style.PETROFF_6[i % len(style.PETROFF_6)],
                linestyle=style.LINESTYLES[i % len(style.LINESTYLES)],
                label=f"{score_bins[i]:.2f} $\\leq$ {_pretty(score_col)} $<$ {score_bins[i + 1]:.2f}")

    ax.set_xlabel(_pretty(constraint_var))
    ax.set_ylabel("Normalised events")
    style.headroom(ax, factor=1.6)
    style.annotate(ax, ["Background MC", style.CONTEXT.extra, _subset_label(title_suffix)])
    style.legend(ax, loc="upper right", fontsize=16)
    style.cms_header(ax, data=False)

    style.save(fig, output_path)


def plot_roc_curves(data, flavor, output_path):
    labels = np.asarray(data["label"])
    weights = np.asarray(data["weight"]) if "weight" in data else None

    fig, ax = plt.subplots(figsize=(9, 9))

    roc_series = [
        ("DNN", "dnn_score")
    ] if flavor == "single" else [
        ("DNN 0", "dnn_0_score"),
        ("DNN 1", "dnn_1_score"),
    ]

    for i, (label_name, score_col) in enumerate(roc_series):
        fpr, tpr, _ = roc_curve(labels, np.asarray(data[score_col]), sample_weight=weights)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.5, color=style.PETROFF_6[i % len(style.PETROFF_6)],
                linestyle=style.LINESTYLES[i % len(style.LINESTYLES)],
                label=f"{label_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.5, color=style.NEUTRAL_COLOR,
            label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    style.annotate(ax, style.CONTEXT.extra)
    style.legend(ax, loc="lower right")
    style.cms_header(ax, data=False)

    style.save(fig, output_path)


def plot_score_densities(data, flavor, output_path):
    score_cols = ["dnn_score"] if flavor == "single" else ["dnn_0_score", "dnn_1_score"]
    fig, axes = plt.subplots(1, len(score_cols), figsize=(9.5 * len(score_cols), 8),
                             squeeze=False, gridspec_kw={"wspace": 0.22})

    labels = np.asarray(data["label"])
    sig_mask = labels == 1
    bkg_mask = labels == 0

    if "weight" in data:
        weights = np.asarray(data["weight"])
        sig_w = weights[sig_mask]
        bkg_w = weights[bkg_mask]
    else:
        sig_w = None
        bkg_w = None

    for idx, col in enumerate(score_cols):
        ax = axes[0, idx]
        values = np.asarray(data[col])
        for mask, w, label, color, linestyle in [
            (sig_mask, sig_w, "Signal", style.SIGNAL_COLOR, "-"),
            (bkg_mask, bkg_w, "Background", style.BACKGROUND_COLOR, "--"),
        ]:
            ax.hist(values[mask], bins=50, range=(0, 1), weights=w, density=True,
                    histtype="step", linewidth=2.5, label=label, color=color,
                    linestyle=linestyle)
        ax.set_xlabel(_pretty(col))
        ax.set_ylabel("Normalised events")
        ax.set_xlim(0, 1)
        style.headroom(ax, factor=1.5)
        style.annotate(ax, style.CONTEXT.extra)
        style.legend(ax, loc="upper center")
        style.cms_header(ax, data=False)

    style.save(fig, output_path)


# Plot permutation importance as a horizontal bar chart, sorted by importance.
# Features that hurt the AUC most when shuffled appear at the top.
def plot_permutation_importance(baseline_auc, importances, output_path):
    sorted_feats = sorted(importances, key=importances.get)
    sorted_vals = [importances[f] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_feats) * 0.42)))
    colors = [style.SIGNAL_COLOR if v > 0 else style.BACKGROUND_COLOR for v in sorted_vals]
    ax.barh([_pretty(f) for f in sorted_feats], sorted_vals, color=colors)
    ax.axvline(0, color=style.DATA_COLOR, linewidth=1.2)
    ax.set_xlabel("AUC loss when the input is shuffled")
    ax.tick_params(axis="y", labelsize=16, length=0)
    ax.tick_params(axis="y", which="minor", length=0)
    style.annotate(ax, style.CONTEXT.extra)
    style.annotate(ax, f"Baseline AUC = {baseline_auc:.3f}", x=0.96, y=0.05, ha="right", va="bottom")
    style.cms_header(ax, data=False)

    style.save(fig, output_path)
