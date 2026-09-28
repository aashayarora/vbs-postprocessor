import os

import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import style

def to_threshold_bin_idx(vals, thresholds):
    """
    Pre-bin continuous values into index arrays using thresholds.
    """
    return np.searchsorted(thresholds, vals, side="right") - 1

def threshold_yields_2d_from_idx(d_idx, v_idx, w, n_dnn, n_vbs):
    """
    Computes cumulative 2D yields for >= threshold in both axes.
    """
    if d_idx.size == 0:
        return np.zeros((n_dnn, n_vbs), dtype=float)

    flat_idx = d_idx * n_vbs + v_idx
    H_flat = np.bincount(flat_idx, weights=w, minlength=n_dnn * n_vbs)
    H = H_flat.reshape(n_dnn, n_vbs)

    # cumulative yields for >= threshold in both axes
    return H[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]

def optimize_cuts(df_sig, df_bkg, 
                  var1_col="boosted_h_candidate_score", var2_col="boosted_v_candidate_score", var3_col=None,
                  disco1_col="dnn_0_score", disco2_col="dnn_1_score",
                  cut_var1_list=None, cut_var2_list=None, cut_var3_list=None,
                  cut_disco1_list=None, cut_disco2_list=None,
                  combine_method="and",
                  df_data=None, use_data_driven_bkg=False, min_data_yield=0.0):
    """
    Optimizes 5D cuts to maximize significance.

    The background is the MC in ``df_bkg``. ``df_data`` / ``min_data_yield`` belong to
    the data-driven mode (background = B*C/D from the data control regions), which
    run_analysis.py exposes as --data-driven but which is not implemented here yet.
    """
    if use_data_driven_bkg:
        raise NotImplementedError(
            "optimize_cuts: the data-driven background (B*C/D from data) is not implemented; "
            "run without --data-driven to optimize against the MC background")
    if cut_var1_list is None:
        cut_var1_list = np.linspace(0, 1, 21)
    if cut_var2_list is None:
        cut_var2_list = np.linspace(0, 1, 21)
    if cut_var3_list is None:
        if var3_col is not None:
            cut_var3_list = np.linspace(0, 1, 21)
        else:
            cut_var3_list = [None]
    if cut_disco1_list is None:
        cut_disco1_list = np.linspace(0, 1, 101)
    if cut_disco2_list is None:
        cut_disco2_list = np.linspace(0, 1, 101)

    n_dnn = len(cut_disco1_list)
    n_vbs = len(cut_disco2_list)

    sig_h = df_sig[var1_col].to_numpy()
    sig_v = df_sig[var2_col].to_numpy()
    if var3_col is not None:
        sig_3 = df_sig[var3_col].to_numpy()
    sig_d = df_sig[disco1_col].to_numpy()
    sig_j = df_sig[disco2_col].to_numpy()
    sig_w = df_sig["weight"].to_numpy()

    bkg_h = df_bkg[var1_col].to_numpy()
    bkg_v = df_bkg[var2_col].to_numpy()
    if var3_col is not None:
        bkg_3 = df_bkg[var3_col].to_numpy()
    bkg_d = df_bkg[disco1_col].to_numpy()
    bkg_j = df_bkg[disco2_col].to_numpy()
    bkg_w = df_bkg["weight"].to_numpy()

    # Precompute threshold comparisons once
    sig_h_ge = sig_h[:, None] >= cut_var1_list[None, :]
    sig_v_ge = sig_v[:, None] >= cut_var2_list[None, :]
    if var3_col is not None:
        sig_3_ge = sig_3[:, None] >= cut_var3_list[None, :]
    
    bkg_h_ge = bkg_h[:, None] >= cut_var1_list[None, :]
    bkg_v_ge = bkg_v[:, None] >= cut_var2_list[None, :]
    if var3_col is not None:
        bkg_3_ge = bkg_3[:, None] >= cut_var3_list[None, :]

    sig_d_idx = to_threshold_bin_idx(sig_d, cut_disco1_list)
    sig_j_idx = to_threshold_bin_idx(sig_j, cut_disco2_list)
    bkg_d_idx = to_threshold_bin_idx(bkg_d, cut_disco1_list)
    bkg_j_idx = to_threshold_bin_idx(bkg_j, cut_disco2_list)

    sig_valid = (sig_d_idx >= 0) & (sig_j_idx >= 0)
    bkg_valid = (bkg_d_idx >= 0) & (bkg_j_idx >= 0)

    best_significance = -np.inf
    best_tuple = None

    for i, cut_var1 in enumerate(cut_var1_list):
        sig_mask_h = sig_h_ge[:, i]
        bkg_mask_h = bkg_h_ge[:, i]

        for j, cut_var2 in enumerate(cut_var2_list):
            sig_mask_hv = sig_mask_h & sig_v_ge[:, j] if combine_method == "and" else sig_mask_h | sig_v_ge[:, j]
            bkg_mask_hv = bkg_mask_h & bkg_v_ge[:, j] if combine_method == "and" else bkg_mask_h | bkg_v_ge[:, j]

            for k, cut_var3 in enumerate(cut_var3_list):
                if var3_col is not None:
                    if combine_method == "and":
                        sig_mask = (sig_mask_hv & sig_3_ge[:, k]) & sig_valid
                        bkg_mask = (bkg_mask_hv & bkg_3_ge[:, k]) & bkg_valid
                    elif combine_method == "or":
                        sig_mask = (sig_mask_hv | sig_3_ge[:, k]) & sig_valid
                        bkg_mask = (bkg_mask_hv | bkg_3_ge[:, k]) & bkg_valid
                    else:
                        raise ValueError("combine_method must be 'and' or 'or'")
                else:
                    sig_mask = sig_mask_hv & sig_valid
                    bkg_mask = bkg_mask_hv & bkg_valid

                if not sig_mask.any() and not bkg_mask.any():
                    continue

                Y_sig = threshold_yields_2d_from_idx(
                    sig_d_idx[sig_mask], sig_j_idx[sig_mask], sig_w[sig_mask], n_dnn, n_vbs
                )
                Y_bkg = threshold_yields_2d_from_idx(
                    bkg_d_idx[bkg_mask], bkg_j_idx[bkg_mask], bkg_w[bkg_mask], n_dnn, n_vbs
                )

                s_over_b = np.divide(Y_sig, Y_bkg, out=np.zeros_like(Y_sig), where=Y_bkg > 0)
                term = np.where(
                    Y_bkg > 0,
                    (Y_sig + Y_bkg) * np.log1p(s_over_b) - Y_sig,
                    0.0,
                )
                S = np.sqrt(np.maximum(2.0 * term, 0.0))

                idx = np.unravel_index(np.argmax(S), S.shape)
                current_significance = S[idx]
                if current_significance > best_significance:
                    best_significance = current_significance
                    best_tuple = (cut_var1, cut_var2, cut_var3, cut_disco1_list[idx[0]], cut_disco2_list[idx[1]])

    if best_tuple is None:
        # Default or fallback if none is found
        return -np.inf, (None, None, None, None, None)

    return best_significance, best_tuple

def get_ABCD_regions(df, cut_var1, cut_var2, cut_disco1, cut_disco2,
                     var1_col="boosted_h_candidate_score", var2_col="boosted_v_candidate_score",
                     disco1_col="dnn_0_score", disco2_col="dnn_1_score",
                     combine_method="and", var3_col=None, cut_var3=None):
    """
    Computes ABCD region yields given the optimized cut thresholds. A third tagger
    (``var3_col``, e.g. the second V candidate in 3fj) is combined like the other two.
    """
    # Apply initial kinematic cuts if applicable
    passes = [df[var1_col] >= cut_var1, df[var2_col] >= cut_var2]
    if var3_col is not None:
        passes.append(df[var3_col] >= cut_var3)
    if combine_method == "and":
        df_filtered = df[np.logical_and.reduce(passes)]
    elif combine_method == "or":
        df_filtered = df[np.logical_or.reduce(passes)]
    else:
        raise ValueError("combine_method must be 'and' or 'or'")
    
    A = df_filtered[(df_filtered[disco1_col] >= cut_disco1) & (df_filtered[disco2_col] >= cut_disco2)].weight.sum()
    B = df_filtered[(df_filtered[disco1_col] >= cut_disco1) & (df_filtered[disco2_col] < cut_disco2)].weight.sum()
    C = df_filtered[(df_filtered[disco1_col] < cut_disco1) & (df_filtered[disco2_col] >= cut_disco2)].weight.sum()
    D = df_filtered[(df_filtered[disco1_col] < cut_disco1) & (df_filtered[disco2_col] < cut_disco2)].weight.sum()
    
    return A, B, C, D


def _weighted_stat(values, weights, statistic):
    """Weighted median or mean of ``values`` (NaN when empty)."""
    if len(values) == 0 or np.sum(weights) <= 0:
        return np.nan
    if statistic == "mean":
        return np.average(values, weights=weights)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])]


def plot_background_decorrelation(df_bkg_kin, cut_dnn, cut_bdt, out_dir, scan_idx,
                                  disco1_col="dnn_score", disco2_col="bdt_score",
                                  df_data_kin=None, use_data_driven=False,
                                  profile_name="background_dnn_vs_vbs_median_profile"):
    """Background in one scan's ABCD plane, and the profile of disco2 in bins of disco1
    (flat when the two are decorrelated). In data-driven mode both are drawn from data
    with region A (above both cuts) removed, so the signal region stays blind.
    ``profile_name`` names the profile plot and sets its statistic (median unless it
    contains "mean")."""
    use_data = use_data_driven and df_data_kin is not None
    df = df_data_kin if use_data else df_bkg_kin
    x = df[disco1_col].to_numpy()
    y = df[disco2_col].to_numpy()
    w = np.ones(len(df)) if use_data else df["weight"].to_numpy()
    if use_data:
        blind = (x > cut_dnn) & (y > cut_bdt)
        x, y, w = x[~blind], y[~blind], w[~blind]
    label = "Data, A blind" if use_data else "Background MC"
    scan = f"scan_{scan_idx + 1}"

    fig, ax = plt.subplots(figsize=style.FIG_PLANE)
    counts, xedges, yedges = np.histogram2d(x, y, bins=100, weights=w)
    mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_where(counts <= 0, counts).T,
                         cmap="Blues", norm=matplotlib.colors.LogNorm(), rasterized=True)
    style.colorbar(fig, mesh, ax, label="Events")
    ax.axvline(cut_dnn, color=style.DATA_COLOR, linestyle="--", linewidth=2)
    ax.axhline(cut_bdt, color=style.DATA_COLOR, linestyle="--", linewidth=2)
    ax.set_xlabel(style.axis_label(disco1_col))
    ax.set_ylabel(style.axis_label(disco2_col))
    style.annotate(ax, [label, style.CONTEXT.extra, f"Scan {scan_idx + 1}"])
    style.cms_header(ax, data=use_data)
    style.save(fig, os.path.join(out_dir, f"{'data' if use_data else 'background'}_abcd_regions_{scan}"))

    statistic = "mean" if "mean" in profile_name else "median"
    edges = np.linspace(np.min(x), np.max(x), 21) if len(x) else np.linspace(0.0, 1.0, 21)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, len(centers) - 1)
    profile = np.array([_weighted_stat(y[bin_idx == i], w[bin_idx == i], statistic)
                        for i in range(len(centers))])

    fig, ax = plt.subplots(figsize=style.FIG_SINGLE)
    ax.plot(centers, profile, marker="o", linewidth=2, color=style.BACKGROUND_COLOR,
            label=f"{statistic.capitalize()} {style.axis_label(disco2_col)}")
    ax.axvline(cut_dnn, color=style.DATA_COLOR, linestyle="--", linewidth=2)
    ax.axhline(cut_bdt, color=style.DATA_COLOR, linestyle=":", linewidth=2)
    ax.set_xlabel(style.axis_label(disco1_col))
    ax.set_ylabel(f"{statistic.capitalize()} {style.axis_label(disco2_col)}")
    style.headroom(ax, factor=1.4)
    style.annotate(ax, [label, style.CONTEXT.extra, f"Scan {scan_idx + 1}"])
    style.legend(ax, loc="upper right")
    style.cms_header(ax, data=use_data)
    style.save(fig, os.path.join(out_dir, f"{profile_name}_{scan}"))
