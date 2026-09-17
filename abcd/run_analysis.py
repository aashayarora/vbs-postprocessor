"""Cut optimization and ABCD yields for one analysis channel.

Replaces the former 1fj_analysis.py / 2fj_analysis.py / 3fj_analysis.py, which
were copies of each other differing only in the channel definition (taggers and
how their cuts combine); that now lives in channels.py.
"""

import argparse
import os

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import yaml

import style
from analysis import get_ABCD_regions, optimize_cuts, plot_background_decorrelation
from channels import CHANNELS
from predictions import read_predictions

DNN_COL = "dnn_score"
BDT_COL = "bdt_score"

# Human-readable channel names for the in-frame plot annotation.
CHANNEL_LABELS = {"1fj": "1 AK8 jet", "2fj": "2 AK8 jets", "3fj": "3 AK8 jets"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ABCD cut optimization for one channel")
    parser.add_argument("-c", "--channel", required=True, choices=list(CHANNELS),
                        help="Analysis channel (sets taggers, preselection, and cut combination)")
    parser.add_argument("-i", "--input", required=True, help="Path to the input predictions file (.parquet)")
    parser.add_argument("-d", "--data", required=False, help="Path to the data predictions file (.parquet, optional)")
    parser.add_argument("-n", "--num-scans", type=int, default=1, help="Number of orthogonal scans to perform")
    parser.add_argument("-dd", "--data-driven", action="store_true",
                        help="Optimize using the data-driven ABCD background estimate (B*C/D from data control regions) instead of MC background. Requires --data. Region A of the data is never used.")
    parser.add_argument("--min-data-yield", type=float, default=0.0,
                        help="Minimum (weighted) yield required in each data control region B, C, D for a cell to be eligible (data-driven mode only).")
    style.add_cli_args(parser)
    args = parser.parse_args()

    if args.data_driven and not args.data:
        parser.error("--data-driven requires --data (the data control regions provide the background estimate)")
    return args


def channel_kwargs(channel):
    """Column kwargs for optimize_cuts/get_ABCD_regions from the channel spec."""
    cols = channel.tagger_cols
    kwargs = dict(
        var1_col=cols[0],
        var2_col=cols[1],
        var3_col=cols[2] if len(cols) > 2 else None,
        disco1_col=DNN_COL,
        disco2_col=BDT_COL,
        combine_method=channel.combine,
    )
    return kwargs


def main():
    args = parse_args()
    channel = CHANNELS[args.channel]
    n_taggers = len(channel.taggers)
    style.configure_from_args(args, extra=CHANNEL_LABELS.get(args.channel, args.channel))

    out_dir = os.path.dirname(args.input) or "."

    presel = channel.preselection_query(threshold=0.1)
    df = read_predictions(args.input).query(presel)
    df_data = read_predictions(args.data).query(presel) if args.data else None

    # ABCD cut optimization runs on the nominal event set only; JES/JER variations, if
    # present in the predictions, are consumed later by the datacards, not here.
    if "variation" in df.columns:
        df = df[df["variation"] == "nominal"]
    if df_data is not None and "variation" in df_data.columns:
        df_data = df_data[df_data["variation"] == "nominal"]

    previous_cuts = []
    scan_cuts = {}

    total_sig = np.zeros(4)
    total_bkg = np.zeros(4)
    total_data = np.zeros(4)  # [A_pred, B, C, D]

    for scan_idx in range(args.num_scans):
        print(f"\n{'='*50}")
        print(f"Starting Scan {scan_idx + 1}/{args.num_scans}")
        print(f"{'='*50}")

        df_sig = df[df["label"] == 1]
        df_bkg = df[df["label"] == 0]

        if len(df_sig) == 0 or len(df_bkg) == 0:
            print("Not enough signal or background events left for further optimization. Stopping.")
            break

        kwargs = channel_kwargs(channel)
        tagger_scan = np.linspace(0.1, 1.0, 46)
        sig, best = optimize_cuts(
            df_sig,
            df_bkg,
            cut_var1_list=tagger_scan,
            cut_var2_list=tagger_scan,
            cut_var3_list=tagger_scan if n_taggers > 2 else None,
            cut_disco1_list=np.linspace(0., 1.0, 51),
            cut_disco2_list=np.linspace(0., 1.0, 51),
            df_data=df_data if args.data_driven else None,
            use_data_driven_bkg=args.data_driven,
            min_data_yield=args.min_data_yield,
            **kwargs,
        )
        if args.data_driven:
            print("(Optimized against data-driven ABCD background estimate B*C/D)")

        if sig == -np.inf:
            print("No valid cuts could be found. Stopping.")
            break

        cut_var1, cut_var2, cut_var3, cut_dnn, cut_bdt = best
        tagger_cuts = [cut_var1, cut_var2, cut_var3][:n_taggers]
        joiner = " AND " if channel.combine == "and" else " OR "

        print(f"Optimal Cuts (Scan {scan_idx + 1}):")
        if previous_cuts:
            print("  Orthogonality constraints applied:")
            for prev in previous_cuts:
                clause = joiner.join(
                    f"{col} >= {val:.4f}" for col, val in zip(channel.tagger_cols, prev)
                )
                print(f"    - NOT ({clause})")

        width = max(len(col) for col in channel.tagger_cols + [DNN_COL, BDT_COL])
        for col, val in zip(channel.tagger_cols, tagger_cuts):
            print(f"  {col:<{width}} > {val:.4f}")
        print(f"  {DNN_COL:<{width}} > {cut_dnn:.4f}")
        print(f"  {BDT_COL:<{width}} > {cut_bdt:.4f}")
        print("-" * 50)
        print(f"Significance at optimal cuts: {sig:.4f}")
        print("=" * 50)

        region_kwargs = dict(kwargs)
        if n_taggers > 2:
            region_kwargs["cut_var3"] = cut_var3
        else:
            region_kwargs.pop("var3_col")
        sig_regions = get_ABCD_regions(df_sig, cut_var1, cut_var2, cut_dnn, cut_bdt, **region_kwargs)
        bkg_regions = get_ABCD_regions(df_bkg, cut_var1, cut_var2, cut_dnn, cut_bdt, **region_kwargs)

        # Background decorrelation plots. In --data-driven mode these are drawn from
        # data with the signal region (A) kept blind; otherwise from background MC.
        kin_query = channel.kinematic_query(tagger_cuts)
        df_bkg_kin = df_bkg.query(kin_query)
        df_data_kin = df_data.query(kin_query) if (df_data is not None and args.data_driven) else None
        plot_background_decorrelation(
            df_bkg_kin, cut_dnn, cut_bdt, out_dir, scan_idx,
            disco1_col=DNN_COL, disco2_col=BDT_COL,
            df_data_kin=df_data_kin, use_data_driven=args.data_driven,
            profile_name=channel.profile_name,
        )

        df_sig_kin = df_sig.query(kin_query)
        fig, ax = plt.subplots(figsize=style.FIG_PLANE)
        counts, xedges, yedges = np.histogram2d(
            df_sig_kin[DNN_COL].to_numpy(), df_sig_kin[BDT_COL].to_numpy(),
            bins=100, weights=df_sig_kin["weight"].to_numpy(),
        )
        mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_where(counts <= 0, counts).T,
                             cmap="Reds", norm=matplotlib.colors.LogNorm(), rasterized=True)
        style.colorbar(fig, mesh, ax, label="Events")
        ax.axvline(cut_dnn, color=style.DATA_COLOR, linestyle="--", linewidth=2)
        ax.axhline(cut_bdt, color=style.DATA_COLOR, linestyle="--", linewidth=2)
        ax.set_xlabel(style.axis_label(DNN_COL))
        ax.set_ylabel(style.axis_label(BDT_COL))
        style.annotate(ax, [
            "Signal", style.CONTEXT.extra, f"Scan {scan_idx + 1}",
            f"{style.axis_label(DNN_COL)} = {cut_dnn:.3f}",
            f"{style.axis_label(BDT_COL)} = {cut_bdt:.3f}",
        ])
        style.cms_header(ax, data=False)
        style.save(fig, os.path.join(out_dir, f"signal_abcd_regions_scan_{scan_idx + 1}"))

        print(f"\nSignal Regions Yields (Scan {scan_idx + 1}):\n"
              f"A: {sig_regions[0]:.4f}, B: {sig_regions[1]:.4f}, C: {sig_regions[2]:.4f}, D: {sig_regions[3]:.4f}")
        print(f"\nBackground Regions Yields (Scan {scan_idx + 1}):\n"
              f"A: {bkg_regions[0]:.4f}, B: {bkg_regions[1]:.4f}, C: {bkg_regions[2]:.4f}, D: {bkg_regions[3]:.4f}")

        total_sig += np.asarray(sig_regions)
        total_bkg += np.asarray(bkg_regions)

        if df_data is not None:
            _, b, c, d = get_ABCD_regions(df_data, cut_var1, cut_var2, cut_dnn, cut_bdt, **region_kwargs)
            # Region A of data is blind: the observed yield is never used.
            a_pred = (b * c / d) if d > 0 else float("nan")  # data-driven prediction

            total_data += np.array([a_pred if d > 0 else 0.0, b, c, d])
            print(f"\nData Regions Yields (Scan {scan_idx + 1}):\n"
                  f"A (blind): -1.0000, A_pred (B*C/D): {a_pred:.4f}, B: {b:.4f}, C: {c:.4f}, D: {d:.4f}")

        # Remove events in the current combined kinematic region so the next
        # scan optimizes over an orthogonal event set.
        orth_query = f"~({kin_query})"
        df = df.query(orth_query)
        if df_data is not None:
            df_data = df_data.query(orth_query)

        previous_cuts.append(tagger_cuts)
        scan_cuts[f"Scan{scan_idx + 1}"] = {
            **{key: round(float(val), 4) for key, val in zip(channel.cut_keys, tagger_cuts)},
            "dnn_cut": round(float(cut_dnn), 4),
            "vbs_cut": round(float(cut_bdt), 4),
        }

    regions_path = os.path.join(out_dir, "regions.yaml")
    with open(regions_path, "w") as f:
        yaml.dump({"channels": scan_cuts}, f, default_flow_style=False, sort_keys=False)
    print(f"\nRegions written to {regions_path}")

    print(f"\n{'='*50}")
    print("Summary of Total Yields across all valid scans:")
    print(f"Total Signal Regions Yields:\n"
          f"A: {total_sig[0]:.4f}, B: {total_sig[1]:.4f}, C: {total_sig[2]:.4f}, D: {total_sig[3]:.4f}")
    print(f"\nTotal Background Regions Yields:\n"
          f"A: {total_bkg[0]:.4f}, B: {total_bkg[1]:.4f}, C: {total_bkg[2]:.4f}, D: {total_bkg[3]:.4f}")
    if df_data is not None:
        print(f"\nTotal Data Regions Yields:\n"
              f"A (blind): -1.0000, A_pred (sum B*C/D): {total_data[0]:.4f}, "
              f"B: {total_data[1]:.4f}, C: {total_data[2]:.4f}, D: {total_data[3]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
