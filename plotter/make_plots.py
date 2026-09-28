#!/usr/bin/env python3
"""Config-driven driver: builds a Plotter per configs.json entry and runs it.

Usage:
    python make_plots.py <config-name>          # one config, hists + MVA
    python make_plots.py all                    # every config
    python make_plots.py <name> --only mva      # just the MVA prediction plots
    python make_plots.py --list                 # show available configs
"""

import json
import logging
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

from plotter import Hist1D, Plotter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_PATH = "/ceph/cms/store/user/aaarora/vbsvvh/postprocessing/latest/"

LUMI = {
    "2016preVFP": 19.52,
    "2016postVFP": 16.81,
    "2017": 41.48,
    "2018": 59.83,
    "2022Re-recoBCD": 7.9804,
    "2022Re-recoE+PromptFG": 26.6717,
    "2023PromptC": 18.063,
    "2023PromptD": 9.693,
    "2022": 34.76,
    "2023": 28.28,
    "2024": 109.08,
    "2025": 110.84,
}
LUMI["Run2"] = LUMI["2016preVFP"] + LUMI["2016postVFP"] + LUMI["2017"] + LUMI["2018"]
# Run 3 MC exists only for 2024 (Summer24) and 2025 (Summer24 reused), so 2022/2023 data is not
# processed (the preselection also refuses it until its metadata is split into run eras).
LUMI["Run3"] = LUMI["2024"] + LUMI["2025"]


def resolve_mva_files(base_path, version=None):
    """Find the (mc, data) prediction files under base_path/version_*.

    With ``version`` pinned, look only there; otherwise take the highest
    version that actually contains an MC prediction file.
    """
    base = Path(base_path)
    if not base.is_dir():
        raise FileNotFoundError(f"MVA base_path does not exist: {base}")

    def mva_files(vdir):
        preds = sorted(vdir.glob("predictions_single*.parquet"))
        data = [c for c in preds if c.name.endswith("_data.parquet")]
        mc = [c for c in preds if not c.name.endswith("_data.parquet")]

        # If data exists, find MC with matching checkpoint
        if data:
            data_file = data[0]
            data_name = data_file.name.replace("_data.parquet", "")
            matching_mc = [c for c in mc if c.name == data_name + ".parquet"]
            if matching_mc:
                return matching_mc[0], data_file

        return (mc[0] if mc else None), (data[0] if data else None)

    if version is not None:
        name = str(version)
        if not name.startswith("version_"):
            name = f"version_{name}"
        vdir = base / name
        mc, data = mva_files(vdir)
        if mc is None:
            raise FileNotFoundError(f"No MC prediction file in {vdir}")
        logger.info(f"MVA {base} -> {vdir.name} (pinned; mc={mc.name}, "
                    f"data={data.name if data else None})")
        return str(mc), (str(data) if data else None)

    versions = sorted(
        (d for d in base.glob("version_*")
         if d.is_dir() and re.fullmatch(r"version_\d+", d.name)),
        key=lambda d: int(d.name.split("_")[1]),
        reverse=True,
    )
    for vdir in versions:
        mc, data = mva_files(vdir)
        if mc is not None:
            logger.info(f"MVA {base} -> {vdir.name} (latest; mc={mc.name}, "
                        f"data={data.name if data else None})")
            return str(mc), (str(data) if data else None)
    raise FileNotFoundError(f"No version_* dir with a prediction file under {base}")


def build_hists(entries, default_scale=1.0):
    """Turn the compact JSON hist entries into Hist1D objects.

    Each entry is [var, xlabel, [nbins, xmin, xmax]] with an optional 4th
    element. That element is either a number (signal scale, overriding the
    given default) or an options dict, e.g.
    {"scale": 1000.0, "data_cut": "dnn_score < 0.9"}, where ``data_cut`` is a
    per-histogram pandas query applied to MVA data only (to blind it).
    """
    hists = []
    for entry in entries:
        var, xlabel, binning = entry[0], entry[1], tuple(entry[2])
        scale, data_cut = default_scale, None
        if len(entry) > 3:
            opt = entry[3]
            if isinstance(opt, dict):
                scale = opt.get("scale", default_scale)
                data_cut = opt.get("data_cut")
            else:
                scale = opt
        hists.append(Hist1D(var, xlabel, binning, scale=scale, data_cut=data_cut))
    return hists


def run_config(name, cfg, output_dir, only=None):
    """Build a Plotter for one config and write all its plots."""
    logger.info("=" * 80)
    logger.info(f"Config '{name}' -> {output_dir}")

    # mc_lumi_scale rescales full-Run MC down to one year's lumi; the year is
    # then selected in data via selection_cut instead of cutting the MC.
    mc_lumi_scale = cfg.get("mc_lumi_scale")
    selection_cut = cfg.get("selection_cut")
    if mc_lumi_scale is not None:
        if mc_lumi_scale not in LUMI:
            raise KeyError(
                f"mc_lumi_scale '{mc_lumi_scale}' not in LUMI map; "
                f"available: {', '.join(LUMI)}"
            )
        base_lumi = LUMI[cfg["year"]]
        lumi = LUMI[mc_lumi_scale]
        mc_scale = lumi / base_lumi
        cut, data_cut = None, selection_cut
    else:
        mc_scale = 1.0
        lumi = LUMI[cfg["year"]]
        cut, data_cut = selection_cut, None

    cut = cut + " & weight < 10000" if cut else "weight < 10000"

    cms_label = cfg.get("cms_label", "Preliminary")

    if cfg.get("hists") and only != "mva":
        plotter = Plotter(
            sig=[BASE_PATH + p for p in cfg["signal_files"]],
            bkg=[BASE_PATH + p for p in cfg["background_files"]],
            data=[BASE_PATH + p for p in cfg["data_files"]],
            bkg_samples_labels=cfg["background_labels"],
            sig_samples_labels=cfg["signal_labels"],
            cut=cut,
            data_cut=data_cut,
            year=cfg["year"],
            lumi=lumi,
            mc_scale=mc_scale,
            cms_label=cms_label,
        )
        plotter.make_plots(
            build_hists(cfg["hists"], cfg.get("default_scale", 1.0)),
            save=True,
            density=cfg.get("density", False),
            save_path=output_dir,
            logy=cfg.get("logy", False),
        )

    mva = cfg.get("mva")
    if mva and only != "hists":
        mc_file = mva.get("mc_file")
        data_file = mva.get("data_file")
        base_path = mva.get("base_path")
        if base_path and not mc_file:
            mc_file, resolved_data = resolve_mva_files(base_path, mva.get("version"))
            if data_file is None:
                data_file = resolved_data
        mva_plotter = Plotter(
            mva_mc=mc_file,
            mva_data=data_file,
            mva_label_col=mva.get("label_col", "label"),
            mva_weight_col=mva.get("weight_col", "weight"),
            mva_data_cut=mva.get("data_cut"),
            bkg_samples_labels=mva.get("background_labels", cfg.get("background_labels")),
            year=cfg["year"],
            lumi=lumi,
            mc_scale=mc_scale,
            cms_label=cms_label,
        )
        mva_plotter.make_mva_plots(
            build_hists(mva["hists"], mva.get("default_scale", cfg.get("default_scale", 1.0))),
            save=True,
            density=cfg.get("density", False),
            save_path=output_dir,
            logy=mva.get("logy", cfg.get("logy", False)),
        )

    logger.info(f"✓ Config '{name}' completed")


def main():
    parser = ArgumentParser(description="Config-driven plotter for physics analysis")
    parser.add_argument("config", nargs="?", help="Config name from the JSON file, or 'all'")
    parser.add_argument("--config-file", default="configs.json",
                        help="Path to the JSON config file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: plots_<config>). Ignored for 'all'.")
    parser.add_argument("--only", choices=["hists", "mva"], default=None,
                        help="Run only the ROOT-file hists or only the MVA prediction plots")
    parser.add_argument("--list", action="store_true",
                        help="List available config names and exit")
    args = parser.parse_args()

    configs = json.loads(Path(args.config_file).read_text())

    if args.list or not args.config:
        print("Available configs:", ", ".join(configs))
        return 0

    if args.config == "all":
        names = list(configs)
    elif args.config in configs:
        names = [args.config]
    else:
        logger.error(f"Unknown config '{args.config}'. Available: {', '.join(configs)}")
        return 1

    failures = []
    for name in names:
        if len(names) > 1 or args.output is None:
            output_dir = f"plots_{name}"
        else:
            output_dir = args.output
        try:
            run_config(name, configs[name], output_dir, only=args.only)
        except Exception as e:
            logger.error(f"✗ Config '{name}' failed: {e}", exc_info=True)
            failures.append(name)

    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Plot generation interrupted by user")
        sys.exit(1)
