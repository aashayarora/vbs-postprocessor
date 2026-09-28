#!/usr/bin/env python3
"""ABCDisCo training and inference on the candidate post-processor parquet.

Entry point for the modular pipeline (config, root_io, bdt, preprocessing, training,
inference). When the config lists ``bdt_features`` (and ``use_bdt_as_constraint`` is not
false), an XGBoost BDT is first trained on those VBS variables and the DNN is
decorrelated against its score instead of the config's ``constraint_var``. The BDT is
saved under ``<output>/bdt`` and reapplied from there at inference.

    python run_abcd.py --config single/config_1lep_boosted_run3.yaml             # train, then score sig + bkg
    python run_abcd.py --config ... --infer [--checkpoint X]                     # score sig (infer paths) + bkg
    python run_abcd.py --config ... --infer --data [--checkpoint X]              # score data
    python run_abcd.py --config ... --rescore                                    # redo the post-training scoring
    python run_abcd.py --config ... --infer --signal 'merged_*/VBS*_c2v1p0_c3_1p0_*/*.parquet' \
        --checkpoint X --output sm.parquet --no-plots                           # one signal point only

Without --checkpoint, inference uses the newest checkpoint of the newest version.
"""

import argparse
import logging
import shutil
from pathlib import Path

import bdt as bdt_lib
import plots
import style
from checkpoints import all_checkpoints, best_and_last, latest_checkpoint, latest_version_dir
from common import concat_sig_bkg, data_length
from config import RunConfig, load_yaml, resolve_paths
from inference import prepare_inference_data, run_inference
from model import ABCDLightningModule
from preprocessing import apply_derived_vars, normalize_class_weights, preprocess_data
from root_io import apply_preselection, load_data
from training import collect_run_artifacts, make_dataloaders, run_training, train_val_indices


def load_kind(run_cfg, kind, for_inference=False, paths=None):
    """Read one sample kind ('sig', 'bkg', 'data') and apply the preselection; ``paths``
    overrides the config's files for that kind."""
    if paths is None:
        paths = run_cfg.sample_paths(kind, for_inference=for_inference)
    logging.info("Loading %s: %d file(s)", kind, len(paths))
    data = load_data(paths, run_cfg.load_features, run_cfg.extra_vars_for(kind),
                     num_workers=run_cfg.io_workers)
    return apply_preselection(data, run_cfg.preselection, kind)


def load_training_samples(run_cfg, bdt_model=None):
    """Signal and background as trained on: derived vars and, with a BDT, its score. The
    BDT is fit here unless an already trained ``bdt_model`` is given."""
    sig = apply_derived_vars(load_kind(run_cfg, "sig"), run_cfg.derived_vars)
    bkg = apply_derived_vars(load_kind(run_cfg, "bkg"), run_cfg.derived_vars)
    logging.info("Signal: %d events, background: %d events", data_length(sig), data_length(bkg))

    if run_cfg.use_bdt:
        if bdt_model is None:
            bdt_model = bdt_lib.train_bdt(sig, bkg, run_cfg.bdt_features, run_cfg.raw, run_cfg.output_dir)
        for d in (sig, bkg):
            d[bdt_lib.BDT_SCORE_NAME] = bdt_lib.predict_bdt(bdt_model, d, run_cfg.bdt_features)
    return sig, bkg


def normalized_training_data(sig, bkg):
    """Signal then background, each class's weights summing to 1, as the DNN trains on."""
    return concat_sig_bkg(*normalize_class_weights(sig, bkg))


def split_for_training(run_cfg, data, scaler_output_path=None):
    """Preprocess ``normalized_training_data`` and split it exactly as the training does;
    returns the loaders, the train/val indices and the preprocessed data."""
    data = preprocess_data(data, run_cfg.training_features, run_cfg.feature_transforms,
                           run_cfg.constraint_var, scaler_output_path=scaler_output_path)
    train_loader, val_loader, train_idx, val_idx = make_dataloaders(
        data, run_cfg.training_features, run_cfg.constraint_var, run_cfg.batch_size)
    return train_loader, val_loader, train_idx, val_idx, data


def rescore(run_cfg):
    """Redo train()'s final scoring of the newest version (best and last checkpoints) without
    retraining, e.g. after a prediction write failed: same events, same train/val split, and
    the BDT saved with that version."""
    version_dir = latest_version_dir(run_cfg.output_dir, run_cfg.flavor)
    bdt_model = bdt_lib.load_bdt(version_dir) if run_cfg.use_bdt else None
    sig, bkg = load_training_samples(run_cfg, bdt_model=bdt_model)
    raw = concat_sig_bkg(sig, bkg)
    del sig, bkg
    train_idx, val_idx = train_val_indices(raw)

    checkpoints = all_checkpoints(run_cfg.output_dir, run_cfg.flavor)
    best = min(checkpoints, key=lambda p: float(p.stem.rsplit("val_loss=", 1)[1]))
    for checkpoint in dict.fromkeys([best, checkpoints[-1]]):
        logging.info("Rescoring %s", checkpoint)
        run_inference(run_cfg, checkpoint, raw, train_idx=train_idx, val_idx=val_idx)
    logging.info("Inference finished.")


def train(run_cfg, config_path):
    out = run_cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sig, bkg = load_training_samples(run_cfg)

    # Raw weights and features (plus derived vars and the BDT score), scored after training.
    raw = concat_sig_bkg(sig, bkg)
    plots.plot_weight_distributions(sig, bkg, out / "input_weight_distributions")

    data = normalized_training_data(sig, bkg)
    del sig, bkg
    train_loader, val_loader, train_idx, val_idx, data = split_for_training(
        run_cfg, data, scaler_output_path=out / "scaler_params.json")
    plots.plot_constraint_var_distribution(data, run_cfg.constraint_var, train_idx, val_idx,
                                           out / "constraint_var_distribution")
    del data
    plots.plot_input_feature_distributions(raw, run_cfg.training_features, train_idx, val_idx, out,
                                           feature_transforms=run_cfg.feature_transforms)

    cfg = run_cfg.raw
    model = ABCDLightningModule(**run_cfg.model_kwargs(input_size=len(run_cfg.training_features)))
    trainer = run_training(
        model, train_loader, val_loader, out,
        max_epochs=cfg.get("n_epochs", 100),
        flavor=run_cfg.flavor,
        devices=cfg.get("devices", [0]),
        check_val_every_n_epoch=cfg.get("check_val_every_n_epoch", 1),
        early_stopping_patience=cfg.get("early_stopping_patience", 40),
        early_stopping_min_delta=cfg.get("early_stopping_min_delta", 1e-4),
    )
    logging.info("Training finished.")

    version_dir = latest_version_dir(out, run_cfg.flavor)
    collect_run_artifacts(version_dir, config_path, out)
    if run_cfg.use_bdt:
        # <output>/bdt is overwritten by the next training; keep this run's BDT with its model.
        shutil.copytree(bdt_lib.bdt_paths(out)["dir"], version_dir / "bdt", dirs_exist_ok=True)

    for label, checkpoint in best_and_last(trainer, out, run_cfg.flavor):
        logging.info("Scoring the %s checkpoint %s", label, checkpoint)
        run_inference(run_cfg, checkpoint, raw, train_idx=train_idx, val_idx=val_idx)
    logging.info("Inference finished.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to the YAML config")
    p.add_argument("--flavor", choices=["single", "double"], default=None,
                   help="single (one output) or double (two outputs); default: the config's, else single")
    p.add_argument("--infer", action="store_true", help="skip training and only score")
    p.add_argument("--data", action="store_true", help="with --infer: score the data samples")
    p.add_argument("--rescore", action="store_true",
                   help="redo the post-training scoring of the newest version (best and last "
                        "checkpoints, same train/val split) without retraining")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint (.ckpt) to score with (default: the newest)")
    p.add_argument("--signal", nargs="+", default=None, metavar="GLOB",
                   help="with --infer: score only these signal files (globs relative to the "
                        "config's signal base path, e.g. one coupling point), no background")
    p.add_argument("--output", default=None,
                   help="with --infer: predictions file to write (default: next to the checkpoint)")
    p.add_argument("--no-plots", action="store_true",
                   help="with --infer: write the predictions only, skip the diagnostic plots")
    args = p.parse_args()
    if args.data and not args.infer:
        p.error("--data can only be used with --infer")
    if args.rescore and args.infer:
        p.error("--rescore and --infer are separate modes")
    if args.signal and (not args.infer or args.data):
        p.error("--signal is a mode of --infer, without --data")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_cfg = load_yaml(args.config)
    run_cfg = RunConfig(raw=raw_cfg, flavor=args.flavor or raw_cfg.get("flavor", "single"))
    style.configure(**run_cfg.plot_style)
    logging.info("flavor=%s  constraint=%s  %d training features  BDT: %s", run_cfg.flavor,
                 run_cfg.constraint_var, len(run_cfg.training_features),
                 ", ".join(run_cfg.bdt_features) if run_cfg.use_bdt else "off")

    if args.rescore:
        rescore(run_cfg)
        return
    if not args.infer:
        train(run_cfg, args.config)
        return

    checkpoint = (Path(args.checkpoint) if args.checkpoint
                  else latest_checkpoint(run_cfg.output_dir, run_cfg.flavor))
    if args.data:
        data = load_kind(run_cfg, "data", for_inference=True)
    elif args.signal:
        base = run_cfg.raw.get("sig_base_path", run_cfg.raw.get("input_base_path", ""))
        paths = resolve_paths(base, args.signal)
        missing = [f for f in paths if not Path(f).exists()]
        if missing:
            p.error(f"--signal matched no files for: {missing}")
        data = concat_sig_bkg(load_kind(run_cfg, "sig", paths=paths), {})
    else:
        data = concat_sig_bkg(load_kind(run_cfg, "sig", for_inference=True),
                              load_kind(run_cfg, "bkg", for_inference=True))
    run_inference(run_cfg, checkpoint, prepare_inference_data(run_cfg, data), is_data=args.data,
                  output_path=args.output, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
