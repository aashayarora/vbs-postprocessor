"""Scoring events with a trained checkpoint and writing the prediction files."""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

import bdt as bdt_lib
import plots
from checkpoints import load_scaler_params, run_dir_for_checkpoint
from common import apply_mask, data_length
from model import ABCDLightningModule
from predictions import SUFFIX, read_predictions, write_predictions
from preprocessing import apply_derived_vars, preprocess_data


def build_model(run_cfg, checkpoint_path, device):
    model = ABCDLightningModule(
        **run_cfg.model_kwargs(input_size=len(run_cfg.training_features), for_inference=True)
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


def batched_scores(model, features_tensor, batch_size, flavor, device):
    scores_0 = []
    scores_1 = []

    with torch.no_grad():
        for start in tqdm(range(0, len(features_tensor), batch_size), desc="Inferring events"):
            batch = features_tensor[start:start + batch_size].to(device)
            logits = model(batch)
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores_0.append(probs[:, 0])
            if flavor == "double":
                scores_1.append(probs[:, 1])

    out_0 = np.concatenate(scores_0) if scores_0 else np.array([], dtype=np.float32)
    out_1 = np.concatenate(scores_1) if scores_1 else np.array([], dtype=np.float32)
    return out_0, out_1


# Compute permutation importance by shuffling each feature one at a time and measuring
# the resulting drop in AUC, averaged over n_repeats shuffles. Features with a larger
# AUC drop are more important to the model's discrimination.
def compute_permutation_importance(model, feature_matrix, labels, weights, training_features, device, batch_size=8192, n_repeats=3):
    def get_auc(features_tensor):
        all_scores = []
        with torch.no_grad():
            for start in range(0, len(features_tensor), batch_size):
                batch = features_tensor[start:start + batch_size].to(device)
                logits = model(batch)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(-1)
                scores = torch.sigmoid(logits).cpu().numpy()[:, 0]
                all_scores.append(scores)
        scores = np.concatenate(all_scores)
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
        return auc(fpr, tpr)

    baseline_auc = get_auc(torch.from_numpy(feature_matrix))

    importances = {}
    for i, feat in enumerate(training_features):
        drops = []
        for _ in range(n_repeats):
            shuffled = feature_matrix.copy()
            np.random.shuffle(shuffled[:, i])
            shuffled_auc = get_auc(torch.from_numpy(shuffled))
            drops.append(baseline_auc - shuffled_auc)
        importances[feat] = np.mean(drops)

    return baseline_auc, importances


def prepare_inference_data(run_cfg, inference_data):
    """Attach derived vars and the BDT constraint score ahead of scoring.

    Returns a new dict; the input is not modified.
    """
    data = dict(inference_data)
    data = apply_derived_vars(data, run_cfg.derived_vars)
    if run_cfg.use_bdt:
        logging.info("Loading trained BDT to compute '%s' constraint...", bdt_lib.BDT_SCORE_NAME)
        bdt_model = bdt_lib.load_bdt(run_cfg.output_dir)
        data[bdt_lib.BDT_SCORE_NAME] = bdt_lib.predict_bdt(bdt_model, data, run_cfg.bdt_features)
    return data


def default_predictions_path(run_cfg, checkpoint_path, is_data=False):
    """Where run_inference writes (and --plots-only reads) a prediction file."""
    checkpoint_path = Path(checkpoint_path)
    tag = "_data" if is_data else ""
    return (run_dir_for_checkpoint(checkpoint_path)
            / f"predictions_{run_cfg.flavor}_{checkpoint_path.stem}{tag}{SUFFIX}")


def make_inference_plots(run_cfg, data, output_path, is_data=False,
                         model=None, feature_matrix=None, device=None):
    """Draw every diagnostic for one prediction set.

    ``model``/``feature_matrix``/``device`` are only needed for the permutation
    importance, which is skipped without them (the --plots-only path, where the
    scores are read back from disk and the model is never built).
    """
    flavor = run_cfg.flavor
    constraint_var = run_cfg.constraint_var
    output_path = Path(output_path)

    # Plot paths carry no extension: style.save writes each figure in every format
    # in style.SAVE_FORMATS (PDF for the note, PNG for browsing) and logs what it wrote.
    if is_data:
        plots.plot_abcd_plane_data(data, flavor=flavor, constraint_var=constraint_var,
                                   output_path=output_path.with_name(f"{output_path.stem}_abcd_plane_data"))
        return

    plots.plot_roc_curves(data, flavor=flavor,
                          output_path=output_path.with_name(f"{output_path.stem}_roc"))
    plots.plot_score_densities(data, flavor=flavor,
                               output_path=output_path.with_name(f"{output_path.stem}_score_density"))

    # Make ABCD plane and decorrelation plots (for all events, for train only, and for val only)
    plot_subsets = [("all", data)]
    if "split" in data:
        split = np.asarray(data["split"])
        for subset_name in ("train", "val"):
            mask = split == subset_name
            if mask.any():
                plot_subsets.append((subset_name, apply_mask(data, mask)))

    for subset_name, subset_data in plot_subsets:
        abcd_path = output_path.with_name(f"{output_path.stem}_abcd_plane_{subset_name}")
        plots.plot_abcd_plane(subset_data, flavor=flavor, constraint_var=constraint_var,
                              output_path=abcd_path, title_suffix=subset_name)

        decorr_path = output_path.with_name(f"{output_path.stem}_decorrelation_check_{subset_name}")
        plots.plot_decorrelation_check(subset_data, flavor=flavor, constraint_var=constraint_var,
                                       output_path=decorr_path, title_suffix=subset_name)

    # Compute and save permutation importance to identify which features the model relies on most
    if model is None or feature_matrix is None:
        logging.info("Skipping permutation importance (needs the model; re-run without --plots-only).")
        return

    importance_path = output_path.with_name(f"{output_path.stem}_permutation_importance")
    baseline_auc, importances = compute_permutation_importance(
        model=model,
        feature_matrix=feature_matrix,
        labels=np.asarray(data["label"]),
        weights=np.asarray(data["weight"]),
        training_features=run_cfg.training_features,
        device=device,
        batch_size=run_cfg.batch_size,
    )
    plots.plot_permutation_importance(baseline_auc, importances, importance_path)


def plots_from_predictions(run_cfg, predictions_path, is_data=False):
    """Redraw the diagnostics from a prediction file written by an earlier run."""
    predictions_path = Path(predictions_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"No prediction file at {predictions_path}; run the inference first "
            "(or point --output-path at an existing file)."
        )

    logging.info("Reading predictions from %s", predictions_path)
    df = read_predictions(predictions_path)
    data = {col: df[col].to_numpy() for col in df.columns}
    logging.info("Read %d rows.", data_length(data))

    make_inference_plots(run_cfg, data, predictions_path, is_data=is_data)


def run_inference(run_cfg, checkpoint_path, inference_data, is_data=False,
                  train_idx=None, val_idx=None, output_path=None, make_plots=True):
    """Score ``inference_data`` (already passed through prepare_inference_data)
    with one checkpoint, write the prediction file, and draw the diagnostics.

    ``make_plots=False`` writes the predictions and stops. The diagnostics are
    dominated by the permutation importance, which costs several minutes and is
    the same for every run that shares a checkpoint -- not worth repeating when
    scoring a series of signal samples through one trained model.
    """
    checkpoint_path = Path(checkpoint_path)
    logging.info("Using checkpoint: %s", checkpoint_path)
    data = dict(inference_data)

    logging.info("Preprocessing inference features...")
    scaler_params = load_scaler_params(run_cfg.output_dir, checkpoint_path)
    processed = preprocess_data(
        data,
        training_features=run_cfg.training_features,
        feature_transforms=run_cfg.feature_transforms,
        constraint_var=run_cfg.constraint_var,
        scaler_params=scaler_params,
    )

    feature_matrix = np.column_stack(
        [processed[f] for f in run_cfg.training_features]
    ).astype(np.float32, copy=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(run_cfg, checkpoint_path, device)

    flavor = run_cfg.flavor
    score_0, score_1 = batched_scores(
        model, torch.from_numpy(feature_matrix), run_cfg.batch_size, flavor, device
    )

    if flavor == "single":
        data["dnn_score"] = score_0
    else:
        data["dnn_0_score"] = score_0
        data["dnn_1_score"] = score_1

    # Tag each event with its split so plots can be made per-subset
    split_col = np.full(data_length(data), "unknown", dtype=object)
    if train_idx is not None and val_idx is not None:
        split_col[train_idx] = "train"
        split_col[val_idx] = "val"
    data["split"] = split_col

    if output_path is None:
        output_path = default_predictions_path(run_cfg, checkpoint_path, is_data=is_data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Writing predictions to %s", output_path)
    write_predictions(data, output_path)
    logging.info("Done. Wrote %d rows.", data_length(data))

    if make_plots:
        make_inference_plots(run_cfg, data, output_path, is_data=is_data,
                             model=model, feature_matrix=feature_matrix, device=device)
