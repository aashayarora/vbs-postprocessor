"""XGBoost BDT framework.

Trains a gradient-boosted decision tree on the VBS variables to separate signal
from background. The resulting BDT score is intended to be used as the
decorrelation ("constraint") variable for the DNN, replacing a single hand-picked
variable such as ``vbs_detajj`` with a learned combination of the VBS kinematics.

The trained model is persisted three ways:
  * ``bdt_model.json``    - native XGBoost model (used to reload for inference)
  * ``bdt_features.json`` - the ordered feature list (defines the column order)
  * ``bdt_tmva.root``     - ROOT ``TMVA::Experimental::RBDT`` weights
"""

import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

import xgboost as xgb

import style
from common import MISSING_VALUE, to_flat_float_column

BDT_SCORE_NAME = "bdt_score"


def parse_bdt_features(bdt_features_cfg):
    """Extract feature names from the config list.

    Accepts the same ``[name, {name: transform}, ...]`` format as
    ``training_features`` for consistency, but transforms are ignored: XGBoost is
    invariant to monotonic feature scaling and handles missing values natively.
    """
    if not bdt_features_cfg:
        return []

    features = []
    for feat in bdt_features_cfg:
        if isinstance(feat, dict):
            features.extend(feat.keys())
        else:
            features.append(feat)
    return list(dict.fromkeys(features))


def _clean_column(arr):
    arr = to_flat_float_column(arr)
    # Route sentinel/non-finite values to NaN so XGBoost's native missing-value
    # handling deals with them instead of treating -999 as a real number.
    return np.where(np.isfinite(arr) & (arr != MISSING_VALUE), arr, np.nan)


def build_bdt_matrix(data, features):
    columns = [_clean_column(data[f]) for f in features]
    return np.column_stack(columns).astype(np.float32, copy=False)


def bdt_paths(output_dir):
    bdt_dir = Path(output_dir) / "bdt"
    return {
        "dir": bdt_dir,
        "model": bdt_dir / "bdt_model.json",
        "features": bdt_dir / "bdt_features.json",
        "tmva": bdt_dir / "bdt_tmva.root",
        "roc": bdt_dir / "bdt_roc",
        "score": bdt_dir / "bdt_score_density",
    }


def _bdt_hyperparameters(cfg, scale_pos_weight):
    bdt_cfg = cfg.get("bdt", {}) or {}
    return dict(
        n_estimators=bdt_cfg.get("n_estimators", 300),
        max_depth=bdt_cfg.get("max_depth", 4),
        learning_rate=bdt_cfg.get("learning_rate", 0.1),
        subsample=bdt_cfg.get("subsample", 0.8),
        colsample_bytree=bdt_cfg.get("colsample_bytree", 0.8),
        min_child_weight=bdt_cfg.get("min_child_weight", 1.0),
        gamma=bdt_cfg.get("gamma", 0.0),
        reg_lambda=bdt_cfg.get("reg_lambda", 1.0),
        reg_alpha=bdt_cfg.get("reg_alpha", 0.0),
        tree_method=bdt_cfg.get("tree_method", "hist"),
        n_jobs=bdt_cfg.get("n_jobs", -1),
        # Balance signal against background. Computed from the weighted class sums
        # (Sum w_bkg / Sum w_sig) but overridable from the config if desired.
        scale_pos_weight=bdt_cfg.get("scale_pos_weight", scale_pos_weight),
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=bdt_cfg.get("early_stopping_rounds", 20),
        random_state=42,
    )


def _prepare_weights(sig_weights, bkg_weights):
    """Return per-event training weights and the signal/background balancing factor.

    The raw physics weights are kept (their relative magnitudes carry the event
    importance) but globally rescaled so the mean weight is ~1. Without this the
    tiny per-event weights would drop the summed Hessian in each node below the
    default ``min_child_weight`` and prevent XGBoost from making any splits.

    Class balancing is left to ``scale_pos_weight = Sum w_bkg / Sum w_sig`` so that
    signal and background contribute equally regardless of integrated luminosity.
    """
    sig_w = np.clip(np.asarray(sig_weights, dtype=np.float64), 0.0, None)
    bkg_w = np.clip(np.asarray(bkg_weights, dtype=np.float64), 0.0, None)

    weights = np.concatenate([sig_w, bkg_w])
    total = weights.sum()
    if total > 0:
        weights = weights * (len(weights) / total)

    sig_sum = sig_w.sum()
    bkg_sum = bkg_w.sum()
    scale_pos_weight = (bkg_sum / sig_sum) if sig_sum > 0 else 1.0

    return weights.astype(np.float32), float(scale_pos_weight)


def train_bdt(sig_data, bkg_data, features, cfg, output_dir):
    """Train the XGBoost BDT on sig vs. bkg VBS features.

    Produces ROC and score-distribution plots on a held-out validation split and
    persists the model (native, feature list, and TMVA ROOT weights). Returns the
    fitted ``XGBClassifier``.
    """
    if not features:
        raise ValueError("train_bdt called with no BDT features")

    logging.info("Training BDT on %d VBS features: %s", len(features), ", ".join(features))

    x_sig = build_bdt_matrix(sig_data, features)
    x_bkg = build_bdt_matrix(bkg_data, features)

    w, scale_pos_weight = _prepare_weights(sig_data.get("weight"), bkg_data.get("weight"))
    logging.info("BDT class balancing: scale_pos_weight=%.4g", scale_pos_weight)

    x = np.concatenate([x_sig, x_bkg], axis=0)
    y = np.concatenate([np.ones(len(x_sig), dtype=np.int32), np.zeros(len(x_bkg), dtype=np.int32)])

    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
        x, y, w, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(**_bdt_hyperparameters(cfg, scale_pos_weight))
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(x_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )
    logging.info("BDT training finished (best_iteration=%s)", getattr(model, "best_iteration", "n/a"))

    # With early stopping the booster keeps all trained trees, but predict_proba
    # only uses up to best_iteration. Trim the booster so the exported model (both
    # the native file and the TMVA RBDT, which dump every tree) matches predict.
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        model._Booster = model.get_booster()[: best_iteration + 1]
        logging.info("Trimmed BDT to %d trees (best_iteration + 1) for a consistent export", best_iteration + 1)

    val_scores = predict_bdt_from_matrix(model, x_val)

    paths = bdt_paths(output_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    model.save_model(str(paths["model"]))
    paths["features"].write_text(json.dumps(list(features), indent=2), encoding="utf-8")
    save_bdt_tmva(model, features, paths["tmva"])
    logging.info("Saved BDT model to %s and TMVA weights to %s", paths["model"], paths["tmva"])

    _plot_bdt_roc(y_val, val_scores, w_val, paths["roc"])
    _plot_bdt_score_density(y_val, val_scores, w_val, paths["score"])
    logging.info("Saved BDT ROC plot to %s and score density to %s", paths["roc"], paths["score"])

    return model


def load_bdt(output_dir):
    paths = bdt_paths(output_dir)
    if not paths["model"].exists():
        raise FileNotFoundError(
            f"No trained BDT found at {paths['model']}. Run training before inference."
        )
    model = xgb.XGBClassifier()
    model.load_model(str(paths["model"]))
    return model


def load_bdt_features(output_dir):
    paths = bdt_paths(output_dir)
    if not paths["features"].exists():
        return None
    return json.loads(paths["features"].read_text(encoding="utf-8"))


def predict_bdt_from_matrix(model, matrix):
    return model.predict_proba(matrix)[:, 1].astype(np.float32)


def predict_bdt(model, data, features):
    matrix = build_bdt_matrix(data, features)
    return predict_bdt_from_matrix(model, matrix)


def save_bdt_tmva(model, features, output_path):
    """Save the BDT as a ROOT ``TMVA::Experimental::RBDT`` object.

    ROOT is imported lazily so that training/inference paths that do not touch it
    do not pay the import cost (and so a missing ROOT install only breaks here).
    """
    from ROOT.TMVA.Experimental import SaveXGBoost

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    SaveXGBoost(model, "bdt", str(output_path), len(features))


def _plot_bdt_roc(labels, scores, weights, output_path):
    labels = np.asarray(labels)
    weights = np.asarray(weights) if weights is not None else None

    fpr, tpr, _ = roc_curve(labels, np.asarray(scores), sample_weight=weights)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(fpr, tpr, linewidth=2.5, color=style.PETROFF_6[0],
            label=f"BDT (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.5, color=style.NEUTRAL_COLOR,
            label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    style.annotate(ax, ["Validation set", style.CONTEXT.extra])
    style.legend(ax, loc="lower right")
    style.cms_header(ax, data=False)

    style.save(fig, output_path)


def _plot_bdt_score_density(labels, scores, weights, output_path):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    sig_mask = labels == 1
    bkg_mask = labels == 0

    if weights is not None:
        weights = np.asarray(weights)
        sig_w = weights[sig_mask]
        bkg_w = weights[bkg_mask]
    else:
        sig_w = bkg_w = None

    fig, ax = plt.subplots(figsize=style.FIG_SINGLE)
    for values, w, label, color, linestyle in [
        (scores[sig_mask], sig_w, "Signal", style.SIGNAL_COLOR, "-"),
        (scores[bkg_mask], bkg_w, "Background", style.BACKGROUND_COLOR, "--"),
    ]:
        ax.hist(values, bins=50, range=(0, 1), weights=w, density=True,
                histtype="step", linewidth=2.5, label=label, color=color,
                linestyle=linestyle)
    ax.set_xlabel("BDT score")
    ax.set_ylabel("Normalised events")
    ax.set_xlim(0, 1)
    style.headroom(ax, factor=1.5)
    style.annotate(ax, ["Validation set", style.CONTEXT.extra])
    style.legend(ax, loc="upper center")
    style.cms_header(ax, data=False)

    style.save(fig, output_path)
