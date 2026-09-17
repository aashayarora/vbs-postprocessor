import glob
import logging
import os
import re
import sys
import csv
import yaml
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader import get_dataloader
from model import ABCDLightningModule

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MISSING_VALUE = -999.0

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(base, paths):
    if isinstance(paths, str):
        paths = [paths]
    resolved = []
    for p in paths:
        full = os.path.expanduser(os.path.join(base, p) if base else p)
        matches = sorted(glob.glob(full))
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(full)
    return resolved

def parse_training_features(training_features_cfg):
    training_features = []
    feature_transforms = {}

    for feat in training_features_cfg:
        if isinstance(feat, dict):
            for name, tf in feat.items():
                training_features.append(name)
                feature_transforms[name] = tf
        else:
            training_features.append(feat)
            feature_transforms[feat] = "none"

    return training_features, feature_transforms


def _extract_expression_variables(expression):
    tokens = re.findall(r"(?<!\.)\b[A-Za-z_][A-Za-z0-9_]*\b", expression)
    excluded = {"np", "abs", "and", "or", "not", "True", "False"}
    return [tok for tok in tokens if tok not in excluded]


def _normalize_derived_vars_cfg(derived_vars_cfg):
    if derived_vars_cfg is None:
        return {}
    if isinstance(derived_vars_cfg, dict):
        return dict(derived_vars_cfg)
    if isinstance(derived_vars_cfg, list):
        normalized = {}
        for item in derived_vars_cfg:
            if not isinstance(item, dict):
                raise ValueError("Each item in 'derived_vars' list must be a mapping of 'new_var: expression'.")
            for key, value in item.items():
                normalized[key] = value
        return normalized
    raise ValueError("'derived_vars' must be either a mapping or a list of single-item mappings.")


def _collect_derived_input_vars(derived_vars_cfg):
    derived_vars_cfg = _normalize_derived_vars_cfg(derived_vars_cfg)

    derived_names = set(derived_vars_cfg.keys())
    needed = []
    for expr in derived_vars_cfg.values():
        if not isinstance(expr, str):
            continue
        for var in _extract_expression_variables(expr):
            if var not in derived_names:
                needed.append(var)
    return list(dict.fromkeys(needed))


def apply_derived_vars(data, derived_vars_cfg):
    derived_vars_cfg = _normalize_derived_vars_cfg(derived_vars_cfg)
    if not derived_vars_cfg:
        return data

    out = {k: np.asarray(v).copy() for k, v in data.items()}
    n_events = _data_length(out)

    for new_var, expression in derived_vars_cfg.items():
        if not isinstance(expression, str):
            raise ValueError(f"derived_vars['{new_var}'] must be a string expression")

        local_dict = {k: np.asarray(v) for k, v in out.items()}
        try:
            values = np.asarray(eval(expression, {"__builtins__": {}, "np": np, "abs": np.abs}, local_dict))
        except NameError as exc:
            raise ValueError(
                f"Failed to evaluate derived var '{new_var}': missing variable in expression '{expression}'"
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"Failed to evaluate derived var '{new_var}' with expression '{expression}': {exc}"
            ) from exc

        if values.ndim == 0:
            values = np.full(n_events, values.item())

        if len(values) != n_events:
            raise ValueError(
                f"Derived var '{new_var}' has length {len(values)} but expected {n_events}"
            )

        out[new_var] = values
        logging.info("Computed derived var '%s' from config expression", new_var)

    return out

def _read_root_frame(path, branches, dataset_idx, sample_idx):
    with uproot.open(path) as root_file:
        arrays = root_file["Events"].arrays(branches, library="np")

    if not isinstance(arrays, dict):
        arrays = {name: arrays[name] for name in arrays.dtype.names}

    columns = {}
    n_events = None
    for branch in branches:
        if branch not in arrays:
            continue

        values = np.asarray(arrays[branch])

        if values.ndim == 1 and values.dtype != object:
            clean_values = values
        else:
            flat_values = [_flatten_awkward_cell(v) for v in values]
            clean_values = np.asarray(flat_values)

        if n_events is None:
            n_events = len(clean_values)
        else:
            n_events = min(n_events, len(clean_values))

        columns[branch] = clean_values

    if n_events is None or n_events == 0:
        return {
            "dataset_idx": np.array([], dtype=np.int32),
            "sample_idx": np.array([], dtype=np.int32),
        }

    trimmed = {name: np.asarray(vals)[:n_events] for name, vals in columns.items()}
    trimmed["dataset_idx"] = np.full(n_events, dataset_idx, dtype=np.int32)
    trimmed["sample_idx"] = np.full(n_events, sample_idx, dtype=np.int32)
    return trimmed


def _flatten_awkward_cell(value):
    if np.isscalar(value):
        return value
    return value[0] if len(value) > 0 else MISSING_VALUE


def _data_length(data):
    if not data:
        return 0
    first_key = next(iter(data))
    return len(data[first_key])


def _apply_mask(data, mask):
    return {k: np.asarray(v)[mask] for k, v in data.items()}


def _concat_chunks(chunks):
    if not chunks:
        return {}

    all_keys = set()
    for chunk in chunks:
        all_keys.update(chunk.keys())

    combined = {}
    for key in all_keys:
        arrays = [np.asarray(chunk[key]) for chunk in chunks if key in chunk]
        combined[key] = np.concatenate(arrays) if arrays else np.array([])
    return combined


def _evaluate_preselection_mask(data, expression):
    local_dict = {k: np.asarray(v) for k, v in data.items()}
    return np.asarray(eval(expression, {"__builtins__": {}, "np": np}, local_dict), dtype=bool)


# Also return fitted min/max so they can be saved for inference-time scaling
def _safe_minmax_scale(values, valid_mask):
    scaled = np.zeros_like(values, dtype=np.float64)
    scaler = MinMaxScaler()
    scaler.fit(values[valid_mask].reshape(-1, 1))
    scaled[valid_mask] = scaler.transform(values[valid_mask].reshape(-1, 1)).ravel()
    return scaled, float(scaler.data_min_[0]), float(scaler.data_max_[0])


def load_data(paths, features, extra_vars, num_workers=1):
    branches = list(dict.fromkeys(features + extra_vars))

    sample_names = []
    for path in paths:
        p = Path(path)
        parent = p.parent.name
        grandparent = p.parent.parent.name if p.parent.parent is not None else ""
        sample_names.append(grandparent if parent.isdigit() and grandparent else parent)

    sample_name_to_idx = {name: idx for idx, name in enumerate(sorted(set(sample_names)))}
    indexed_paths = [
        (dataset_idx, path, sample_name_to_idx[sample_name])
        for dataset_idx, (path, sample_name) in enumerate(zip(paths, sample_names))
    ]
    chunks = [None] * len(indexed_paths)

    if num_workers > 1 and len(indexed_paths) > 1:
        max_workers = min(num_workers, len(indexed_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_read_root_frame, path, branches, dataset_idx, sample_idx): dataset_idx
                for dataset_idx, path, sample_idx in indexed_paths
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading ROOT files"):
                dataset_idx = futures[future]
                chunks[dataset_idx] = future.result()
    else:
        for dataset_idx, path, sample_idx in tqdm(indexed_paths, total=len(indexed_paths), desc="Loading ROOT files"):
            chunks[dataset_idx] = _read_root_frame(path, branches, dataset_idx, sample_idx)

    chunks = [chunk for chunk in chunks if chunk is not None]

    data = _concat_chunks(chunks)
    logging.info("Loaded %d events from %d files.", _data_length(data), len(paths))
    if "weight" in data:
        data = _apply_mask(data, np.asarray(data["weight"]) > 0)

    return data


# If scaler_output_path is provided, fitted min/max per feature are saved to JSON for use at inference time
def preprocess_data(data, training_features, feature_transforms, constraint_var, scaler_output_path=None):
    out = {k: np.asarray(v).copy() for k, v in data.items()}
    cols_to_clean = list(dict.fromkeys(training_features + [constraint_var]))
    present_cols = [col for col in cols_to_clean if col in out]
    for col in present_cols:
        arr = np.asarray(out[col])
        if arr.dtype == object:
            arr = np.asarray([_flatten_awkward_cell(v) for v in arr], dtype=np.float64)
        else:
            arr = arr.astype(np.float64, copy=False)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        out[col] = np.nan_to_num(arr, nan=MISSING_VALUE, posinf=MISSING_VALUE, neginf=MISSING_VALUE)
    scaler_params = {}
    for feat in training_features:
        feat_arr = np.asarray(out[feat], dtype=np.float64)
        valid = (feat_arr != MISSING_VALUE) & np.isfinite(feat_arr)
        transform = feature_transforms.get(feat, "none")
        if transform == "log":
            positive = valid & (feat_arr > 0)
            feat_arr[valid & ~positive] = MISSING_VALUE
            valid = positive
        if transform == "log":
            feat_arr[valid] = np.log(feat_arr[valid])
        if valid.sum() == 0:
            print(f"WARNING: feature '{feat}' has no valid samples, skipping scaler fit")
            out[feat] = np.zeros_like(feat_arr, dtype=np.float64)
            scaler_params[feat] = {"transform": transform, "min": 0.0, "max": 1.0}
            continue
        out[feat], fmin, fmax = _safe_minmax_scale(feat_arr, valid)
        scaler_params[feat] = {"transform": transform, "min": fmin, "max": fmax}
    if constraint_var not in training_features:
        constraint_arr = np.asarray(out[constraint_var], dtype=np.float64)
        valid_constraint = constraint_arr != MISSING_VALUE
        out[constraint_var], fmin, fmax = _safe_minmax_scale(constraint_arr, valid_constraint)
        scaler_params[constraint_var] = {"transform": "none", "min": fmin, "max": fmax}
    if scaler_output_path is not None:
        import json
        scaler_out = {
            "_training_features": training_features,
            "_constraint_var": constraint_var,
        }
        scaler_out.update(scaler_params)
        with open(scaler_output_path, "w") as f:
            json.dump(scaler_out, f, indent=2)
        logging.info("Saved scaler params to %s", scaler_output_path)
    return out

def normalize_class_weights(sig_data, bkg_data):
    sig = {k: np.asarray(v).copy() for k, v in sig_data.items()}
    bkg = {k: np.asarray(v).copy() for k, v in bkg_data.items()}

    sig_weight_sum = np.sum(sig["weight"])
    bkg_weight_sum = np.sum(bkg["weight"])

    sig["weight"] = sig["weight"] / sig_weight_sum
    bkg["weight"] = bkg["weight"] / bkg_weight_sum
    return sig, bkg


def make_dataloaders(data, training_features, constraint_var, batch_size):
    labels_str = np.asarray(data["label"]).astype(np.int32).astype(str)
    stratify_col = "sample_idx" if "sample_idx" in data else "dataset_idx"
    sample_str = np.asarray(data[stratify_col]).astype(np.int64).astype(str)
    stratify_key = np.char.add(np.char.add(labels_str, "_"), sample_str)
    logging.info("Using stratification column '%s' for train/val split", stratify_col)

    unique_keys, unique_counts = np.unique(stratify_key, return_counts=True)
    rare_keys = unique_keys[unique_counts < 2]
    if len(rare_keys) > 0:
        rare_mask = np.isin(stratify_key, rare_keys)
        stratify_key[rare_mask] = np.char.add(labels_str[rare_mask], "_rare")
        logging.info(
            "Collapsed %d rare strata (<2 events) into label-level rare bins for stable splitting",
            len(rare_keys),
        )

    all_indices = np.arange(_data_length(data))

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=42,
        stratify=stratify_key,
    )

    feature_matrix = np.column_stack([data[f] for f in training_features]).astype(np.float32, copy=False)
    constraint_values = np.asarray(data[constraint_var], dtype=np.float32).reshape(-1, 1)
    labels = np.asarray(data["label"], dtype=np.float32)
    weights = np.asarray(data["weight"], dtype=np.float32)

    train_loader = get_dataloader(
        dnn_input_data=torch.from_numpy(feature_matrix[train_idx]),
        constraint_data=torch.from_numpy(constraint_values[train_idx]),
        labels=torch.from_numpy(labels[train_idx]),
        weights=torch.from_numpy(weights[train_idx]),
        batch_size=batch_size,
        use_sampler=False,
    )

    val_loader = get_dataloader(
        dnn_input_data=torch.from_numpy(feature_matrix[val_idx]),
        constraint_data=torch.from_numpy(constraint_values[val_idx]),
        labels=torch.from_numpy(labels[val_idx]),
        weights=torch.from_numpy(weights[val_idx]),
        batch_size=batch_size,
        use_sampler=False,
        is_validation=True,
    )

    # Also return indices so callers can tag events by their split for later analysis
    return train_loader, val_loader, train_idx, val_idx

def save_tensorboard_plots(log_dir, output_dir):
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps  = [e.step  for e in events]
        values = [e.value for e in events]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, values, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(tag)
        ax.set_title(tag)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_tag = tag.replace("/", "_")
        plt.savefig(output_dir / f"{safe_tag}.png", dpi=150)
        plt.close()
        print(f"Saved {output_dir / safe_tag}.png")

def run_training(
    model,
    train_loader,
    val_loader,
    output_dir,
    max_epochs,
    flavor,
    devices=[0],
    check_val_every_n_epoch=1,
    early_stopping_patience=20,
    early_stopping_min_delta=1e-4,
):
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=str(run_dir), name=flavor)

    checkpoint = ModelCheckpoint(
        dirpath=str(Path(logger.log_dir) / "checkpoints"),
        filename=f"{model.flavor}-abcdisco-{{epoch:03d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint, lr_monitor]
    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
        )

    use_gpu = torch.cuda.is_available()
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices if use_gpu else None,
        logger=logger,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    save_tensorboard_plots(logger.log_dir, logger.log_dir)
    return trainer


def _concat_sig_bkg(sig_data, bkg_data):
    column_order = list(sig_data.keys())
    for key in bkg_data.keys():
        if key not in sig_data:
            column_order.append(key)

    combined = {}
    for key in column_order:
        sig_col = np.asarray(sig_data.get(key, np.array([])))
        bkg_col = np.asarray(bkg_data.get(key, np.array([])))
        combined[key] = np.concatenate([sig_col, bkg_col])

    n_sig = _data_length(sig_data)
    n_bkg = _data_length(bkg_data)
    combined["label"] = np.concatenate(
        [
            np.ones(n_sig, dtype=np.int32),
            np.zeros(n_bkg, dtype=np.int32),
        ]
    )
    return combined


def _write_csv_numpy(data, output_csv):
    columns = list(data.keys())
    arrays = [np.asarray(data[col]) for col in columns]

    if not arrays:
        Path(output_csv).write_text("\n", encoding="utf-8")
        return

    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        raise ValueError(f"All output columns must have the same length, got lengths={sorted(lengths)}")

    all_numeric = all(np.issubdtype(arr.dtype, np.number) for arr in arrays)
    if all_numeric:
        matrix = np.column_stack(arrays)
        np.savetxt(
            output_csv,
            matrix,
            delimiter=",",
            header=",".join(columns),
            comments="",
            fmt="%.10g",
        )
        return

    def _format_cell(val):
        if isinstance(val, (np.floating, float)):
            return f"{float(val):.10g}"
        if isinstance(val, (np.integer, int)):
            return str(int(val))
        if isinstance(val, (np.bool_, bool)):
            return str(int(val))
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)

    output_path = Path(output_csv)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in zip(*arrays):
            writer.writerow([_format_cell(v) for v in row])


def _latest_version_dir(output_dir: Path, flavor) -> Path:
    logs_dir = output_dir / flavor
    version_dirs = [p for p in logs_dir.glob("version_*") if p.is_dir()]
    if not version_dirs:
        raise FileNotFoundError(f"No version_* directories found under {logs_dir}")

    def _version_key(path: Path):
        suffix = path.name.replace("version_", "")
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, path.stat().st_mtime)

    return sorted(version_dirs, key=_version_key)[-1]


def _latest_checkpoint_from_config(cfg, flavor) -> Path:
    output_dir = Path(cfg.get("output", "simple_abcdisco_output"))
    version_dir = _latest_version_dir(output_dir, flavor=flavor)
    ckpt_dir = version_dir / "checkpoints"
    ckpts = [p for p in ckpt_dir.glob("*.ckpt") if p.is_file()]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {ckpt_dir}")
    return sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]

def _all_checkpoints_from_config(cfg, flavor) -> list:
    output_dir = Path(cfg.get("output", "simple_abcdisco_output"))
    version_dir = _latest_version_dir(output_dir, flavor=flavor)
    ckpt_dir = version_dir / "checkpoints"
    ckpts = [p for p in ckpt_dir.glob("*.ckpt") if p.is_file()]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {ckpt_dir}")
    return sorted(ckpts, key=lambda p: p.stat().st_mtime)


def _inference_output_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _build_model_from_config(cfg, flavor, input_size, checkpoint_path, device):
    model = ABCDLightningModule(
        input_size=input_size,
        hidden_layers=cfg.get("architecture", [64, 32, 16]),
        learning_rate=cfg.get("learning_rate", 1e-3),
        bce_weight=cfg.get("bce_weight", 1.0),
        disco_lambda=cfg.get("disco_lambda", 0.0),
        flavor=flavor,
        use_batchnorm=cfg.get("use_batchnorm", True),
        dropout=cfg.get("dropout", 0.0),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.to(device)
    model.eval()
    return model


def _copy_input_plots_to_version_dir(source_dir, version_dir):
    source_dir = Path(source_dir)
    version_dir = Path(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)

    for plot_path in source_dir.glob("inputs_*.png"):
        dest = version_dir / plot_path.name
        shutil.copy(str(plot_path), str(dest))
        logging.info("Copied input plot to %s", dest)


def _batched_scores(model, features_tensor, batch_size, flavor, device):
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

def _plot_input_feature_distributions(
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
    weights = np.asarray(raw_data["weight"]) if "weight" in raw_data else np.ones(_data_length(raw_data), dtype=np.float64)

    train_mask = np.zeros(_data_length(raw_data), dtype=bool)
    val_mask = np.zeros(_data_length(raw_data), dtype=bool)
    train_mask[np.asarray(train_idx, dtype=np.int64)] = True
    val_mask[np.asarray(val_idx, dtype=np.int64)] = True

    training_feature_set = set(training_features)
    feature_transforms = feature_transforms or {}
    skip_cols = set(skip_cols or [])

    default_skip = {
        "label", "weight", "dataset_idx", "sample_idx", "split",
        "dnn_score", "dnn_0_score", "dnn_1_score",
    }
    skip_cols |= default_skip

    def _safe_filename(name):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

    def _weighted_density(values, weights, bins):
        counts, edges = np.histogram(values, bins=bins, weights=weights, density=False)
        total = np.sum(counts)
        if total > 0:
            counts = counts / total
        return counts, edges

    def _get_plot_values(arr):
        arr = np.asarray(arr)

        if arr.dtype == object:
            try:
                arr = np.asarray([_flatten_awkward_cell(v) for v in arr], dtype=np.float64)
            except Exception:
                return None
        elif np.issubdtype(arr.dtype, np.number):
            arr = arr.astype(np.float64, copy=False)
        else:
            return None

        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return arr

    all_features = []
    for key in raw_data.keys():
        if key in skip_cols:
            continue
        vals = _get_plot_values(raw_data[key])
        if vals is None:
            continue
        all_features.append(key)

    all_features = sorted(all_features)

    color_map = {
        ("bkg", "train"): "tab:blue",
        ("bkg", "val"): "tab:green",
        ("sig", "train"): "tab:red",
        ("sig", "val"): "tab:orange",
    }

    for feat in all_features:
        values = _get_plot_values(raw_data[feat])
        if values is None:
            logging.info("Skipping non-numeric or empty feature '%s' for input plotting", feat)
            continue

        finite_mask = np.isfinite(np.asarray(raw_data[feat], dtype=object) if np.asarray(raw_data[feat]).dtype == object else np.asarray(raw_data[feat]))
        # Rebuild cleaned full-length numeric array for masking
        full_arr = np.asarray(raw_data[feat])
        if full_arr.dtype == object:
            try:
                full_arr = np.asarray([_flatten_awkward_cell(v) for v in full_arr], dtype=np.float64)
            except Exception:
                logging.info("Skipping feature '%s' because it could not be flattened", feat)
                continue
        else:
            full_arr = full_arr.astype(np.float64, copy=False)

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

        if not np.isfinite(vmin) or not np.isfinite(vmax):
            logging.info("Skipping feature '%s' because plotting range is not finite", feat)
            continue

        if vmin == vmax:
            eps = 0.5 if vmin == 0 else 0.05 * abs(vmin)
            vmin -= eps
            vmax += eps

        bins = np.linspace(vmin, vmax, 51)

        fig, ax = plt.subplots(figsize=(8, 6))

        subsets = [
            ("bkg", "train", (plot_labels == 0) & plot_train_mask, "Background train"),
            ("bkg", "val",   (plot_labels == 0) & plot_val_mask,   "Background val"),
            ("sig", "train", (plot_labels == 1) & plot_train_mask, "Signal train"),
            ("sig", "val",   (plot_labels == 1) & plot_val_mask,   "Signal val"),
        ]

        drew_any = False
        for cls_name, split_name, mask, legend_label in subsets:
            vals = plot_vals[mask]
            wts = plot_weights[mask]
            if len(vals) == 0 or np.sum(wts) <= 0:
                continue

            counts, edges = _weighted_density(vals, wts, bins=bins)
            ax.step(
                edges[:-1],
                counts,
                where="post",
                linewidth=2,
                color=color_map[(cls_name, split_name)],
                label=f"{legend_label} (n={len(vals)})",
            )
            drew_any = True

        if not drew_any:
            plt.close(fig)
            logging.info("Skipping feature '%s' because no drawable subsets were found", feat)
            continue

        ax.set_title(feat)
        ax.set_xlabel(feat)
        ax.set_ylabel("Unit-normalized weighted yield")
        legend = ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if feat in training_feature_set:
            transform = feature_transforms.get(feat, "none")
            training_note = "★ used in training"
            if transform != "none":
                training_note += f" (transform: {transform})"

            fig.canvas.draw()
            if legend is not None:
                bbox_disp = legend.get_window_extent(fig.canvas.get_renderer())
                bbox_axes = bbox_disp.transformed(ax.transAxes.inverted())
                x_text = bbox_axes.x0
                y_text = max(0.02, bbox_axes.y0 - 0.06)
            else:
                x_text = 0.02
                y_text = 0.02

            ax.text(
                x_text,
                y_text,
                training_note,
                transform=ax.transAxes,
                fontsize=10,
                color="black",
                ha="left",
                va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="gold", alpha=0.25, edgecolor="goldenrod"),
            )

        output_path = output_dir / f"inputs_{_safe_filename(feat)}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logging.info("Saved input feature plot to %s", output_path)


def _plot_constraint_var_distribution(data, constraint_var, train_idx, val_idx, output_path):
    labels = np.asarray(data["label"])
    constraint = np.asarray(data[constraint_var])
    weights = np.asarray(data["weight"]) if "weight" in data else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    for col, (mask, title) in enumerate([
        (labels == 0, "Background"),
        (labels == 1, "Signal"),
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

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        ax_main.step(bins[:-1], all_counts_norm, where="post", linewidth=2,
                     color="black", label=f"All ({len(all_vals)})")
        ax_main.step(bins[:-1], train_counts_norm, where="post", linewidth=2,
                     color="tab:blue", label=f"Train ({len(train_vals)})")
        ax_main.step(bins[:-1], val_counts_norm, where="post", linewidth=2,
                     color="tab:orange", label=f"Val ({len(val_vals)})")
        ax_main.set_ylabel("Density")
        ax_main.set_title(f"{title} {constraint_var} distribution")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        ratio = np.where(train_counts_norm > 1e-12, val_counts_norm / train_counts_norm, np.nan)
        ax_ratio.step(bins[:-1], ratio, where="post", linewidth=2, color="black")
        ax_ratio.axhline(1.0, color="red", linewidth=1, linestyle="--")
        ax_ratio.set_xlabel(constraint_var)
        ax_ratio.set_ylabel("Val / Train")
        ax_ratio.set_title("Val / Train ratio", fontsize=10)
        ax_ratio.set_ylim(0, 2)
        ax_ratio.grid(True, alpha=0.3)

    fig.suptitle(f"{constraint_var} train/val split distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info("Saved constraint var distribution plot to %s", output_path)


def _plot_weight_distributions(sig_data, bkg_data, output_path):
    sig_weights = np.asarray(sig_data["weight"])
    bkg_weights = np.asarray(bkg_data["weight"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(sig_weights, bins=50, histtype="step", linewidth=2, color="tab:red")
    axes[0].set_xlabel("Weight")
    axes[0].set_ylabel("Counts")
    axes[0].set_title("Signal weight distribution")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(bkg_weights, bins=50, histtype="step", linewidth=2, color="tab:blue")
    axes[1].set_xlabel("Weight")
    axes[1].set_ylabel("Counts")
    axes[1].set_title("Background weight distribution")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Input weight distributions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info("Saved weight distribution plot to %s", output_path)


def _plot_abcd_plane(data, flavor, constraint_var, output_path, title_suffix=""):
    def profile_overlay(ax, x, y, bins, xrange, weights=None, color='yellow', label='Profile mean'):
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
        ax.errorbar(bin_centers, means, yerr=errors, color=color,
                    fmt='o', markersize=3, linewidth=1.5, label=label)
        ax.legend(fontsize=9)

    labels = np.asarray(data["label"])
    signal = {k: np.asarray(v)[labels == 1] for k, v in data.items()}
    background = {k: np.asarray(v)[labels == 0] for k, v in data.items()}

    score_col = "dnn_score" if flavor == "single" else "dnn_0_score"

    bins = [50, 50]
    dnn_range = (0, 1)
    constrain_var_range = (0, max(np.asarray(data[constraint_var])))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    h1 = axes[0].hist2d(background[score_col], background[constraint_var], bins=bins,
                         range=[dnn_range, constrain_var_range], cmap='Blues',
                         norm=matplotlib.colors.LogNorm())
    plt.colorbar(h1[3], ax=axes[0], label='Counts')
    axes[0].set_title('Background')
    axes[0].set_xlabel('DNN Score')
    axes[0].set_ylabel(constraint_var)
    profile_overlay(axes[0], background[score_col], background[constraint_var], bins[0], dnn_range)
    profile_overlay(axes[0], background[score_col], background[constraint_var], bins[0], dnn_range,
                    weights=background["weight"] if "weight" in background else None,
                    color='orange', label='Profile mean (weighted)')

    h2 = axes[1].hist2d(signal[score_col], signal[constraint_var], bins=bins,
                         range=[dnn_range, constrain_var_range], cmap='Reds',
                         norm=matplotlib.colors.LogNorm())
    plt.colorbar(h2[3], ax=axes[1], label='Counts')
    axes[1].set_title('Signal')
    axes[1].set_xlabel('DNN Score')
    axes[1].set_ylabel(constraint_var)
    profile_overlay(axes[1], signal[score_col], signal[constraint_var], bins[0], dnn_range)
    profile_overlay(axes[1], signal[score_col], signal[constraint_var], bins[0], dnn_range,
                    weights=signal["weight"] if "weight" in signal else None,
                    color='orange', label='Profile mean (weighted)')

    fig.suptitle(f'ABCD Plane{" - " + title_suffix if title_suffix else ""}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info("Saved ABCD plane plot to %s", output_path)


def _plot_decorrelation_check(data, flavor, constraint_var, output_path, title_suffix=""):
    score_col = "dnn_score" if flavor == "single" else "dnn_0_score"
    labels = np.asarray(data["label"])
    background = {k: np.asarray(v)[labels == 0] for k, v in data.items()}

    score_bins = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(len(score_bins) - 1):
        mask = (background[score_col] >= score_bins[i]) & (background[score_col] < score_bins[i + 1])
        ax.hist(background[constraint_var][mask], bins=50, density=True,
                histtype='step', label=f'DNN score [{score_bins[i]}, {score_bins[i + 1]}]')

    ax.set_xlabel(constraint_var)
    ax.set_ylabel('Normalised counts')
    ax.legend()
    ax.set_title(f'{constraint_var} in DNN score slices (background){" - " + title_suffix if title_suffix else ""}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info("Saved decorrelation check plot to %s", output_path)

def _plot_roc_curves(data, flavor, output_path):
    labels = np.asarray(data["label"])
    weights = np.asarray(data["weight"]) if "weight" in data else None

    plt.figure(figsize=(8, 7))

    roc_series = [
        ("DNN", "dnn_score")
    ] if flavor == "single" else [
        ("DNN 0", "dnn_0_score"),
        ("DNN 1", "dnn_1_score"),
    ]

    for label_name, score_col in roc_series:
        fpr, tpr, _ = roc_curve(labels, np.asarray(data[score_col]), sample_weight=weights)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{label_name} (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({flavor} flavor)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_score_densities(data, flavor, output_path):
    score_cols = ["dnn_score"] if flavor == "single" else ["dnn_0_score", "dnn_1_score"]
    fig, axes = plt.subplots(1, len(score_cols), figsize=(8 * len(score_cols), 6), squeeze=False)

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
        ax.hist(
            values[sig_mask],
            bins=50,
            range=(0, 1),
            weights=sig_w,
            density=True,
            histtype="step",
            linewidth=2,
            label="Signal",
            color="tab:red",
        )
        ax.hist(
            values[bkg_mask],
            bins=50,
            range=(0, 1),
            weights=bkg_w,
            density=True,
            histtype="step",
            linewidth=2,
            label="Background",
            color="tab:blue",
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_title(f"Score density: {col}")
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

# Compute permutation importance by shuffling each feature one at a time and measuring
# the resulting drop in AUC, averaged over n_repeats shuffles. Features with a larger
# AUC drop are more important to the model's discrimination.
def compute_permutation_importance(model, feature_matrix, labels, weights, training_features, device, batch_size=8192, n_repeats=3):
    """Compute permutation importance by shuffling each feature and measuring AUC drop."""
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

    baseline_tensor = torch.from_numpy(feature_matrix)
    baseline_auc = get_auc(baseline_tensor)

    importances = {}
    for i, feat in enumerate(training_features):
        drops = []
        for _ in range(n_repeats):
            shuffled = feature_matrix.copy()
            np.random.shuffle(shuffled[:, i])
            shuffled_tensor = torch.from_numpy(shuffled)
            shuffled_auc = get_auc(shuffled_tensor)
            drops.append(baseline_auc - shuffled_auc)
        importances[feat] = np.mean(drops)

    return baseline_auc, importances


# Plot permutation importance as a horizontal bar chart, sorted by importance.
# Features that hurt the AUC most when shuffled appear at the top.
def plot_permutation_importance(baseline_auc, importances, output_path):
    """Plot permutation importance as a horizontal bar chart sorted by importance."""
    sorted_feats = sorted(importances, key=importances.get)
    sorted_vals  = [importances[f] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_feats) * 0.25)))
    colors = ["tab:red" if v > 0 else "tab:blue" for v in sorted_vals]
    ax.barh(sorted_feats, sorted_vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean AUC drop when feature is shuffled")
    ax.set_title(f"Permutation Importance (baseline AUC={baseline_auc:.4f})")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def run_inference(args, cfg, flavor, sig_data, bkg_data, training_features, feature_transforms, constraint_var, derived_vars_cfg, is_data=False, inference_data=None, train_idx=None, val_idx=None):
    logging.info("Starting inference steps...")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint_from_config(cfg, flavor=flavor)
    logging.info("Using checkpoint: %s", checkpoint_path)

    if is_data:
        full_inference_data = inference_data
    else:
        full_inference_data = _concat_sig_bkg(sig_data, bkg_data)  # NOTE: must remain signal-first to match ordering assumed by train_idx/val_idx
        
    full_inference_data = apply_derived_vars(full_inference_data, derived_vars_cfg)

    logging.info("Preprocessing inference features...")
    processed = preprocess_data(
        full_inference_data,
        training_features=training_features,
        feature_transforms=feature_transforms,
        constraint_var=constraint_var,
    )

    feature_matrix = np.column_stack([processed[f] for f in training_features]).astype(np.float32, copy=False)
    features_tensor = torch.from_numpy(feature_matrix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model_from_config(
        cfg=cfg,
        flavor=flavor,
        input_size=len(training_features),
        checkpoint_path=checkpoint_path,
        device=device,
    )

    score_0, score_1 = _batched_scores(model, features_tensor, cfg.get("batch_size", 8192), flavor, device)

    if flavor == "single":
        full_inference_data["dnn_score"] = score_0
    else:
        full_inference_data["dnn_0_score"] = score_0
        full_inference_data["dnn_1_score"] = score_1

    # Tag each event with its split so plots can be made per-subset
    n_events = _data_length(full_inference_data)
    split_col = np.full(n_events, "unknown", dtype=object)
    if train_idx is not None and val_idx is not None:
        split_col[train_idx] = "train"
        split_col[val_idx] = "val"
    full_inference_data["split"] = split_col

    if is_data:
        default_out = _inference_output_dir_from_checkpoint(checkpoint_path) / f"predictions_{flavor}_{checkpoint_path.stem}_data.csv"
    else:
        default_out = _inference_output_dir_from_checkpoint(checkpoint_path) / f"predictions_{flavor}_{checkpoint_path.stem}.csv"

    output_csv = Path(args.output_csv) if args.output_csv else default_out
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Writing inference CSV to %s", output_csv)
    _write_csv_numpy(full_inference_data, output_csv)
    logging.info("Done. Wrote %d rows.", _data_length(full_inference_data))

    if not is_data:
        roc_path = output_csv.with_name(f"{output_csv.stem}_roc.png")
        density_path = output_csv.with_name(f"{output_csv.stem}_score_density.png")
        _plot_roc_curves(full_inference_data, flavor=flavor, output_path=roc_path)
        _plot_score_densities(full_inference_data, flavor=flavor, output_path=density_path)
        logging.info("Saved ROC plot to %s", roc_path)
        logging.info("Saved score density plot to %s", density_path)

        # Make ABCD plane and decorrelation plots (for all events, for train only, and for val only)
        plot_subsets = [("all", full_inference_data)]
        if train_idx is not None and val_idx is not None:
            train_mask = np.asarray(full_inference_data["split"]) == "train"
            val_mask   = np.asarray(full_inference_data["split"]) == "val"
            plot_subsets.append(("train", _apply_mask(full_inference_data, train_mask)))
            plot_subsets.append(("val",   _apply_mask(full_inference_data, val_mask)))

        for subset_name, subset_data in plot_subsets:
            abcd_path = output_csv.with_name(f"{output_csv.stem}_abcd_plane_{subset_name}.png")
            _plot_abcd_plane(subset_data, flavor=flavor, constraint_var=constraint_var,
                             output_path=abcd_path, title_suffix=subset_name)

            decorr_path = output_csv.with_name(f"{output_csv.stem}_decorrelation_check_{subset_name}.png")
            _plot_decorrelation_check(subset_data, flavor=flavor, constraint_var=constraint_var,
                                      output_path=decorr_path, title_suffix=subset_name)

        # Compute and save permutation importance to identify which features the model relies on most
        importance_path = output_csv.with_name(f"{output_csv.stem}_permutation_importance.png")
        baseline_auc, importances = compute_permutation_importance(
            model=model,
            feature_matrix=feature_matrix,
            labels=np.asarray(full_inference_data["label"]),
            weights=np.asarray(full_inference_data["weight"]),
            training_features=training_features,
            device=device,
            batch_size=cfg.get("batch_size", 8192),
        )
        plot_permutation_importance(baseline_auc, importances, importance_path)
        logging.info("Saved permutation importance plot to %s", importance_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--flavor", choices=["single", "double"], default=None, help="Training flavor: single (one output) or double (two outputs)")
    parser.add_argument("--data", action="store_true", help="Run inference data (without training) using the latest checkpoint from config")
    parser.add_argument("--infer", action="store_true", help="Skip training and run inference only")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.ckpt) for inference. If omitted, auto-picks newest checkpoint.")
    parser.add_argument("--output-csv", default=None, help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    flavor = args.flavor if args.flavor is not None else cfg.get("flavor", "single")
    logging.info("Using flavor=%s", flavor)

    training_features, feature_transforms = parse_training_features(cfg["training_features"])
    constraint_var = cfg["constraint_var"]
    derived_vars_cfg = _normalize_derived_vars_cfg(cfg.get("derived_vars", {}))
    if flavor == "double" and constraint_var not in training_features:
        training_features.append(constraint_var)
        feature_transforms[constraint_var] = cfg.get("constraint_as_feature_transform", "none")
        logging.info(
            "Double flavor: added constraint_var '%s' to training features as regular input",
            constraint_var,
        )

    if cfg.get("auto_include_derived_vars", False):
        auto_tf = cfg.get("auto_include_derived_vars_transform", "none")
        added_count = 0
        for derived_name in derived_vars_cfg.keys():
            if derived_name in training_features:
                continue
            training_features.append(derived_name)
            feature_transforms[derived_name] = auto_tf
            added_count += 1
        logging.info(
            "Auto-include derived vars enabled: added %d derived features with transform='%s'",
            added_count,
            auto_tf,
        )

    derived_var_names = set(derived_vars_cfg.keys())
    derived_input_vars = _collect_derived_input_vars(derived_vars_cfg)

    load_training_features = [f for f in training_features if f not in derived_var_names]
    load_constraint = [] if constraint_var in derived_var_names else [constraint_var]
    load_features = list(dict.fromkeys(load_training_features + load_constraint + derived_input_vars))
    extra_vars = cfg.get("extra_vars", [])
    if "weight" not in extra_vars:
        extra_vars.append("weight")

    sig_base = cfg.get("sig_base_path", cfg.get("input_base_path", ""))
    bkg_base = cfg.get("bkg_base_path", cfg.get("input_base_path", ""))

    sig_paths = resolve_paths(sig_base, cfg["sig_path"])
    bkg_paths = resolve_paths(bkg_base, cfg["bkg_path"])
    io_workers = int(cfg.get("io_workers", min(8, os.cpu_count() or 1)))
    logging.info("Using io_workers=%d for ROOT loading", io_workers)

    if args.data:
        if not args.infer:
            parser.error("--data can only be used with --infer")
        
        data_base = cfg.get("data_base_path", cfg.get("input_base_path", ""))
        if "data_path" not in cfg:
            parser.error("Config must contain 'data_path' when using --data")

        data_paths = resolve_paths(data_base, cfg["data_path"])
        logging.info("Loading real data...")
        real_data = load_data(data_paths, load_features, extra_vars, num_workers=io_workers)
        
        logging.info("Data samples: %d", _data_length(real_data))
        if cfg.get("preselection"):
            data_mask = _evaluate_preselection_mask(real_data, cfg["preselection"])
            real_data = _apply_mask(real_data, data_mask)
        logging.info(
            "After preselection - Data samples: %d",
            _data_length(real_data),
        )

        logging.info("Skipping training. Running inference on data only...")
        run_inference(args, cfg, flavor, None, None, training_features, feature_transforms, constraint_var, derived_vars_cfg, is_data=True, inference_data=real_data, train_idx=None, val_idx=None)
        return

    logging.info("Loading signal data...")
    sig_data = load_data(sig_paths, load_features, extra_vars, num_workers=io_workers)
    logging.info("Loading background data...")
    bkg_data = load_data(bkg_paths, load_features, extra_vars, num_workers=io_workers)


    logging.info("Signal samples: %d, Background samples: %d", _data_length(sig_data), _data_length(bkg_data))
    if cfg.get("preselection"):
        sig_mask = _evaluate_preselection_mask(sig_data, cfg["preselection"])
        bkg_mask = _evaluate_preselection_mask(bkg_data, cfg["preselection"])
        sig_data = _apply_mask(sig_data, sig_mask)
        bkg_data = _apply_mask(bkg_data, bkg_mask)
    logging.info(
        "After preselection - Signal samples: %d, Background samples: %d",
        _data_length(sig_data),
        _data_length(bkg_data),
    )

    sig_data["label"] = np.ones(_data_length(sig_data), dtype=np.float32)
    bkg_data["label"] = np.zeros(_data_length(bkg_data), dtype=np.float32)
    
    # Keep copies of the original data (with raw weights and features) for inference
    raw_sig_data = {k: np.copy(v) for k, v in sig_data.items()}
    raw_bkg_data = {k: np.copy(v) for k, v in bkg_data.items()}

    weight_plot_path = Path(cfg.get("output", "simple_abcdisco_output")) / "input_weight_distributions.png"
    weight_plot_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_weight_distributions(sig_data, bkg_data, weight_plot_path)

    sig_data, bkg_data = normalize_class_weights(sig_data, bkg_data)

    logging.info("Preprocessing data...")
    combined_keys = set(sig_data.keys()) | set(bkg_data.keys())
    data = {
        key: np.concatenate([sig_data.get(key, np.array([])), bkg_data.get(key, np.array([]))])
        for key in combined_keys
    }
    data = apply_derived_vars(data, derived_vars_cfg)
    # Save scaler params alongside the output so inference can apply identical scaling
    scaler_path = Path(cfg.get("output", "simple_abcdisco_output")) / "scaler_params.json"
    data = preprocess_data(data, training_features, feature_transforms, constraint_var, scaler_output_path=scaler_path)

    if args.infer:
        logging.info("Skipping training. Running inference only...")
        run_inference(args, cfg, flavor, raw_sig_data, raw_bkg_data, training_features, feature_transforms, constraint_var, derived_vars_cfg, train_idx=None, val_idx=None)
        return

    logging.info("Creating data loaders...")
    train_loader, val_loader, train_idx, val_idx = make_dataloaders(
        data=data,
        training_features=training_features,
        constraint_var=constraint_var,
        batch_size=cfg.get("batch_size", 4096),
    )

    constraint_plot_path = Path(cfg.get("output", "simple_abcdisco_output")) / "constraint_var_distribution.png"
    constraint_plot_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_constraint_var_distribution(data, constraint_var, train_idx, val_idx, constraint_plot_path)

    raw_plot_data = _concat_sig_bkg(raw_sig_data, raw_bkg_data)
    raw_plot_data = apply_derived_vars(raw_plot_data, derived_vars_cfg)
    _plot_input_feature_distributions(
        raw_data=raw_plot_data,
        training_features=training_features,
        train_idx=train_idx,
        val_idx=val_idx,
        output_dir=Path(cfg.get("output", "simple_abcdisco_output")),
        feature_transforms=feature_transforms,
    )

    lightning_model = ABCDLightningModule(
        input_size=len(training_features),
        hidden_layers=cfg.get("architecture", [64, 32, 16]),
        learning_rate=cfg.get("learning_rate", 1e-3),
        bce_weight=cfg.get("bce_weight", 1.0),
        disco_lambda=cfg.get("disco_lambda", 0.0),
        flavor=flavor,
        use_batchnorm=cfg.get("use_batchnorm", True),
        dropout=cfg.get("dropout", 0.0),
        weight_decay=cfg.get("weight_decay", 1e-2),
        label_smoothing=cfg.get("label_smoothing", 0.0),
        use_lr_scheduler=cfg.get("use_lr_scheduler", True),
        lr_scheduler_patience=cfg.get("lr_scheduler_patience", 40),
        lr_scheduler_factor=cfg.get("lr_scheduler_factor", 0.5),
        lr_scheduler_min_lr=cfg.get("lr_scheduler_min_lr", 1e-6),
    )

    output_dir = cfg.get("output", "simple_abcdisco_output")

    run_training(
        model=lightning_model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        max_epochs=cfg.get("n_epochs", 100),
        flavor=flavor,
        devices=cfg.get("devices", [0]),
        check_val_every_n_epoch=cfg.get("check_val_every_n_epoch", 1),
        early_stopping_patience=cfg.get("early_stopping_patience", 40),
        early_stopping_min_delta=cfg.get("early_stopping_min_delta", 1e-4),
    )

    logging.info("Training finished.")
    checkpoint_path = _latest_checkpoint_from_config(cfg, flavor=flavor)
    version_dir = _inference_output_dir_from_checkpoint(checkpoint_path)

    # Save a copy of the config used for this run to the versioned directory for reference
    shutil.copy(args.config, version_dir / Path(args.config).name)
    logging.info("Saved config to %s", version_dir / Path(args.config).name)

    # Copy scaler params to the versioned checkpoint directory so they stay with the model
    scaler_path_final = version_dir / "scaler_params.json"
    shutil.copy(str(scaler_path), str(scaler_path_final))
    logging.info("Copied scaler params to %s", scaler_path_final)

    # Copy input diagnostic plots to the versioned directory for reference
    shutil.copy(str(weight_plot_path), str(version_dir / "input_weight_distributions.png"))
    logging.info("Copied weight distribution plot to %s", version_dir / "input_weight_distributions.png")

    constraint_plot_path_src = Path(cfg.get("output", "simple_abcdisco_output")) / "constraint_var_distribution.png"
    shutil.copy(str(constraint_plot_path_src), str(version_dir / "constraint_var_distribution.png"))
    logging.info("Copied constraint var distribution plot to %s", version_dir / "constraint_var_distribution.png")

    _copy_input_plots_to_version_dir(
        source_dir=Path(cfg.get("output", "simple_abcdisco_output")),
        version_dir=version_dir,
    )

    logging.info("Running inference...")
    all_checkpoints = _all_checkpoints_from_config(cfg, flavor=flavor)
    for ckpt_path in all_checkpoints:
        logging.info("Running inference for checkpoint: %s", ckpt_path)
        args.checkpoint = str(ckpt_path)
        run_inference(args, cfg, flavor, raw_sig_data, raw_bkg_data, training_features, feature_transforms, constraint_var, derived_vars_cfg, train_idx=train_idx, val_idx=val_idx)
    logging.info("Inference finished.")

if __name__ == "__main__":
    main()
