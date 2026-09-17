"""Train/val dataloaders, the Lightning training run, and post-run artifact copying."""

import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

import style
from common import data_length
from dataloader import get_dataloader
from plots import save_tensorboard_plots


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

    all_indices = np.arange(data_length(data))

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


def collect_run_artifacts(version_dir, config_path, output_dir):
    """Copy the run config, scaler params, and input diagnostic plots into the
    versioned checkpoint directory so they stay with the model."""
    version_dir = Path(version_dir)
    output_dir = Path(output_dir)
    version_dir.mkdir(parents=True, exist_ok=True)

    # Plots are written in every format in style.SAVE_FORMATS (PDF for the note,
    # PNG for browsing), so each one is collected under all of its extensions.
    artifacts = [Path(config_path), output_dir / "scaler_params.json"]
    for stem in ("input_weight_distributions", "constraint_var_distribution"):
        artifacts += [output_dir / f"{stem}.{ext}" for ext in style.SAVE_FORMATS]

    for src in artifacts:
        if not src.is_file():
            logging.warning("Expected run artifact %s not found, skipping copy", src)
            continue
        shutil.copy(str(src), str(version_dir / src.name))
        logging.info("Copied %s to %s", src.name, version_dir / src.name)

    # Input feature plots are per-run, so move rather than copy them.
    for plot_path in sorted(p for ext in style.SAVE_FORMATS for p in output_dir.glob(f"inputs_*.{ext}")):
        try:
            dest = version_dir / plot_path.name
            shutil.move(str(plot_path), str(dest))
            logging.info("Moved input plot to %s", dest)
        except Exception as e:
            logging.warning("Failed to move input plot %s: %s", plot_path, e)
