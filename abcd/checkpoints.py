"""Locating trained model artifacts (Lightning version dirs, checkpoints, scaler params)."""

import json
import logging
from pathlib import Path


def latest_version_dir(output_dir, flavor) -> Path:
    logs_dir = Path(output_dir) / flavor
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


def all_checkpoints(output_dir, flavor) -> list:
    """All .ckpt files in the newest version dir, oldest first."""
    ckpt_dir = latest_version_dir(output_dir, flavor=flavor) / "checkpoints"
    ckpts = [p for p in ckpt_dir.glob("*.ckpt") if p.is_file()]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {ckpt_dir}")
    return sorted(ckpts, key=lambda p: p.stat().st_mtime)


def latest_checkpoint(output_dir, flavor) -> Path:
    return all_checkpoints(output_dir, flavor)[-1]


def best_and_last(trainer, output_dir, flavor) -> list:
    """The checkpoints worth scoring after a training run, as (label, path).

    Training keeps the top-k by val_loss; inference only needs the best of those
    and the most recently written one. Returns one entry when they coincide,
    which is the common case for a run that improved right up to the end.
    """
    last = latest_checkpoint(output_dir, flavor)

    best_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
    best = Path(best_path) if best_path else None
    if best is None or not best.is_file():
        logging.warning("No best checkpoint recorded by the trainer; scoring the last one only.")
        return [("last", last)]

    if best == last:
        return [("best/last", best)]
    return [("best", best), ("last", last)]


def run_dir_for_checkpoint(checkpoint_path: Path) -> Path:
    """The version dir a checkpoint belongs to (where its outputs are written)."""
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


# Load the frozen scaler params saved at training time so inference scales data/MC
# identically to training. Prefer the copy next to the checkpoint, else the output dir.
def load_scaler_params(output_dir, checkpoint_path):
    candidates = [
        run_dir_for_checkpoint(checkpoint_path) / "scaler_params.json",
        Path(output_dir) / "scaler_params.json",
    ]
    for path in candidates:
        if path.is_file():
            with open(path) as f:
                raw = json.load(f)
            logging.info("Loaded scaler params from %s", path)
            return {k: v for k, v in raw.items() if not k.startswith("_")}
    raise FileNotFoundError(
        "Could not find scaler_params.json for inference (looked in: "
        + ", ".join(str(c) for c in candidates)
        + "). Re-run training to regenerate it so inference reuses the training-time scaling."
    )
