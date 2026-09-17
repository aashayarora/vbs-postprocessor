"""Derived variables, feature scaling, and class-weight normalization."""

import json
import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from common import MISSING_VALUE, data_length, evaluate_expression, to_flat_float_column


def apply_derived_vars(data, derived_vars_cfg):
    """Attach config-defined derived columns (``new_var: expression``) to the data."""
    if not derived_vars_cfg:
        return data

    out = {k: np.asarray(v).copy() for k, v in data.items()}
    n_events = data_length(out)

    for new_var, expression in derived_vars_cfg.items():
        if not isinstance(expression, str):
            raise ValueError(f"derived_vars['{new_var}'] must be a string expression")

        try:
            values = np.asarray(evaluate_expression(out, expression))
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


def _fit_minmax_scale(values, valid_mask):
    scaled = np.zeros_like(values, dtype=np.float64)
    scaler = MinMaxScaler()
    scaler.fit(values[valid_mask].reshape(-1, 1))
    scaled[valid_mask] = scaler.transform(values[valid_mask].reshape(-1, 1)).ravel()
    return scaled, float(scaler.data_min_[0]), float(scaler.data_max_[0])


def _apply_saved_scale(values, valid_mask, fmin, fmax):
    scaled = np.zeros_like(values, dtype=np.float64)
    fmin = float(fmin)
    data_range = float(fmax) - fmin
    scale = 1.0 / data_range if data_range != 0 else 1.0
    scaled[valid_mask] = (values[valid_mask] - fmin) * scale
    return scaled


def _scale_column(out, fitted_params, name, transform, saved):
    """Transform + minmax-scale one column in place; record the fitted params."""
    arr = np.asarray(out[name], dtype=np.float64)
    valid = (arr != MISSING_VALUE) & np.isfinite(arr)
    if transform == "log":
        positive = valid & (arr > 0)
        arr[valid & ~positive] = MISSING_VALUE
        valid = positive
        arr[valid] = np.log(arr[valid])

    if saved is not None:
        out[name] = _apply_saved_scale(arr, valid, saved["min"], saved["max"])
        fitted_params[name] = {"transform": transform, "min": float(saved["min"]), "max": float(saved["max"])}
        return

    if valid.sum() == 0:
        logging.warning("Feature '%s' has no valid samples, skipping scaler fit", name)
        out[name] = np.zeros_like(arr, dtype=np.float64)
        fitted_params[name] = {"transform": transform, "min": 0.0, "max": 1.0}
        return

    out[name], fmin, fmax = _fit_minmax_scale(arr, valid)
    fitted_params[name] = {"transform": transform, "min": fmin, "max": fmax}


def preprocess_data(data, training_features, feature_transforms, constraint_var, scaler_output_path=None, scaler_params=None):
    out = {k: np.asarray(v).copy() for k, v in data.items()}
    cols_to_clean = list(dict.fromkeys(training_features + [constraint_var]))
    for col in cols_to_clean:
        if col not in out:
            continue
        arr = to_flat_float_column(out[col])
        arr = np.where(np.isfinite(arr), arr, np.nan)
        out[col] = np.nan_to_num(arr, nan=MISSING_VALUE, posinf=MISSING_VALUE, neginf=MISSING_VALUE)

    def _saved_for(name):
        if scaler_params is None:
            return None
        if name not in scaler_params:
            raise ValueError(
                f"scaler_params provided but has no entry for '{name}'; the checkpoint's "
                f"scaler_params.json is out of sync with the config's features."
            )
        return scaler_params[name]

    fitted_params = {}
    for feat in training_features:
        _scale_column(out, fitted_params, feat, feature_transforms.get(feat, "none"), _saved_for(feat))
    if constraint_var not in training_features:
        _scale_column(out, fitted_params, constraint_var, "none", _saved_for(constraint_var))

    if scaler_output_path is not None:
        scaler_out = {
            "_training_features": training_features,
            "_constraint_var": constraint_var,
        }
        scaler_out.update(fitted_params)
        with open(scaler_output_path, "w") as f:
            json.dump(scaler_out, f, indent=2)
        logging.info("Saved scaler params to %s", scaler_output_path)
    return out


def normalize_class_weights(sig_data, bkg_data):
    """Rescale weights so signal and background each sum to 1."""
    sig = {k: np.asarray(v).copy() for k, v in sig_data.items()}
    bkg = {k: np.asarray(v).copy() for k, v in bkg_data.items()}

    sig["weight"] = sig["weight"] / np.sum(sig["weight"])
    bkg["weight"] = bkg["weight"] / np.sum(bkg["weight"])
    return sig, bkg
