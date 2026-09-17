"""Small shared helpers for the ABCD pipeline.

Everything here operates on the pipeline's in-memory event format: a plain dict
mapping column name -> 1D numpy array, with all columns the same length.
"""

import numpy as np

# Sentinel for missing/invalid values in flat columns (e.g. events with no
# second fat jet). Kept out of scaler fits and routed to NaN for the BDT.
MISSING_VALUE = -999.0


def flatten_cell(value):
    """Collapse a possibly-jagged cell to a scalar (first element, or MISSING_VALUE)."""
    if np.isscalar(value):
        return value
    return value[0] if len(value) > 0 else MISSING_VALUE


def to_flat_float_column(arr):
    """Return ``arr`` as a flat float64 column, flattening object cells if needed."""
    arr = np.asarray(arr)
    if arr.dtype == object:
        return np.asarray([flatten_cell(v) for v in arr], dtype=np.float64)
    return arr.astype(np.float64, copy=False)


def data_length(data):
    if not data:
        return 0
    first_key = next(iter(data))
    return len(data[first_key])


def apply_mask(data, mask):
    return {k: np.asarray(v)[mask] for k, v in data.items()}


def concat_chunks(chunks):
    """Concatenate a list of column dicts, unioning their keys."""
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


def concat_sig_bkg(sig_data, bkg_data):
    """Concatenate signal first, then background, and attach a 'label' column.

    Signal-first ordering is load-bearing: train/val indices computed on the
    combined array assume it.

    A column present on only one side (the systematic weights, which are read
    for signal only) is padded with NaN on the other, so every column keeps the
    full combined length. NaN rather than a neutral 1.0 because the value is not
    applicable there, and anything that does try to use it should fail visibly
    instead of quietly reading as "no variation".
    """
    n_sig = data_length(sig_data)
    n_bkg = data_length(bkg_data)

    column_order = list(sig_data.keys())
    for key in bkg_data.keys():
        if key not in sig_data:
            column_order.append(key)

    combined = {}
    for key in column_order:
        sig_col = (np.asarray(sig_data[key]) if key in sig_data
                   else np.full(n_sig, np.nan))
        bkg_col = (np.asarray(bkg_data[key]) if key in bkg_data
                   else np.full(n_bkg, np.nan))
        combined[key] = np.concatenate([sig_col, bkg_col])

    combined["label"] = np.concatenate(
        [
            np.ones(data_length(sig_data), dtype=np.int32),
            np.zeros(data_length(bkg_data), dtype=np.int32),
        ]
    )
    return combined


def evaluate_expression(data, expression):
    """Evaluate a config expression (preselection, derived var) over the columns."""
    local_dict = {k: np.asarray(v) for k, v in data.items()}
    return eval(expression, {"__builtins__": {}, "np": np, "abs": np.abs}, local_dict)


def score_column(flavor):
    """Primary DNN score column name for a training flavor."""
    return "dnn_score" if flavor == "single" else "dnn_0_score"
