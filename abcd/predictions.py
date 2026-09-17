"""Reading and writing the per-event prediction files.

Predictions are stored as Parquet: columnar, so the downstream cut scans read
only the columns they touch, typed, so float64 scores survive a round trip
exactly, and compressed, which matters at tens of millions of events.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SUFFIX = ".parquet"


def _to_arrow(values):
    arr = np.asarray(values)
    if arr.dtype == object:
        # e.g. the 'split' tag column, which numpy carries as object strings
        return pa.array(arr.tolist())
    return pa.array(arr)


def write_predictions(data, output_path):
    """Write a column dict (name -> equal-length 1D array) to Parquet."""
    columns = list(data.keys())
    arrays = [_to_arrow(data[col]) for col in columns]

    if not arrays:
        pq.write_table(pa.table({}), output_path)
        return

    lengths = {len(arr) for arr in arrays}
    if len(lengths) != 1:
        raise ValueError(f"All output columns must have the same length, got lengths={sorted(lengths)}")

    table = pa.Table.from_arrays(arrays, names=columns)
    pq.write_table(table, output_path, compression="zstd")


def read_predictions(path, columns=None):
    """Read a prediction file into a DataFrame.

    Pass ``columns`` to pull only those off disk — the main reason for the
    columnar format.
    """
    return pd.read_parquet(path, columns=columns)
