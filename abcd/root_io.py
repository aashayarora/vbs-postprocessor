"""Reading event columns from candidate post-processor parquet into flat numpy dicts."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from common import apply_mask, concat_chunks, data_length, evaluate_expression
from systematics import SYST_WEIGHTS, ratio_columns


def _read_parquet_frame(path, branches, dataset_idx, sample_idx, variation="nominal"):
    """Read one candidate post-processor parquet, selecting a single ``variation``.

    The post-processor already pre-splits each systematic into ``<branch>_syst_up`` /
    ``_dn`` ratio columns, so a requested SYST_WEIGHTS branch maps directly to those two
    columns (forgiven when absent, e.g. on data). Non-systematic branches must exist —
    a missing one is a config error and fails loudly, mirroring the ROOT reader. The
    string ``variation`` column is carried through so predictions record their variation.
    """
    available = set(pq.read_schema(path).names)

    read_cols, missing = [], []
    for branch in branches:
        if branch in SYST_WEIGHTS:
            read_cols.extend(c for c in ratio_columns(branch) if c in available)
        elif branch in available:
            read_cols.append(branch)
        else:
            missing.append(branch)
    if missing:
        raise KeyError(f"{path}: required column(s) not found: {sorted(missing)}")
    # Always carry the variation tag (event-set selection / datacards) and the
    # data-taking year (per-year JES decorrelation in the datacards) when present.
    for meta in ("variation", "year"):
        if meta in available:
            read_cols.append(meta)
    read_cols = list(dict.fromkeys(read_cols))

    table = pq.read_table(path, columns=read_cols)
    columns = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}

    # variation=None keeps every variation (the column is carried through for the
    # datacards to split on); a string keeps only that variation's event set.
    if variation is not None and "variation" in columns:
        keep = columns["variation"].astype(str) == variation
        columns = {name: vals[keep] for name, vals in columns.items()}

    n_events = len(next(iter(columns.values()))) if columns else 0
    columns["dataset_idx"] = np.full(n_events, dataset_idx, dtype=np.int32)
    columns["sample_idx"] = np.full(n_events, sample_idx, dtype=np.int32)
    return columns


def load_data(paths, features, extra_vars, num_workers=1, variation="nominal"):
    branches = list(dict.fromkeys(features + extra_vars))

    def _read(path, br, di, si):
        return _read_parquet_frame(path, br, di, si, variation)

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
                pool.submit(_read, path, branches, dataset_idx, sample_idx): dataset_idx
                for dataset_idx, path, sample_idx in indexed_paths
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading parquet files"):
                dataset_idx = futures[future]
                chunks[dataset_idx] = future.result()
    else:
        for dataset_idx, path, sample_idx in tqdm(indexed_paths, total=len(indexed_paths), desc="Loading parquet files"):
            chunks[dataset_idx] = _read(path, branches, dataset_idx, sample_idx)

    chunks = [chunk for chunk in chunks if chunk is not None]

    data = concat_chunks(chunks)
    logging.info("Loaded %d events from %d files.", data_length(data), len(paths))
    if "weight" in data:
        weights = np.asarray(data["weight"])
        data = apply_mask(data, (weights > 0) & (weights < 1e4))

    return data


def apply_preselection(data, expression, label=""):
    """Apply the config preselection expression (no-op if expression is falsy)."""
    if not expression:
        return data
    mask = np.asarray(evaluate_expression(data, expression), dtype=bool)
    out = apply_mask(data, mask)
    logging.info("After preselection - %s samples: %d", label or "input", data_length(out))
    return out
