#!/usr/bin/env python
"""Histogram filling for physics analysis plots (parquet + hist).

``Plotter`` reads candidate post-processor Parquet files into ``hist.Hist``
histograms: data, stacked backgrounds split by ``shortname`` category, and
any number of signal samples. Only the nominal variation is drawn; aliases and
cuts are evaluated with numexpr. Drawing is delegated to ``rendering.py``,
which owns all CMS style conventions.
"""

import glob
import logging
import re
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import awkward as ak
import hist
from hist import Hist
import matplotlib.pyplot as plt

from rendering import PlotStyle, render_1d, render_2d, scale_hist

logger = logging.getLogger(__name__)

MAX_WORKERS = min(16, mp.cpu_count())
COM_TEV = {"Run2": 13.0, "Run3": 13.6}


def _eval_expr(expr: str, cols: dict):
    """Evaluate a config alias/cut expression over numpy columns.

    Restricted-eval scheme (only ``np`` and ``abs`` exposed, no builtins),
    matching ``abcd/common.py``; numpy vector comparisons use ``&``/``|`` with
    parenthesised operands, e.g. ``(pt > 30) & (abs(eta) < 2.5)``.
    """
    local_dict = {k: np.asarray(v) for k, v in cols.items()}
    return eval(expr, {"__builtins__": {}, "np": np, "abs": np.abs}, local_dict)


@dataclass
class Hist1D:
    """Configuration + filled storage for a 1D histogram.

    Attributes:
        var: Branch name to plot
        xlabel: Label for x-axis
        binning: Tuple of (nbins, xmin, xmax)
        scale: Scale factor applied to signal overlays when drawing
        logy: Use logarithmic y-axis
        data_cut: optional pandas ``query`` selection applied to MVA *data*
            only (not MC) when filling this histogram; used to blind data.
        hist_data/hist_bkg/hist_sig: filled ``hist.Hist`` objects (lists)
    """
    var: str
    xlabel: str
    binning: Tuple[int, float, float]
    scale: float = 1.0
    logy: bool = False
    data_cut: Optional[str] = None
    hist_data: List = field(default_factory=list)
    hist_bkg: List = field(default_factory=list)
    hist_sig: List = field(default_factory=list)

    def __post_init__(self):
        if not self.var:
            raise ValueError("Variable name cannot be empty")
        if len(self.binning) != 3:
            raise ValueError("Binning must be a tuple of (nbins, xmin, xmax)")
        if self.binning[0] <= 0:
            raise ValueError("Number of bins must be positive")
        if self.binning[1] >= self.binning[2]:
            raise ValueError("xmin must be less than xmax")
        if self.scale <= 0:
            raise ValueError("Scale factor must be positive")

    @property
    def vars(self) -> Tuple[str, ...]:
        return (self.var,)

    def new_hist(self) -> Hist:
        nbins, xmin, xmax = self.binning
        return Hist(
            hist.axis.Regular(int(nbins), xmin, xmax, name=self.var),
            storage=hist.storage.Weight(),
        )


@dataclass
class Hist2D:
    """Configuration + filled storage for a 2D histogram."""
    xvar: str
    yvar: str
    xlabel: str
    ylabel: str
    xbinning: Tuple[int, float, float]
    ybinning: Tuple[int, float, float]
    scale: float = 1.0
    logy: bool = False
    hist_data: List = field(default_factory=list)
    hist_bkg: List = field(default_factory=list)
    hist_sig: List = field(default_factory=list)

    def __post_init__(self):
        if not self.xvar or not self.yvar:
            raise ValueError("Variable names cannot be empty")
        for binning, name in [(self.xbinning, "xbinning"), (self.ybinning, "ybinning")]:
            if len(binning) != 3:
                raise ValueError(f"{name} must be a tuple of (nbins, min, max)")
            if binning[0] <= 0:
                raise ValueError(f"{name}: number of bins must be positive")
            if binning[1] >= binning[2]:
                raise ValueError(f"{name}: min must be less than max")

    @property
    def vars(self) -> Tuple[str, ...]:
        return (self.xvar, self.yvar)

    def new_hist(self) -> Hist:
        nx, xmin, xmax = self.xbinning
        ny, ymin, ymax = self.ybinning
        return Hist(
            hist.axis.Regular(int(nx), xmin, xmax, name=self.xvar),
            hist.axis.Regular(int(ny), ymin, ymax, name=self.yvar),
            storage=hist.storage.Weight(),
        )


class Plotter:
    """Fill and draw physics analysis plots from ROOT files or MVA Parquet files."""

    def __init__(
        self,
        sig: Optional[Union[str, List[str]]] = None,
        bkg: Optional[Union[str, List[str]]] = None,
        data: Optional[Union[str, List[str]]] = None,
        bkg_samples_labels: Optional[Dict[str, str]] = None,
        sig_samples_labels: Optional[List[str]] = None,
        cut: Optional[str] = None,
        data_cut: Optional[str] = None,
        year: Optional[Union[int, str]] = None,
        lumi: Optional[float] = None,
        mc_scale: float = 1.0,
        cms_label: str = "Preliminary",
        define_vars: Optional[Dict[str, str]] = None,
        tree: str = "Events",
        max_workers: int = MAX_WORKERS,
        mva_mc: Optional[Union[str, List[str]]] = None,
        mva_data: Optional[Union[str, List[str]]] = None,
        mva_label_col: str = "label",
        mva_weight_col: str = "weight",
        mva_data_cut: Optional[str] = None,
    ):
        """
        Args:
            sig/bkg/data: path(s)/glob(s) to ROOT files. ``sig`` may be a list
                of patterns, one per signal sample (each becomes its own curve).
            bkg_samples_labels: dict mapping a substring of ``shortname`` to a
                display label; background is split into these categories plus an
                "Other" catch-all. If None, background is a single stack entry.
            sig_samples_labels: display labels, one per signal sample.
            cut: optional selection, an uproot/numexpr expression (e.g. "pt>30").
                Applied to every sample (data + MC).
            data_cut: optional selection applied to data only, ANDed with
                ``cut``. Use it to restrict data to one year (e.g. "is2025")
                while leaving MC uncut (MC is instead rescaled via mc_scale).
            year/lumi: used for the CMS label (year picks sqrt(s)).
            mc_scale: multiplicative factor applied to all MC (signal +
                background) histograms after filling. Use it to rescale MC
                whose weights are normalized to one lumi down to another
                (e.g. full Run3 -> a single year). Data is never scaled.
            cms_label: text after "CMS" on the plots ("" for publication).
            define_vars: {name: expression} aliases evaluated by uproot.
            tree: TTree name (default "Events").
            max_workers: number of files to read concurrently.
            mva_mc: path(s)/glob(s) to Parquet file(s) written after MVA, holding
                signal + background in one file split by ``mva_label_col``
                (1 = signal, 0 = background). Consumed by ``make_mva_plots``.
            mva_data: path(s)/glob(s) to the Parquet file(s) holding data (no label
                column). Consumed by ``make_mva_plots``.
            mva_label_col/mva_weight_col: column names in the MVA files.
            mva_data_cut: default pandas ``query`` expression applied to the
                MVA data file only (MC is left uncut), used to blind data, e.g.
                "dnn_score < 0.9". Applied to every histogram that does not set
                its own ``Hist1D.data_cut`` (which overrides this). The
                expression selects the rows to keep and may reference any column
                in the file, not just plotted ones.
        """
        self.year = year
        self.lumi = lumi
        self.mc_scale = float(mc_scale)
        self.cms_label = cms_label
        self.tree = tree
        self.cut = cut
        self.data_cut = data_cut
        self.max_workers = max(1, int(max_workers))
        self.aliases = dict(define_vars or {})
        self.bkg_samples_labels = bkg_samples_labels
        self.sig_samples_labels = sig_samples_labels
        self.mva_label_col = mva_label_col
        self.mva_weight_col = mva_weight_col
        self.mva_data_cut = mva_data_cut

        if sig is None and bkg is None and data is None and mva_mc is None and mva_data is None:
            raise ValueError(
                "At least one of sig, bkg, data, mva_mc, or mva_data must be provided"
            )

        # Expand patterns up front so a bad path fails fast.
        self.data_files = self._expand_patterns(data) if data else None
        self.bkg_files = self._expand_patterns(bkg) if bkg else None
        self.mva_mc_files = self._expand_patterns(mva_mc) if mva_mc else None
        self.mva_data_files = self._expand_patterns(mva_data) if mva_data else None
        if sig is None:
            self.sig_sources = None
        elif isinstance(sig, str):
            self.sig_sources = [self._expand_patterns(sig)]
        else:
            self.sig_sources = [self._expand_patterns(s) for s in sig]

        for label, files in [("data", self.data_files), ("background", self.bkg_files)]:
            if files is not None:
                logger.info(f"Expanded {label} patterns to {len(files)} file(s)")
        if self.sig_sources is not None:
            logger.info(f"Expanded signal patterns to {sum(len(s) for s in self.sig_sources)} "
                        f"file(s) across {len(self.sig_sources)} sample(s)")

        self._validate_sample_labels()

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _validate_sample_labels(self) -> None:
        if self.sig_sources is not None and self.sig_samples_labels is not None:
            if len(self.sig_samples_labels) != len(self.sig_sources):
                raise ValueError(
                    f"Number of signal labels ({len(self.sig_samples_labels)}) must match "
                    f"number of signal samples ({len(self.sig_sources)})"
                )
        if self.bkg_samples_labels is None and self.bkg_files is not None:
            logger.warning("No background labels provided; using a single background entry")

    def define_variable(self, var_name: str, var_expr: str) -> None:
        """Register an alias expression (evaluated by uproot when reading)."""
        self.aliases[var_name] = var_expr

    def define_variables(self, var_dict: Dict[str, str]) -> None:
        self.aliases.update(var_dict)

    def _expand_patterns(self, patterns: Union[str, List[str]]) -> List[str]:
        """Expand glob (and, as a fallback, regex) patterns to file paths."""
        pattern_list = [patterns] if isinstance(patterns, str) else list(patterns)
        expanded_files: List[str] = []

        for pattern in pattern_list:
            glob_matches = glob.glob(pattern)
            if glob_matches:
                expanded_files.extend(sorted(glob_matches))
                continue

            # Regex fallback against files under the longest existing directory.
            if any(c in pattern for c in '^$[]()+?*.|\\'):
                parts = pattern.split('/')
                search_dir, regex_pattern = Path('.'), pattern
                # Narrow to the longest leading directory that actually exists,
                # then treat the remainder as a regex. Works for absolute
                # patterns too. Never fall back to search_dir='/': a fixed-depth
                # glob from the filesystem root walks the whole tree, follows
                # /proc/self/root back into itself, and stats /run/docker.sock,
                # which raises PermissionError [Errno 13].
                for i in range(len(parts), 0, -1):
                    candidate = '/'.join(parts[:i])
                    if candidate and Path(candidate).is_dir():
                        search_dir = Path(candidate)
                        regex_pattern = '/'.join(parts[i:])
                        break
                try:
                    regex = re.compile(regex_pattern)
                    # Bound the walk to the directory depth implied by the
                    # pattern. An unbounded rglob over a network filesystem
                    # (e.g. /ceph) walks the entire tree and hangs when the
                    # pattern matches nothing.
                    depth = regex_pattern.count('/') + 1
                    candidates = search_dir.glob('/'.join(['*'] * depth))
                    matches = [
                        str(fp) for fp in candidates
                        if fp.is_file() and (
                            regex.search(str(fp.relative_to(search_dir))) or regex.search(str(fp))
                        )
                    ]
                    if matches:
                        expanded_files.extend(sorted(set(matches)))
                        continue
                except re.error:
                    pass

            if Path(pattern).exists():
                expanded_files.append(pattern)
            else:
                logger.warning(f"Pattern '{pattern}' did not match any files")

        if not expanded_files:
            raise FileNotFoundError(f"No files found matching patterns: {pattern_list}")
        return expanded_files

    # ------------------------------------------------------------------ #
    # Reading / filling (ROOT trees)
    # ------------------------------------------------------------------ #
    def _available_branches(self, files: List[str]) -> set:
        """Column names present in the first parquet file."""
        import pyarrow.parquet as pq
        try:
            return set(pq.read_schema(files[0]).names)
        except Exception as e:
            logger.warning(f"Could not read column list from {files[0]}: {e}")
            return set()

    def _resolve_vars(self, hists: List, available: set) -> List[str]:
        """Branches to read: hist vars that exist in the tree or are aliases."""
        wanted = {v for h in hists for v in h.vars}
        usable, missing = [], []
        for v in wanted:
            (usable if (v in available or v in self.aliases) else missing).append(v)
        if missing:
            logger.warning(f"Skipping variables not found in tree: {sorted(missing)}")
        return usable

    def _combined_data_cut(self) -> Optional[str]:
        """Combined selection for data: ``cut`` ANDed with ``data_cut``."""
        cuts = [c for c in (self.cut, self.data_cut) if c]
        if not cuts:
            return None
        return " & ".join(f"({c})" for c in cuts)

    def _iterate(self, files: List[str], expressions: List[str], cut: Optional[str] = None):
        """Read each candidate post-processor parquet fully and yield its arrays.

        Keeps the nominal variation, evaluates the ``define_vars`` aliases and the
        ``cut`` with numexpr (the candidate columns are scalar-per-event, so no
        jagged handling is needed), and yields an awkward record matching the fill
        interface. Only the columns referenced by the requested vars, aliases and
        cut are read — parquet is columnar, so the rest never leaves disk.
        """
        import pyarrow.parquet as pq

        cut = cut if cut is not None else self.cut
        aliases = self.aliases or {}
        for f in files:
            header = set(pq.read_schema(f).names)

            def _cols_in(expr):
                return {c for c in header if re.search(rf"\b{re.escape(c)}\b", expr)}

            read = {e for e in expressions if e in header}
            for expr in aliases.values():
                read |= _cols_in(expr)
            if cut:
                read |= _cols_in(cut)
            if "variation" in header:
                read.add("variation")
            if not read:
                continue

            table = pq.read_table(f, columns=sorted(read))
            data = {n: table[n].to_numpy(zero_copy_only=False) for n in table.column_names}

            # Nominal event set only (the plotter does not draw JES/JER variations);
            # the variation tag has served its purpose once filtered.
            if "variation" in data:
                keep = data.pop("variation").astype(str) == "nominal"
                data = {n: v[keep] for n, v in data.items()}

            # define_vars aliases, then the selection cut, evaluated on the numpy
            # columns (same restricted-eval scheme abcd uses in common.py).
            for name, expr in aliases.items():
                try:
                    data[name] = _eval_expr(expr, data)
                except Exception:
                    pass
            if cut:
                mask = np.asarray(_eval_expr(cut, data), dtype=bool)
                data = {n: v[mask] for n, v in data.items()}

            # awkward can't build a record from object dtype, so string columns
            # (e.g. shortname) are converted to awkward strings via a Python list.
            record = {}
            for name, vals in data.items():
                vals = np.asarray(vals)
                record[name] = ak.Array(vals.tolist()) if vals.dtype == object else vals
            yield ak.Array(record)

    @staticmethod
    def _flatten(values, weight):
        """Flatten a (possibly jagged) per-event branch to a flat array,
        broadcasting the per-event weight across each event's elements.

        Mirrors ROOT's Histo1D, which fills every element of a vector column
        with that event's weight. Flat (scalar-per-event) branches pass through.
        """
        if getattr(values, "ndim", 1) > 1:
            weight = ak.flatten(ak.broadcast_arrays(weight, values)[0])
            values = ak.flatten(values)
        return np.asarray(values), np.asarray(weight)

    def _fill_one(self, cfg, hobj: Hist, arrays, weight) -> None:
        """Fill one histogram for one config from per-event arrays + weights."""
        if isinstance(cfg, Hist1D):
            vals, w = self._flatten(arrays[cfg.var], weight)
            hobj.fill(vals, weight=w)
        else:
            xv, yv = arrays[cfg.xvar], arrays[cfg.yvar]
            if getattr(xv, "ndim", 1) > 1 or getattr(yv, "ndim", 1) > 1:
                xb, yb, wb = ak.broadcast_arrays(xv, yv, weight)
                xv, yv, w = ak.flatten(xb), ak.flatten(yb), ak.flatten(wb)
            else:
                w = weight
            hobj.fill(np.asarray(xv), np.asarray(yv), weight=np.asarray(w))

    @staticmethod
    def _category_masks(shortname, samples: List[str]) -> List[np.ndarray]:
        """Boolean masks assigning each row to a background category.

        ``shortname`` is matched against each entry of ``samples`` by substring
        (the same rule ``bkg_samples_labels`` uses). Returns one mask per sample
        in order, followed by a final "Other" catch-all for rows matching none.
        """
        shortname = np.asarray(shortname).astype(str)
        masks, assigned = [], np.zeros(len(shortname), dtype=bool)
        for sample in samples:
            m = np.char.find(shortname, sample) >= 0
            assigned |= m
            masks.append(m)
        masks.append(~assigned)  # "Other"
        return masks

    def _fill_file(self, file: str, hists: List, expressions: List[str],
                   have_weight: bool, cut: Optional[str],
                   samples: Optional[List[str]]) -> Tuple[List[List[Hist]], int]:
        """Read one file and fill a fresh set of histograms from it.

        Returns ``hobjs[config_index][category_index]``. With ``samples`` None
        there is a single category; otherwise one per sample plus "Other".
        """
        n_cats = len(samples) + 1 if samples is not None else 1
        hobjs = [[h.new_hist() for _ in range(n_cats)] for h in hists]
        processed = 0
        for arrays in self._iterate([file], expressions, cut=cut):
            n = len(arrays[expressions[0]]) if expressions else 0
            w = arrays["weight"] if have_weight else np.ones(n)

            if samples is not None:
                shortname = np.asarray(ak.to_list(arrays["shortname"])).astype(str)
                masks = self._category_masks(shortname, samples)
            else:
                masks = [None]

            for cfg, cat_hists in zip(hists, hobjs):
                if not all(v in arrays.fields for v in cfg.vars):
                    continue
                for hobj, mask in zip(cat_hists, masks):
                    if mask is None:
                        self._fill_one(cfg, hobj, arrays, w)
                    elif mask.any():
                        sub = {v: arrays[v][mask] for v in cfg.vars}
                        self._fill_one(cfg, hobj, sub, w[mask])
            processed += n
        return hobjs, processed

    def _fill_source(self, files: List[str], hists: List, label: str,
                     cut: Optional[str] = None,
                     categorize: bool = False) -> List[List[Hist]]:
        """Fill all histogram configs from one sample in a single pass.

        Files are read concurrently (one worker per file); each worker fills
        its own histograms, which are summed into the totals as they finish.
        ``cut`` overrides ``self.cut`` for this sample (e.g. a data-only cut).
        With ``categorize`` the background is split by ``shortname`` into the
        ``bkg_samples_labels`` categories plus "Other".

        Returns ``totals[config_index][category_index]`` (one category unless
        ``categorize``).
        """
        available = self._available_branches(files)
        usable = self._resolve_vars(hists, available)
        have_weight = "weight" in available
        if not have_weight:
            logger.warning(f"{label}: no 'weight' branch; filling unweighted")

        samples = None
        if categorize and self.bkg_samples_labels is not None:
            if "shortname" in available:
                samples = list(self.bkg_samples_labels.keys())
            else:
                logger.warning("'shortname' branch missing; collapsing background into one entry")

        expressions = list(usable)
        if have_weight:
            expressions.append("weight")
        if samples is not None:
            expressions.append("shortname")

        n_cats = len(samples) + 1 if samples is not None else 1
        totals = [[h.new_hist() for _ in range(n_cats)] for h in hists]

        processed, t0 = 0, time.time()
        workers = min(self.max_workers, len(files))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(self._fill_file, f, hists, expressions, have_weight, cut, samples)
                for f in files
            ]
            for fut in as_completed(futures):
                hobjs, n = fut.result()
                for total_cats, cat_hists in zip(totals, hobjs):
                    for total, h in zip(total_cats, cat_hists):
                        total += h
                processed += n
                self._log_progress(label, processed, t0)
        logger.info(f"{label}: done, {processed} events in {time.time() - t0:.1f}s")
        return totals

    def _log_progress(self, label: str, processed: int, t0: float) -> None:
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        logger.info(f"{label}: {processed:,} events read ({rate:,.0f} evt/s)")

    # ------------------------------------------------------------------ #
    # MVA prediction reading / filling
    # ------------------------------------------------------------------ #
    def _read_mva_frame(self, files: List[str], hists: List["Hist1D"],
                        want_label: bool, cut_exprs: Optional[List[str]] = None,
                        want_shortname: bool = False
                        ) -> Tuple["object", List[str], bool]:
        """Read the columns needed for ``hists`` from one or more MVA Parquet files.

        Returns ``(df, usable_vars, have_weight)``. Only the histogram vars
        present in the file (plus weight, label when ``want_label``, the
        ``shortname`` column when ``want_shortname``, and any columns referenced
        by ``cut_exprs``) are read — Parquet is columnar, so the unread columns
        never leave disk. Vars absent from the file are dropped with a warning,
        mirroring the ROOT path's ``_resolve_vars``. The frame is returned
        unfiltered; per-histogram cuts are applied by the caller so each
        histogram can blind data differently.
        """
        import pandas as pd
        import pyarrow.parquet as pq

        header = list(pq.read_schema(files[0]).names)
        wanted = sorted({h.var for h in hists})
        usable = [v for v in wanted if v in header]
        missing = [v for v in wanted if v not in header]
        if missing:
            logger.warning(f"MVA: skipping variables not found in file: {missing}")

        cols = list(usable)
        have_weight = self.mva_weight_col in header
        if have_weight:
            cols.append(self.mva_weight_col)
        else:
            logger.warning(f"MVA: no '{self.mva_weight_col}' column; filling unweighted")
        if want_label:
            if self.mva_label_col not in header:
                raise KeyError(f"MVA MC file missing label column '{self.mva_label_col}'")
            cols.append(self.mva_label_col)
        if want_shortname and "shortname" in header:
            cols.append("shortname")
        # Also read any columns the cut expressions reference so query() sees them.
        for expr in cut_exprs or []:
            cols += [c for c in header
                     if c not in cols and re.search(rf"\b{re.escape(c)}\b", expr)]

        frames = [pd.read_parquet(f, columns=cols) for f in files]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        logger.info(f"MVA: read {len(df):,} rows x {len(cols)} col(s) from {len(files)} file(s)")
        return df, usable, have_weight

    def _fill_mva_mc(self, hists: List["Hist1D"]) -> None:
        """Fill signal + per-category background hists from the MVA MC file."""
        labels_mode = self.bkg_samples_labels is not None
        df, usable, have_weight = self._read_mva_frame(
            self.mva_mc_files, hists, want_label=True, want_shortname=labels_mode)
        if labels_mode and "shortname" not in df.columns:
            logger.warning("MVA: 'shortname' column missing; "
                           "collapsing background into one entry")
            labels_mode = False
        w = df[self.mva_weight_col].to_numpy() if have_weight else np.ones(len(df))
        label = df[self.mva_label_col].to_numpy()
        is_sig, is_bkg = label == 1, label == 0

        # Per-category background masks (samples..., then "Other"), each
        # ANDed with is_bkg so signal rows never leak into the stack.
        if labels_mode:
            samples = list(self.bkg_samples_labels.keys())
            bkg_masks = [m & is_bkg
                         for m in self._category_masks(df["shortname"].to_numpy(), samples)]
        else:
            bkg_masks = [is_bkg]

        for cfg in hists:
            if cfg.var not in usable:
                continue
            vals = df[cfg.var].to_numpy()
            sig_h = cfg.new_hist()
            sig_h.fill(vals[is_sig], weight=w[is_sig])
            cfg.hist_sig = [scale_hist(sig_h, self.mc_scale)]

            cat_hists = []
            for mask in bkg_masks:
                bkg_h = cfg.new_hist()
                bkg_h.fill(vals[mask], weight=w[mask])
                cat_hists.append(scale_hist(bkg_h, self.mc_scale))
            cfg.hist_bkg = cat_hists

    def _fill_mva_data(self, hists: List["Hist1D"]) -> None:
        """Fill data hists from the MVA data file, applying per-hist blinding."""
        cut_exprs = [c for c in
                     ([h.data_cut for h in hists] + [self.mva_data_cut]) if c]
        df, usable, have_weight = self._read_mva_frame(
            self.mva_data_files, hists, want_label=False, cut_exprs=cut_exprs)
        for cfg in hists:
            if cfg.var not in usable:
                continue
            cut = cfg.data_cut or self.mva_data_cut
            sub = df.query(cut) if cut else df
            if cut:
                logger.info(f"MVA data cut for '{cfg.var}': '{cut}' "
                            f"kept {len(sub):,}/{len(df):,} rows")
            w = sub[self.mva_weight_col].to_numpy() if have_weight else np.ones(len(sub))
            data_h = cfg.new_hist()
            data_h.fill(sub[cfg.var].to_numpy(), weight=w)
            cfg.hist_data = [data_h]

    # ------------------------------------------------------------------ #
    # Top-level drivers
    # ------------------------------------------------------------------ #
    def make_plots(
        self,
        hists: List[Union[Hist1D, Hist2D]],
        density: bool = False,
        save: bool = True,
        save_path: str = "plots",
        logy: bool = False,
    ) -> None:
        """Fill all histograms from the ROOT files, then draw them."""
        if not hists:
            logger.warning("No histograms provided to plot")
            return
        logger.info(f"Creating {len(hists)} plots")

        if self.mc_scale != 1.0:
            logger.info(f"Rescaling all MC by mc_scale={self.mc_scale:.4g}")

        # Each source is read once; all histograms are filled in that single pass.
        if self.data_files:
            totals = self._fill_source(self.data_files, hists, "data",
                                       cut=self._combined_data_cut())
            for cfg, cats in zip(hists, totals):
                cfg.hist_data = [cats[0]]

        if self.sig_sources:
            for i, source in enumerate(self.sig_sources):
                totals = self._fill_source(source, hists, f"signal[{i}]")
                for cfg, cats in zip(hists, totals):
                    cfg.hist_sig.append(scale_hist(cats[0], self.mc_scale))

        if self.bkg_files:
            totals = self._fill_source(self.bkg_files, hists, "background",
                                       categorize=True)
            for cfg, cats in zip(hists, totals):
                cfg.hist_bkg = [scale_hist(h, self.mc_scale) for h in cats]

        self._render_all(hists, density=density, save=save,
                         save_path=save_path, logy=logy)

    def make_mva_plots(
        self,
        hists: List["Hist1D"],
        density: bool = False,
        save: bool = True,
        save_path: str = "plots",
        logy: bool = False,
    ) -> None:
        """Fill 1D histograms from MVA Parquet files (e.g. bdt_score), then draw them.

        Signal and background come from ``mva_mc`` (one file, split by the
        label column), data from ``mva_data``. Styling is identical to
        :meth:`make_plots`.
        """
        hists = [h for h in hists if isinstance(h, Hist1D)]
        if not hists:
            logger.warning("No 1D histograms provided for MVA plots")
            return
        logger.info(f"Creating {len(hists)} MVA plot(s) from Parquet")

        if self.mva_mc_files:
            self._fill_mva_mc(hists)
        if self.mva_data_files:
            self._fill_mva_data(hists)

        self._render_all(hists, density=density, save=save,
                         save_path=save_path, logy=logy)

    def _render_all(self, hists: List, density: bool, save: bool,
                    save_path: str, logy: bool) -> None:
        success = 0
        for i, cfg in enumerate(hists):
            try:
                if isinstance(cfg, Hist1D):
                    if logy:
                        cfg.logy = True
                    self.plot1D(cfg, density=density, save=save, save_path=save_path)
                else:
                    self.plot2D(cfg, save=save, save_path=save_path)
                success += 1
            except Exception as e:
                logger.error(f"Error plotting histogram {i + 1} "
                             f"({getattr(cfg, 'var', '?')}): {e}")
                logger.debug("", exc_info=True)
                plt.close('all')
        logger.info(f"Successfully created {success}/{len(hists)} plots")

    # ------------------------------------------------------------------ #
    # Drawing (delegates to rendering.py)
    # ------------------------------------------------------------------ #
    def _plot_style(self) -> PlotStyle:
        lumi = round(self.lumi, 1) if self.lumi else None
        if lumi is not None and lumi == int(lumi):
            lumi = int(lumi)
        return PlotStyle(
            cms_label=self.cms_label,
            lumi=lumi,
            com=COM_TEV.get(str(self.year), 13.6),
        )

    def _bkg_labels(self, n_hists: int) -> List[str]:
        if self.bkg_samples_labels is None:
            return ["Background"]
        labels = list(self.bkg_samples_labels.values())
        if n_hists == len(labels) + 1:
            return labels + ["Other"]
        if n_hists != len(labels):
            logger.warning(f"Background stack/label mismatch: "
                           f"{n_hists} hists vs {len(labels)} labels")
            return [f"Background {i + 1}" for i in range(n_hists)]
        return labels

    def plot1D(self, cfg: Hist1D, density: bool = False, save: bool = True,
               save_path: str = "plots") -> None:
        bkg_labels = self._bkg_labels(len(cfg.hist_bkg)) if cfg.hist_bkg else None
        render_1d(cfg, self._plot_style(), bkg_labels=bkg_labels,
                  sig_labels=self.sig_samples_labels, density=density,
                  save=save, save_path=save_path)

    def plot2D(self, cfg: Hist2D, save: bool = True, save_path: str = "plots") -> None:
        render_2d(cfg, self._plot_style(), save=save, save_path=save_path)
