#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow
import ROOT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import RECONSTRUCTORS, CHANNEL_GATE  # noqa: E402
from objects import build_collections, add_weight_columns  # noqa: E402
from cutflow import Cutflow, parse_preselection_logs, write_combined_table  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRESEL_DIR = str(Path(HERE).parent.parent / "cmstas-run3-vbsvvh" / "preselection")
SIGNAL_POINT_RE = r"VBS[A-Z]+H(?:_[OS]S)?_c2v[0-9p]+_c3_[0-9p]+"
PASSTHROUGH = ["run", "luminosityBlock", "event", "shortname", "year"]


def load_helpers():
    """Declare the C++ helper functions (dR / pairing / invariant-mass) into cling."""
    with open(os.path.join(HERE, "helpers.h")) as fh:
        if not ROOT.gInterpreter.Declare(fh.read()):
            raise RuntimeError("failed to declare postprocess/helpers.h into cling")


def detect_variations(columns) -> list[str]:
    """Variation suffixes present in the file: those with both a varied FatJet_pt and the
    matching per-variation good-jet masks. Nominal ("") is always first."""
    cols = set(str(c) for c in columns)
    out = [""]
    for c in sorted(cols):
        if not c.startswith("FatJet_pt_"):
            continue
        sfx = c[len("FatJet_pt_"):]
        needed = [f"FatJet_isGood_{sfx}", f"Jet_isGood_{sfx}",
                  f"Jet_pt_{sfx}", f"vbs_jet1_Jetidx_{sfx}"]
        if all(n in cols for n in needed):
            out.append(sfx)
    return out


def book_variation(df, channel, sfx, passthrough, cutflow=None):
    df = build_collections(df, sfx)

    flag = f"passes_{channel}_{'nom' if not sfx else sfx}"
    if flag in set(str(c) for c in df.GetColumnNames()):
        df = df.Filter(f"{flag} == 1", f"channel gate ({flag})")
    else:
        print(f"    [warn] {flag} not in input; falling back to recomputed gate")
        df = df.Filter(CHANNEL_GATE[channel], "channel gate (recomputed fallback)")
    if cutflow is not None:
        cutflow.add(df, "channel selection (nominal)")

    df, cand_cols = RECONSTRUCTORS[channel](df, cutflow)
    df, weight_cols = add_weight_columns(df)

    present = set(str(c) for c in df.GetColumnNames())
    out_cols = [c for c in passthrough if c in present] + weight_cols + cand_cols
    return df.AsNumpy(out_cols, lazy=True), out_cols, df


_RVEC_RE = re.compile(r"ROOT::VecOps::RVec<(.+)>")
_CPP_DTYPES = {"float": "float32", "double": "float64", "int": "int32", "unsigned int": "uint32",
               "bool": "bool", "Long64_t": "int64", "ULong64_t": "uint64"}


def _to_awkward(values, cpp_type):
    if values.dtype != object:
        return ak.from_numpy(values)
    if cpp_type in ("string", "std::string"):
        return ak.from_numpy(values.astype(str))
    m = _RVEC_RE.fullmatch(cpp_type)
    if not m or m.group(1) not in _CPP_DTYPES:
        raise TypeError(f"unsupported AsNumpy column type {cpp_type}")
    dtype = np.dtype(_CPP_DTYPES[m.group(1)])
    counts = np.fromiter(map(len, values), np.int64, len(values))
    content = (np.concatenate([np.asarray(v, dtype=dtype) for v in values]) if len(values)
               else np.empty(0, dtype))
    return ak.unflatten(content, counts)


def mirror_output_path(input_file, output_dir):
    p = Path(input_file)
    parts = [name for name in (p.parent.parent.name, p.parent.name) if name]
    return str(Path(output_dir, *parts, p.stem + ".parquet"))


def process_file(input_files, tree, channel, nominal_only, want_cutflow=False):
    """Reconstruct every variation of ``input_files`` in one event loop; return the combined
    awkward record array (with a ``variation`` field) and the cutflow rows (or None)."""
    df = ROOT.RDataFrame(tree, input_files)
    all_cols = [str(c) for c in df.GetColumnNames()]
    variations = [""] if nominal_only else detect_variations(all_cols)
    print(f"[postprocess] channel={channel}  files={len(input_files)}  "
          f"variations={[v or 'nominal' for v in variations]}")

    cf = Cutflow() if want_cutflow else None
    booked = [(sfx, *book_variation(df, channel, sfx, PASSTHROUGH, cf if sfx == "" else None))
              for sfx in variations]
    pieces = []
    for sfx, result, cols, node in booked:
        data = result.GetValue()  # the first GetValue runs the loop for every variation
        arr = ak.zip({c: _to_awkward(data[c], node.GetColumnType(c) if data[c].dtype == object
                                     else None) for c in cols}, depth_limit=1)
        label = sfx if sfx else "nominal"
        arr = ak.with_field(arr, ak.Array([label] * len(arr)), "variation")
        print(f"  - {label:<24} {len(arr):>10} events")
        pieces.append(arr)

    combined = ak.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    return combined, (cf.rows() if cf is not None else None)


# awkward's default parquet codec; the LCG views' pyarrow is built without it.
PARQUET_CODEC = "zstd" if pyarrow.Codec.is_available("zstd") else "snappy"


def write_parquet(arr, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    part_path = output_path + ".part"
    ak.to_parquet(arr, part_path, compression=PARQUET_CODEC)
    os.replace(part_path, output_path)
    print(f"[postprocess] wrote {len(arr)} rows -> {output_path}")


def process_to_parquet(input_files, output_path, tree, channel, nominal_only, want_cutflow=False):
    arr, rows = process_file(input_files, tree, channel, nominal_only, want_cutflow)
    write_parquet(arr, output_path)
    return rows


def _dask_task(path, tree, channel, nominal_only, want_cutflow, helpers_src):
    """One input file on a Dask worker. Declaring the helpers again in a warm worker is a
    no-op (helpers.h has an include guard)."""
    if not ROOT.gInterpreter.Declare(helpers_src):
        raise RuntimeError("failed to declare postprocess/helpers.h into cling")
    return process_file([path], tree, channel, nominal_only, want_cutflow)


def _run_dask(args, input_files, want_cf, on_rows):
    import cloudpickle
    from dask.distributed import Client, as_completed
    import channels
    import cutflow
    import objects
    for mod in (channels, cutflow, objects):
        cloudpickle.register_pickle_by_value(mod)
    with open(os.path.join(HERE, "helpers.h")) as fh:
        helpers_src = fh.read()

    todo = [(f, mirror_output_path(f, args.output_dir)) for f in input_files]
    if args.skip_existing:
        todo = [(f, out) for f, out in todo if not os.path.exists(out)]
    # Largest files first, so the longest tasks do not end up in the tail.
    todo.sort(key=lambda t: os.path.getsize(t[0]) if os.path.exists(t[0]) else 0, reverse=True)
    print(f"[postprocess] dask mode: {len(todo)} of {len(input_files)} input file(s) -> "
          f"{args.output_dir}  via {args.scheduler}", flush=True)

    client = Client(args.scheduler)
    futures = {client.submit(_dask_task, f, args.tree, args.channel, args.nominal_only, want_cf,
                             helpers_src, pure=False, retries=args.retries): (f, out)
               for f, out in todo}
    cluster_failed = []
    for i, fut in enumerate(as_completed(futures), 1):
        f, out = futures.pop(fut)
        try:
            arr, rows = fut.result()
        except Exception as exc:
            print(f"[{i}/{len(todo)}] cluster failure {f}: {type(exc).__name__}: {exc}", flush=True)
            cluster_failed.append(f)
            continue
        write_parquet(arr, out)
        on_rows(f, rows)
        print(f"[{i}/{len(todo)}] {f}", flush=True)
    client.close()

    # Outliers the workers cannot hold (the 2.9 GB r2 1lep_1FJ WJets HT-2500 file is killed at
    # the worker memory limit) run here instead, where memory is plentiful.
    failed = []
    for f in cluster_failed:
        print(f"[postprocess] running {f} locally after it failed on the cluster", flush=True)
        try:
            rows = process_to_parquet([f], mirror_output_path(f, args.output_dir), args.tree,
                                      args.channel, args.nominal_only, want_cf)
        except Exception as exc:
            print(f"[postprocess] FAILED {f}: {type(exc).__name__}: {exc}", flush=True)
            failed.append(f)
            continue
        on_rows(f, rows)
    return failed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", required=True, choices=sorted(RECONSTRUCTORS),
                   help="reconstructed channel to process")
    p.add_argument("--input", required=True, nargs="+",
                   help="preselection-output ROOT file(s) or glob(s) for this channel")
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--output", help="single output parquet path (all --input read as one)")
    out.add_argument("--output-dir", help="mirror each input file to <output-dir>/<jobgroup>/"
                     "<sample>/<chunk>.parquet, so split samples stay grouped by sample dir")
    p.add_argument("--tree", default="Events", help="input TTree name (default: Events)")
    p.add_argument("--threads", type=int, default=1, help="ROOT implicit-MT threads (default: 1)")
    p.add_argument("--nominal-only", action="store_true", help="process only the nominal variation")
    p.add_argument("--skip-existing", action="store_true",
                   help="in --output-dir mode, skip inputs whose parquet already exists")
    p.add_argument("--cutflow", help="write a combined preselection+post-processor cutflow "
                   "table (this channel) to this .txt path")
    p.add_argument("--cutflow-logs", default=None,
                   help="preselection condor jobs dir with the .stdout logs (default: "
                        "$VBSVVH_PRESEL_DIR/condor/jobs, else the sibling checkout at "
                        f"{DEFAULT_PRESEL_DIR}). Only jobs submitted with the "
                        "preselection's --cutflow flag print a cutflow table; without it "
                        "only the post-processor stage is written.")
    p.add_argument("--cutflow-split", default=None,
                   help="regex matched against each input path to split the cutflow into one "
                        f"table per group (e.g. '{SIGNAL_POINT_RE}' for per-signal-point "
                        "cutflows). The matched text (group 1 if present) is the group key and "
                        "is appended to the --cutflow filename. Mirror mode only.")
    p.add_argument("--files-per-process", type=int, default=50,
                   help="mirror mode: process the inputs in batches of this many files, each in a "
                        "fresh child process, retrying a batch that crashes (default: 50; 0 = all "
                        "in this process). A long single process eventually segfaults inside "
                        "awkward's from_rdataframe (a cppyy std::string it caches as a template "
                        "argument gets freed), so bound the number of files per process.")
    p.add_argument("--scheduler", default=None,
                   help="mirror mode: run each input file as a task on this Dask scheduler "
                        "(the address dask_cluster.py writes) instead of locally. The driver must "
                        "run in the same LCG view as the workers; --threads and "
                        "--files-per-process do not apply.")
    p.add_argument("--retries", type=int, default=2,
                   help="dask mode: rerun a file whose task raised up to this many times (default: 2)")
    p.add_argument("--cutflow-rows-out", help=argparse.SUPPRESS)  # child mode: stream rows here
    args = p.parse_args(argv)

    if args.threads > 1:
        ROOT.EnableImplicitMT(args.threads)
    load_helpers()

    input_files = [f for pat in args.input for f in (sorted(glob.glob(pat)) or [pat])]
    if not input_files:
        p.error(f"no input files matched: {args.input}")

    want_cf = bool(args.cutflow) or bool(args.cutflow_rows_out)
    split_re = re.compile(args.cutflow_split) if args.cutflow_split else None
    if split_re is not None and args.output:
        p.error("--cutflow-split requires --output-dir (per-file processing), not --output")
    if args.scheduler and args.output:
        p.error("--scheduler requires --output-dir (per-file processing), not --output")
    cf_acc = {}  # group key -> {cut label -> summed Sum(w)} (insertion-ordered)

    def _group_key(path):
        if split_re is None:
            return "all"
        m = split_re.search(str(path))
        if not m:
            return "other"
        return m.group(1) if m.groups() else m.group(0)

    counted = set()  # input files whose rows are in cf_acc (skipped files contribute none)

    def _accumulate(group, rows, f=None):
        if f is not None:
            counted.add(f)
        d = cf_acc.setdefault(group, {})
        for label, sum_w in (rows or []):
            d[label] = d.get(label, 0.0) + sum_w

    failed = []
    if args.output:
        _accumulate("all", process_to_parquet(input_files, args.output, args.tree, args.channel,
                                              args.nominal_only, want_cf))
        counted.update(input_files)
    elif args.scheduler:
        failed = _run_dask(args, input_files, want_cf,
                           lambda f, rows: _accumulate(_group_key(f), rows, f))
    elif 0 < args.files_per_process < len(input_files):
        failed = _run_batches(args, input_files, want_cf,
                              lambda f, rows: _accumulate(_group_key(f), rows, f))
    else:
        print(f"[postprocess] mirror mode: {len(input_files)} input file(s) -> {args.output_dir}")
        for i, f in enumerate(input_files, 1):
            out_path = mirror_output_path(f, args.output_dir)
            if args.skip_existing and os.path.exists(out_path):
                print(f"[{i}/{len(input_files)}] skip existing {out_path}")
                continue
            print(f"[{i}/{len(input_files)}] {f}")
            rows = process_to_parquet([f], out_path, args.tree, args.channel,
                                      args.nominal_only, want_cf)
            _accumulate(_group_key(f), rows, f)
            if args.cutflow_rows_out:
                with open(args.cutflow_rows_out, "a") as fh:
                    fh.write(json.dumps({"file": f, "rows": rows}) + "\n")

    if args.cutflow:
        _emit_cutflow(args, input_files, cf_acc, n_uncounted=len(set(input_files) - counted))

    if failed:
        print(f"[postprocess] ERROR: {len(failed)} input file(s) produced no output:")
        for f in failed:
            print(f"    {f}")
        return 1
    return 0


MAX_BATCH_ATTEMPTS = 3


def _run_batches(args, input_files, want_cf, on_rows):
    n = args.files_per_process
    batches = [input_files[i:i + n] for i in range(0, len(input_files), n)]
    print(f"[postprocess] mirror mode: {len(input_files)} input file(s) -> {args.output_dir}  "
          f"in {len(batches)} batch(es) of <= {n} file(s), one process each", flush=True)

    def _child(files, rows_file, skip_existing):
        cmd = [sys.executable, os.path.abspath(__file__), "--channel", args.channel,
               "--tree", args.tree, "--threads", str(args.threads),
               "--output-dir", args.output_dir, "--files-per-process", "0", "--input", *files]
        if args.nominal_only:
            cmd.append("--nominal-only")
        if skip_existing:
            cmd.append("--skip-existing")
        if want_cf:
            cmd += ["--cutflow-rows-out", rows_file]
        return subprocess.run(cmd).returncode

    def _missing(files):
        return [f for f in files if not os.path.exists(mirror_output_path(f, args.output_dir))]

    failed = []
    with tempfile.TemporaryDirectory(prefix="postprocess_rows_") as tmp:
        for b, batch in enumerate(batches, 1):
            rows_file = os.path.join(tmp, f"batch{b}.jsonl")
            for attempt in range(MAX_BATCH_ATTEMPTS):
                print(f"[postprocess] batch {b}/{len(batches)}: {len(batch)} file(s)"
                      + (f"  (attempt {attempt + 1})" if attempt else ""), flush=True)
                rc = _child(batch, rows_file, args.skip_existing or attempt > 0)
                if rc == 0:
                    break
                print(f"[postprocess] WARNING: batch {b} exited with code {rc}; "
                      f"{len(_missing(batch))} file(s) of it still without output", flush=True)
            else:
                for f in _missing(batch):
                    print(f"[postprocess] retrying on its own: {f}", flush=True)
                    _child([f], rows_file, True)
            if want_cf and os.path.exists(rows_file):
                with open(rows_file) as fh:
                    for line in fh:
                        rec = json.loads(line)
                        on_rows(rec["file"], rec["rows"])
            failed += _missing(batch)
    return failed


def _emit_cutflow(args, input_files, cf_acc, n_uncounted=0):
    jobgroups = sorted({Path(f).parent.parent.name for f in input_files})
    logs_dir = Path(args.cutflow_logs) if args.cutflow_logs else \
        Path(os.environ.get("VBSVVH_PRESEL_DIR", DEFAULT_PRESEL_DIR)) / "condor" / "jobs"
    all_logs = [str(p) for jg in jobgroups for p in (logs_dir / jg).rglob("*.stdout")]
    out = Path(args.cutflow)

    for group, rows_dict in sorted(cf_acc.items()):
        logs = all_logs if group == "all" else [lf for lf in all_logs if group in lf]
        out_path = out if group == "all" else out.with_name(f"{out.stem}_{group}{out.suffix}")
        presel_rows, n_logs = parse_preselection_logs(logs)
        if not presel_rows:
            print(f"[cutflow] WARNING: no preselection cutflow found under {logs_dir} for "
                  f"group '{group}'; writing the post-processor stage only")
        grp = "" if group == "all" else f"  group={group}"
        title = f"Cutflow  channel={args.channel}{grp}  jobgroup(s)={', '.join(jobgroups)}"
        if n_uncounted:
            title += (f"\nINCOMPLETE: {n_uncounted} of {len(input_files)} input file(s) not counted "
                      f"(skipped as already processed, or failed); rerun without --skip-existing "
                      f"for full post-processor yields")
        write_combined_table(presel_rows, list(rows_dict.items()), str(out_path), title, n_logs)
        print(f"[cutflow] wrote combined cutflow -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
