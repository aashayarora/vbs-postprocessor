#!/usr/bin/env python3
"""Candidate post-processor for the VBS VVH analysis.

Reads a preselection-output ntuple for one reconstructed channel (0lep_3FJ / 1lep_1FJ /
1lep_2FJ), and for the nominal plus every JES/JER variation stored in the file, runs the
boson-candidate + resolved + m_lb reconstruction (ported from the old 1lep-cutbased
selections) using that variation's good-object collections and the preselection's stored
VBS pair. Candidates for all variations are written to a single long-format parquet file
with a ``variation`` column.

Usage:
    python postprocess.py --channel 1lep_2FJ \
        --input /path/to/presel_1lep_2FJ.root [more.root ...] \
        --output out/1lep_2FJ.parquet [--tree Events] [--threads 8] [--nominal-only]

Requires: ROOT (PyROOT), awkward, numpy. Run inside the project pixi env, e.g.
    CONDA_OVERRIDE_CUDA=12.0 pixi run --manifest-path env/pixi.toml python postprocess/postprocess.py ...
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import awkward as ak
import ROOT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import RECONSTRUCTORS, CHANNEL_GATE  # noqa: E402
from objects import build_collections, add_weight_columns  # noqa: E402
from cutflow import Cutflow, parse_preselection_logs, write_combined_table  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
# Event-identifier / bookkeeping columns copied through when present (weights are added
# separately by add_weight_columns). `shortname` lets downstream (plotter, abcd) split by
# sample; `year` is handy for lumi.
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
                  f"Jet_isGoodNoFJClean_{sfx}", f"Jet_pt_{sfx}", f"vbs_jet1_Jetidx_{sfx}"]
        if all(n in cols for n in needed):
            out.append(sfx)
    return out


def process_variation(input_files, tree, channel, sfx, passthrough, cutflow=None):
    """Reconstruct one variation; return an awkward record array with a variation field.

    ``cutflow`` (only passed for the nominal variation) books weighted checkpoints at the
    channel gate and each reconstruction cut, filled when the event loop runs below."""
    df = ROOT.RDataFrame(tree, input_files)
    df = build_collections(df, sfx)

    # Select this variation's event set with the preselection's own per-variation channel
    # flag (the file is the OR of all variations). This is authoritative — never recompute
    # the jet-count gate from a counter. Fall back to the recomputed gate only if the flag
    # column is absent (input predating the flags).
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
    arr = ak.from_rdataframe(df, columns=tuple(out_cols))
    label = sfx if sfx else "nominal"
    arr = ak.with_field(arr, ak.Array([label] * len(arr)), "variation")
    return arr


def mirror_output_path(input_file, output_dir):
    """Parquet path mirroring the preselection layout <jobgroup>/<sample>/<chunk>.root
    -> <output_dir>/<jobgroup>/<sample>/<chunk>.parquet. Preserving the <sample>/ dir
    keeps each big sample's chunks together, so the abcd/plotter readers group them as
    one sample (they derive the sample from the parent directory)."""
    p = Path(input_file)
    parts = [name for name in (p.parent.parent.name, p.parent.name) if name]
    return str(Path(output_dir, *parts, p.stem + ".parquet"))


def process_to_parquet(input_files, output_path, tree, channel, nominal_only, want_cutflow=False):
    """Reconstruct every variation from input_files and write one long-format parquet.

    Returns the nominal post-processor cutflow as [(label, sum_w)] when want_cutflow, else
    None. The checkpoints are booked on the nominal graph and filled by its event loop.
    """
    all_cols = [str(c) for c in ROOT.RDataFrame(tree, input_files).GetColumnNames()]
    variations = [""] if nominal_only else detect_variations(all_cols)
    print(f"[postprocess] channel={channel}  files={len(input_files)}  -> {output_path}  "
          f"variations={[v or 'nominal' for v in variations]}")

    cf = Cutflow() if want_cutflow else None
    pieces = []
    for sfx in variations:
        arr = process_variation(input_files, tree, channel, sfx, PASSTHROUGH,
                                cutflow=(cf if sfx == "" else None))
        print(f"  - {sfx or 'nominal':<24} {len(arr):>10} events")
        pieces.append(arr)

    combined = ak.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    ak.to_parquet(combined, output_path)
    print(f"[postprocess] wrote {len(combined)} rows -> {output_path}")
    return cf.rows() if cf is not None else None


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
                   help="preselection condor jobs dir with the .stdout logs "
                        "(default: <repo>/preselection/condor/jobs)")
    p.add_argument("--cutflow-split", default=None,
                   help="regex matched against each input path to split the cutflow into one "
                        "table per group (e.g. 'C2V_[^_/]+_C3_[^_/]+' for per-signal-point "
                        "cutflows). The matched text (group 1 if present) is the group key and "
                        "is appended to the --cutflow filename. Mirror mode only.")
    args = p.parse_args(argv)

    if args.threads > 1:
        ROOT.EnableImplicitMT(args.threads)
    load_helpers()

    input_files = [f for pat in args.input for f in (sorted(glob.glob(pat)) or [pat])]
    if not input_files:
        p.error(f"no input files matched: {args.input}")

    want_cf = bool(args.cutflow)
    split_re = re.compile(args.cutflow_split) if args.cutflow_split else None
    if split_re is not None and args.output:
        p.error("--cutflow-split requires --output-dir (per-file processing), not --output")
    cf_acc = {}  # group key -> {cut label -> summed Sum(w)} (insertion-ordered)

    def _group_key(path):
        if split_re is None:
            return "all"
        m = split_re.search(str(path))
        if not m:
            return "other"
        return m.group(1) if m.groups() else m.group(0)

    def _accumulate(group, rows):
        d = cf_acc.setdefault(group, {})
        for label, sum_w in (rows or []):
            d[label] = d.get(label, 0.0) + sum_w

    if args.output:
        # Single-output mode: read every input file as one frame -> one parquet.
        _accumulate("all", process_to_parquet(input_files, args.output, args.tree, args.channel,
                                              args.nominal_only, want_cf))
    else:
        # Mirror mode: one parquet per input chunk, under the mirrored <jobgroup>/<sample>/ tree.
        print(f"[postprocess] mirror mode: {len(input_files)} input file(s) -> {args.output_dir}")
        for i, f in enumerate(input_files, 1):
            out_path = mirror_output_path(f, args.output_dir)
            if args.skip_existing and os.path.exists(out_path):
                print(f"[{i}/{len(input_files)}] skip existing {out_path}")
                continue
            print(f"[{i}/{len(input_files)}] {f}")
            _accumulate(_group_key(f), process_to_parquet([f], out_path, args.tree, args.channel,
                                                          args.nominal_only, want_cf))

    if want_cf:
        _emit_cutflow(args, input_files, cf_acc)


def _emit_cutflow(args, input_files, cf_acc):
    """Aggregate the preselection cutflow from the condor logs of the processed job-group(s)
    and append the accumulated post-processor cutflow, writing one combined table per group
    (one overall table when --cutflow-split is not used)."""
    jobgroups = sorted({Path(f).parent.parent.name for f in input_files})
    logs_dir = Path(args.cutflow_logs) if args.cutflow_logs else \
        Path(__file__).resolve().parent.parent / "preselection" / "condor" / "jobs"
    all_logs = [str(p) for jg in jobgroups for p in (logs_dir / jg).rglob("*.stdout")]
    out = Path(args.cutflow)

    for group, rows_dict in sorted(cf_acc.items()):
        # Preselection logs for this group: the whole job-group unless splitting, in which case
        # keep only logs whose path carries the group key (the sample dir names match).
        logs = all_logs if group == "all" else [lf for lf in all_logs if group in lf]
        out_path = out if group == "all" else out.with_name(f"{out.stem}_{group}{out.suffix}")
        presel_rows, n_logs = parse_preselection_logs(logs)
        if not presel_rows:
            print(f"[cutflow] WARNING: no preselection cutflow found under {logs_dir} for "
                  f"group '{group}'; writing the post-processor stage only")
        grp = "" if group == "all" else f"  group={group}"
        title = f"Cutflow  channel={args.channel}{grp}  jobgroup(s)={', '.join(jobgroups)}"
        write_combined_table(presel_rows, list(rows_dict.items()), str(out_path), title, n_logs)
        print(f"[cutflow] wrote combined cutflow -> {out_path}")


if __name__ == "__main__":
    main()
