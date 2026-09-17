"""Combined cutflow across the two analysis stages.

The event selection is now split: the preselection (C++/condor) does everything up to the
channel jet-selection, printing a weighted cutflow table to each job's ``.stdout``; the
post-processor then applies the boson-candidate / resolved / VBS assignment cuts. This
module aggregates the preselection cutflow from the condor logs and appends the
post-processor's own cuts, writing one combined table per channel to a text file.

Preselection weighting (see ``preselection/src/main.cpp``): ``Sum(weight)`` for MC,
``Count`` for data. The post-processor mirrors it: ``Sum(weight)`` when a ``weight`` column
is present, else event counts.
"""
from __future__ import annotations

import gzip
import re
from collections import OrderedDict


# Matches the tabulate rows the preselection prints:
# | # | Cut name | Sum(w) | +/- stat | rel. eff. | abs. eff. |
_ROW = re.compile(r'\|\s+(\d+)\s+\|\s+(.+?)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|')


class Cutflow:
    """Books a lazy weighted yield (or count) at each named checkpoint on an RDF graph.

    Call :meth:`add` after each cut while building the graph; read :meth:`rows` after the
    event loop has run (e.g. after ``ak.from_rdataframe`` triggers it — the checkpoints
    share the loop manager, so they are filled in the same pass).
    """

    def __init__(self, weight_col="weight"):
        self.weight_col = weight_col
        self._entries = []  # (label, RResultPtr)

    def add(self, df, label):
        if self.weight_col and self.weight_col in set(str(c) for c in df.GetColumnNames()):
            self._entries.append((label, df.Sum(self.weight_col)))
        else:
            self._entries.append((label, df.Count()))

    def rows(self):
        """[(label, value)] after the loop has run."""
        return [(label, float(res.GetValue())) for label, res in self._entries]


def _read_text(path):
    with open(path, "rb") as raw:
        gzipped = raw.read(2) == b"\x1f\x8b"
    opener = (lambda p: gzip.open(p, "rt", errors="replace")) if gzipped else (lambda p: open(p, "r", errors="replace"))
    with opener(path) as f:
        return f.read()


def parse_preselection_logs(log_files):
    """Aggregate the preselection cutflow (summing Sum(w) per cut) across job stdout files.

    Returns ``[(cut_name, sum_w), ...]`` ordered by the preselection's cut number, and the
    number of log files that actually contained a cutflow table.
    """
    agg = {}  # cut_num -> [name, sum_w]
    n_used = 0
    for path in log_files:
        try:
            matches = _ROW.findall(_read_text(path))
        except OSError:
            continue
        if matches:
            n_used += 1
        for num, name, sum_w, _err, _rel, _abs in matches:
            num = int(num)
            entry = agg.setdefault(num, [name.strip(), 0.0])
            entry[1] += float(sum_w)
    rows = [(agg[k][0], agg[k][1]) for k in sorted(agg)]
    return rows, n_used


def write_combined_table(presel_rows, postproc_rows, out_path, title, n_logs):
    """Write a single combined cutflow table (preselection stage + post-processor stage).

    rel. eff. is versus the previous row; abs. eff. is versus the first row. A divider marks
    the hand-off between the two stages.
    """
    rows = ([("preselection", name, sw) for name, sw in presel_rows]
            + [("postprocess", name, sw) for name, sw in postproc_rows])
    lines = []
    lines.append("=" * 92)
    lines.append(title)
    lines.append(f"(preselection cutflow aggregated from {n_logs} condor stdout log(s))")
    lines.append("=" * 92)
    lines.append(f"{'#':<3} {'stage':<12} {'Cut name':<40} {'Sum(w)':>14} {'rel.eff':>8} {'abs.eff':>8}")
    lines.append("-" * 92)

    first = rows[0][2] if rows else 0.0
    prev = None
    prev_stage = None
    for i, (stage, name, sw) in enumerate(rows):
        if prev_stage == "preselection" and stage == "postprocess":
            lines.append("-" * 92 + "   << post-processor candidate assignment >>")
        rel = (sw / prev) if (prev and prev > 0) else 1.0
        ab = (sw / first) if first > 0 else 1.0
        lines.append(f"{i:<3} {stage:<12} {name:<40} {sw:>14.3f} {rel:>8.4f} {ab:>8.4f}")
        prev, prev_stage = sw, stage
    lines.append("=" * 92)

    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text
