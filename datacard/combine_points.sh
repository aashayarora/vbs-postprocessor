#!/bin/bash
#
# Merge each region's per-scan cards into one card per region and signal point:
#
#   points/<point>/<N>lep_<M>fj_<run>/datacard_scan_Scan<k>.dat  ->  <out>/dc_aa<N>l<M>fj_<run>_<sm|c2v|c3>.txt
#
# (points: sm = C2V 1.0 / C3 1, c2v = C2V 1.5 / C3 1, c3 = C2V 1.0 / C3 10.) The scans are
# orthogonal by construction (each ABCD scan drops the previous scans' events), so the
# merged card is a valid combination. Bin labels are aa<N>l<M>fj<run>s<k>, unique per region
# and scan so the merged cards can themselves be combined, and short (see combine_cards.sh).
#
# Usage (source cmsenv first, for combineCards.py):
#   ./combine_points.sh [output-dir]      # default: points/combined
#
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-$HERE/points/combined}
mkdir -p "$OUT"

declare -A SUFFIX=([sm]=sm [c2v1p5]=c2v [c3_10]=c3)

for pt in sm c2v1p5 c3_10; do
    for dir in "$HERE"/points/$pt/*/; do
        region=$(basename "$dir")
        [[ $region =~ ^([0-9])lep_([0-9])fj_(r[23])$ ]] || continue
        stem="aa${BASH_REMATCH[1]}l${BASH_REMATCH[2]}fj"
        run=${BASH_REMATCH[3]}
        args=()
        for f in "$dir"datacard_scan_Scan*.dat; do
            k=$(basename "$f" .dat); k=${k#datacard_scan_Scan}
            args+=("${stem}${run}s${k}=$f")
        done
        card=$OUT/dc_${stem}_${run}_${SUFFIX[$pt]}.txt
        combineCards.py "${args[@]}" > "$card"
        echo "$(basename "$card"): ${#args[@]} scans"
    done
done
