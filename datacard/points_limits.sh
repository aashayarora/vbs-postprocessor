#!/bin/bash
#
# Blind expected limits per channel, era and coupling point from the per-point cards:
# merge each region's scans (combine_points.sh), combine Run 2 + Run 3 per channel, then
# AsymptoticLimits. Writes points/limits/limits.json:
#   {"<channel>": {"<r2|r3|r2r3>": {"<sm|c2v|c3>": [-2s, -1s, median, +1s, +2s]}}}
#
# Usage: ./points_limits.sh      (sets up the CMSSW_16_0_0 combine environment itself)
set -eo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$HERE/CMSSW_16_0_0/src" && eval "$(scramv1 runtime -sh)" && cd "$HERE"

./combine_points.sh "$HERE/points/combined"

OUT=$HERE/points/limits
mkdir -p "$OUT"
cd "$OUT"
for ch in aa0l3fj aa1l1fj aa1l2fj; do
    for pt in sm c2v c3; do
        r2=$HERE/points/combined/dc_${ch}_r2_${pt}.txt
        r3=$HERE/points/combined/dc_${ch}_r3_${pt}.txt
        combineCards.py "$r2" "$r3" > dc_${ch}_r2r3_${pt}.txt
        for era in r2 r3 r2r3; do
            card=$([ $era = r2r3 ] && echo dc_${ch}_r2r3_${pt}.txt || echo "$HERE/points/combined/dc_${ch}_${era}_${pt}.txt")
            combine -M AsymptoticLimits "$card" --run blind -n _${ch}_${era}_${pt} > limit_${ch}_${era}_${pt}.log 2>&1 &
        done
    done
done
wait

python3 - <<'EOF'
import glob, json, re
out = {}
for log in sorted(glob.glob("limit_*.log")):
    ch, era, pt = re.match(r"limit_(aa\w+?)_(r2r3|r2|r3)_(\w+)\.log", log).groups()
    vals = dict(re.findall(r"Expected\s+([\d.]+)%: r < ([\d.eE+-]+)", open(log).read()))
    keys = ["2.5", "16.0", "50.0", "84.0", "97.5"]
    out.setdefault(ch, {}).setdefault(era, {})[pt] = [float(vals[k]) if k in vals else None for k in keys]
json.dump(out, open("limits.json", "w"), indent=1)
for ch in out:
    for era in ("r2", "r3", "r2r3"):
        print(ch, era, {pt: out[ch][era][pt][2] for pt in out[ch][era]})
EOF
