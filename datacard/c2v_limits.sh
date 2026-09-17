#!/bin/bash
#
# C2V scan: datacards and limits at each anomalous-coupling point.
#
# The signal regions are the ones optimised on C2V=1.5 and are NOT re-derived --
# only the signal template changes from point to point. Run 3 channels only,
# since Run 2 does not have samples at every C2V point.
#
# Expects the per-point predictions produced by scoring each signal sample
# through the trained model (abcd/main.py --infer --no-plots).
#
# Usage (source cmsenv first):
#   ./c2v_limits.sh <predictions-dir> [output-dir]
#
# <predictions-dir> holds <CHANNEL>_c2v<pt>.parquet; C2V=1p5 is taken from the
# existing abcd output instead, since it was already scored.
#
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
PRED=${1:?usage: c2v_limits.sh <predictions-dir> [output-dir]}
OUT=${2:-$HERE/c2vscan}
ABCD=$REPO/abcd/output

POINTS=(0p25 0p75 1p0 1p5 2p0)
# <abcd output dir>:<process/card name>:<extra datacard.py flags>
CHANNELS=(
    "0LEP_3FJ_RUN3:0lep_3fj_r3:"
    "1LEP_1FJ_RUN3:1lep_1fj_r3:--combination or"
    "1LEP_2FJ_RUN3:1lep_2fj_r3:"
)

latest_version() { ls -d "$ABCD/$1/single/version_"* | sort -t_ -k2 -n | tail -1; }

mkdir -p "$OUT"
: > "$OUT/limits_c2v.txt"

for pt in "${POINTS[@]}"; do
    echo "=== C2V = $pt ==="
    dir=$OUT/$pt
    mkdir -p "$dir"

    for row in "${CHANNELS[@]}"; do
        IFS=: read -r chan proc extra <<<"$row"
        v=$(latest_version "$chan")
        data=$(ls "$v"/predictions_single*_data.parquet | tail -1)
        if [[ $pt == 1p5 ]]; then
            # Already scored: the nominal predictions are the C2V=1.5 template.
            sig=$(ls "$v"/predictions_single*.parquet | grep -v '_data\.parquet$' | tail -1)
        else
            sig=$PRED/${chan}_c2v${pt}.parquet
        fi
        [[ -f $sig ]] || { echo "ERROR: missing $sig" >&2; exit 1; }
        # shellcheck disable=SC2086
        python3 "$HERE/datacard.py" --sig "$sig" --data "$data" \
            --out "$dir/$proc/datacard_scan" --config "$v/regions.yaml" \
            --proc "$proc" --unblind $extra > "$dir/${proc}_cards.log" 2>&1
    done

    # Combine the three Run 3 channels. Short bin labels for the same RooFit
    # factory-string reason as combine_cards.sh.
    cards=""; ri=0
    for row in "${CHANNELS[@]}"; do
        IFS=: read -r _ proc _ <<<"$row"
        ri=$((ri + 1))
        for f in "$dir/$proc"/datacard_scan_Scan*.dat; do
            n=$(basename "$f" .dat)
            cards+="r${ri}s${n#datacard_scan_Scan}=$f "
        done
    done
    ( cd "$dir" && combineCards.py $cards > datacard_combined.dat \
        && text2workspace.py datacard_combined.dat -o workspace.root >/dev/null 2>&1 )

    # Empty control regions leave rateParams at 0, which makes the initial NLL
    # infinite wherever the matching region A holds an event.
    setp=$(grep rateParam "$dir/datacard_combined.dat" | grep -vE '\(@0' \
           | awk '$5==0{printf "%s=1,", $1}' | sed 's/,$//')
    args=(-M AsymptoticLimits "$dir/workspace.root" -m 125
          --cminDefaultMinimizerStrategy 0 -n ".c2v$pt")
    [[ -n $setp ]] && args+=(--setParameters "$setp")

    ( cd "$dir" && combine "${args[@]}" 2>&1 ) | tee "$dir/limit.log" \
        | awk -v pt="$pt" '
            /Observed Limit/ {o=$NF}
            /Expected  2.5%/ {a=$NF} /Expected 16.0%/ {b=$NF}
            /Expected 50.0%/ {c=$NF} /Expected 84.0%/ {d=$NF} /Expected 97.5%/ {e=$NF}
            END {printf "%-6s %10s %10s %10s %10s %10s %10s\n", pt, o, a, b, c, d, e}' \
        | tee -a "$OUT/limits_c2v.txt"
done

echo
echo "mu limits per C2V point -> $OUT/limits_c2v.txt"
