#!/bin/bash
#
# Fine C2V scan: one real combine limit at every point of a grid.
#
# The workspace carries C2V as a continuous parameter (c2v_model.py), so each
# grid point is a genuine AsymptoticLimits evaluation -- nothing is interpolated
# between the simulated couplings.
#
# Build the inputs first:
#   python3 c2vquad.py                       # cards from the three sample nodes
#   combineCards.py ... > datacard_combined.dat
#   text2workspace.py ... -P c2v_model:c2vQuadratic -o workspace.root
#
# Usage (source cmsenv first):
#   ./c2v_scan.sh [workspace] [min] [max] [step]
#
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=${1:-$HERE/c2vscan/quad/workspace.root}
MIN=${2:-0.25}
MAX=${3:-2.00}
STEP=${4:-0.05}

DIR=$(dirname "$WS")
CARD=$DIR/datacard_combined.dat
OUT=$DIR/limits_grid.txt
: > "$OUT"

# Empty control regions leave rateParams at 0, which makes the initial NLL
# infinite wherever the matching region A holds an event.
SETP=$(grep rateParam "$CARD" | grep -vE '\(@0' \
       | awk '$5==0{printf "%s=1,", $1}' | sed 's/,$//')

cd "$DIR"
for x in $(seq "$MIN" "$STEP" "$MAX"); do
    args=(-M AsymptoticLimits "$WS" -m 125 --cminDefaultMinimizerStrategy 0
          -n ".c2v$x" --setParameters "C2V=$x${SETP:+,$SETP}"
          --freezeParameters C2V)
    combine "${args[@]}" 2>&1 \
        | awk -v x="$x" '
            /Observed Limit/ {o=$NF}
            /Expected  2.5%/ {a=$NF} /Expected 16.0%/ {b=$NF}
            /Expected 50.0%/ {c=$NF} /Expected 84.0%/ {d=$NF} /Expected 97.5%/ {e=$NF}
            END {if (o != "") printf "%s %s %s %s %s %s %s\n", x, o, a, b, c, d, e}' \
        | tee -a "$OUT"
done

echo
echo "wrote $OUT ($(wc -l < "$OUT") grid points)"
