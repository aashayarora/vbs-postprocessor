#!/bin/bash
#
# CMS unblinding step 3 -- observed limits, post-fit expected limits, the
# best-fit signal strength and the observed significance.
#
# Runs against cards already built and combined:
#     ./make_datacards.sh --unblind
#     cd unblind && ../combine_cards.sh all
#
# AsymptoticLimits is run *without* --run blind, so the expected limits it
# reports are the a-posteriori (post-fit) ones -- combine fits the background to
# the observed data before building the Asimov. The a-priori (pre-fit) expected
# limit that combine_cards.sh prints is shown alongside for comparison.
#
# There is a single POI, so no simultaneous multi-signal-strength measurement and
# hence no SM-compatibility p-value; the observed significance covers it.
#
# Usage:
#   ./unblind_step3.sh [-d <dir>]        (source cmsenv first)
#
set -euo pipefail

DIR=unblind
[[ ${1:-} == -d ]] && { DIR=$2; shift 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/$DIR"

CHANNELS=(0lep_3fj_r2 0lep_3fj_r3 1lep_1fj_r2 1lep_1fj_r3 1lep_2fj_r2 1lep_2fj_r3)

# Empty control regions leave their rateParams at 0, which makes the initial NLL
# infinite wherever the matching region A holds an event. Nudge the starting
# value off the singular point; the fitted ranges are untouched.
nudge_for() {
    grep "rateParam" "$1" | grep -vE '\(@0' \
        | awk '$5==0{printf "%s=1,", $1}' | sed 's/,$//'
}

# limits_for <card> <workspace> <tag> -> "observed exp2.5 exp16 exp50 exp84 exp97.5"
limits_for() {
    local card=$1 ws=$2 tag=$3
    local setp; setp=$(nudge_for "$card")
    local args=(-M AsymptoticLimits "$ws" -m 125 --cminDefaultMinimizerStrategy 0 -n ".$tag")
    [[ -n $setp ]] && args+=(--setParameters "$setp")
    combine "${args[@]}" 2>&1 | awk '
        /Observed Limit/ {o=$NF}
        /Expected  2.5%/ {a=$NF} /Expected 16.0%/ {b=$NF}
        /Expected 50.0%/ {c=$NF} /Expected 84.0%/ {d=$NF} /Expected 97.5%/ {e=$NF}
        END {print o, a, b, c, d, e}'
}

TABLE=limits_step3.txt
: > "$TABLE"

echo "=== per-channel and combined limits ==="
printf "%-14s %10s %10s %10s %10s %10s %10s\n" \
    channel observed "exp -2s" "exp -1s" "exp med" "exp +1s" "exp +2s" | tee -a "$TABLE"

for ch in "${CHANNELS[@]}" combined; do
    card=datacard_$ch.dat
    ws=datacard_${ch}_workspace.root
    if [[ $ch != combined ]]; then
        # One channel on its own: r1s1..r1s5.
        cards=""
        for f in "$ch"/datacard_scan_Scan*.dat; do
            n=$(basename "$f" .dat)
            cards+="r1s${n#datacard_scan_Scan}=$f "
        done
        combineCards.py $cards > "$card" 2>/dev/null
    fi
    [[ -f $ws && $ws -nt $card ]] || text2workspace.py "$card" -o "$ws" >/dev/null 2>&1
    # shellcheck disable=SC2046
    printf "%-14s %10s %10s %10s %10s %10s %10s\n" "$ch" $(limits_for "$card" "$ws" "$ch") \
        | tee -a "$TABLE"
done

echo
echo "=== combined: measurement and significance ==="
CARD=datacard_combined.dat
WS=datacard_combined_workspace.root
SETP=$(nudge_for "$CARD")

combine -M MultiDimFit "$WS" --algo singles --redefineSignalPOIs r -m 125 \
    --setParameters "$SETP" --cminDefaultMinimizerStrategy 0 --robustFit 1 \
    --rMin=0 --rMax=10 -n .bestfit 2>&1 | grep -E "^ *r :" || true

combine -M Significance "$WS" -m 125 --setParameters "$SETP" \
    --cminDefaultMinimizerStrategy 0 -n .signif 2>&1 | grep -i "^Significance" || true

echo
echo "Done."
