#!/bin/bash
#
# CMS unblinding step 1 -- impacts and goodness-of-fit.
#
# Runs against an already-combined datacard; it does not make datacards. Build
# and combine them first:
#
#     ./make_datacards.sh --unblind
#     cd unblind && ../combine_cards.sh all
#
# The signal strength is never plotted: plotImpacts.py is always called with
# --blind. AsymptoticLimits / Significance are not run here -- those are step 3.
#
# Usage:
#   ./unblind_step1.sh [-c <datacard.dat>] [step ...]
#
#   steps        impacts gof pulls          (default: impacts gof)
#   -c <path>    combined card to fit       (default: ./unblind/datacard_combined.dat)
#   ASIMOV=1     rank against an r=1 Asimov, not data
#   RMIN=<x>     lower bound on r                                    (default 0)
#   NPROC=<n>    parallel impact fits                                (default 16)
#   NTOYS=<n>    toys for the GOF p-value                            (default 500)
#   NJOBS=<n>    concurrent GOF toy jobs to split NTOYS over         (default 10)
#
set -euo pipefail


# ----------------------------------------------------------------- configuration

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
CARD=$HERE/unblind/datacard_combined.dat

if [[ ${1:-} == -c ]]; then
    CARD=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
    shift 2
fi

# Outputs land next to the card. combine_cards.sh stops at the .dat, so the
# workspace is built here (and rebuilt whenever the card is newer).
OUT=$(dirname "$CARD")
WS=${CARD%.dat}_workspace.root

[[ -f $CARD ]] || {
    echo "ERROR: no combined card at $CARD" >&2
    echo "       run: ./make_datacards.sh --unblind && cd unblind && ../combine_cards.sh all" >&2
    exit 1
}

# combine_cards.sh stops at the .dat, so build the workspace here; rebuild it
# whenever the card is newer.
if [[ ! -f $WS || $CARD -nt $WS ]]; then
    echo "building workspace: $WS"
    text2workspace.py "$CARD" -o "$WS"
fi

# ----------------------------------------------------------------------- helpers

# An empty control region in data leaves its rateParam at exactly 0, which zeroes
# that scan's ABCD prediction too. Where the matching region A still holds an
# event the initial NLL is infinite, and the minimiser never leaves the starting
# point. Nudging the *starting value* off the singular point clears this; the
# fitted ranges in the cards are untouched.
#
# Sets NUDGE to the combine flags, or an empty array if no control region is empty.
set_nudge() {
    local setp=""
    [[ -f $CARD ]] && setp=$(grep "rateParam" "$CARD" | grep -vE '\(@0' \
                             | awk '$5==0{printf "%s=1,", $1}' | sed 's/,$//')
    NUDGE=()
    [[ -n $setp ]] && NUDGE=(--setParameters "$setp")
    echo "  empty-CR rateParams nudged: ${setp:-(none)}"
}

# ------------------------------------------------------------------------- steps

# Region A is modelled as B*C/D + r*S, and for several scans S exceeds the ABCD
# prediction, so the likelihood is only defined down to r ~ -0.04; wider negative
# ranges crash the minimiser. The data prefer a small negative r, so at the
# default rMin=0 the POI sits on its boundary and every impact on data comes out
# identically zero. ASIMOV=1 reruns the same ranking against an r=1 Asimov, where
# the POI is interior and the ranking is meaningful.
step_impacts() {
    cd "$OUT"
    local tag=data
    local opts=(-m 125 --cminDefaultMinimizerStrategy 0 --robustFit 1
                --rMin="${RMIN:-0}" --rMax=10.0)

    set_nudge
    opts+=("${NUDGE[@]}")
    if [[ ${ASIMOV:-0} == 1 ]]; then
        opts+=(-t -1 --expectSignal=1)
        tag=asimov
    fi

    echo "  initial fit ($tag)"
    combineTool.py -M Impacts -d "$WS" "${opts[@]}" \
        --doInitialFit                      > "impacts_${tag}_initial.log" 2>&1

    echo "  per-nuisance fits"
    combineTool.py -M Impacts -d "$WS" "${opts[@]}" \
        --doFits --parallel "${NPROC:-16}"  > "impacts_${tag}_fits.log" 2>&1

    combineTool.py -M Impacts -d "$WS" "${opts[@]}" \
     -o "impacts_${tag}.json"               > "impacts_${tag}_json.log" 2>&1

    # --blind so the signal strength is never rendered; no --max-pages so every
    # nuisance is written, not just the leading page.
    plotImpacts.py -i "impacts_${tag}.json" -o "impacts_${tag}" --blind --summary
    echo "  wrote impacts_${tag}.pdf"
}

# Saturated-model GOF, observed plus a toy distribution for the p-value.
#
# This does not currently work on these cards and reports rather than aborts.
# combine builds the saturated pdf as a RooDataHist over the dataset's
# observables; for a pure counting card those are continuous RooRealVars
# (n_obs_bin*) spanning [0, inf) with 100 default bins, so the histogram is
# 100^N cells -- nan for a single card, a segfault for the combination. AD and
# KS are degenerate here for the same reason. Emitting the regions as 1-bin
# shape cards (a TH1 per region) would give a real binned observable and make
# this step work as written.
step_gof() {
    cd "$OUT"
    local ntoys=${NTOYS:-500}
    local njobs=${NJOBS:-10}
    local ok=1

    echo "  observed (saturated)"
    combine -M GoodnessOfFit "$WS" --algo saturated -n .obs -m 125 \
        --rMin=0 --rMax=10 > gof_obs.log 2>&1 || ok=0
    grep -qiE "segmentation violation|-nan|\*\*\* Break" gof_obs.log && ok=0

    if (( ! ok )); then
        echo "  WARNING: saturated GOF failed on these counting cards; see gof_obs.log"
        echo "           (needs 1-bin shape cards -- see the comment above step_gof)"
        return 0
    fi

    # The toys are independent, so NTOYS is split over NJOBS concurrent combine
    # processes, one seed each. combine appends the seed to the filename whenever
    # -s is given, so every job writes its own file and they are collected below.
    # Stale per-job files are cleared first, or a re-run with a smaller NJOBS
    # would fold trees from the previous run into this one's p-value.
    echo "  $ntoys toys over $njobs jobs"
    rm -f higgsCombine.toys_job*.GoodnessOfFit.mH125.*.root gof_toys_job*.log gof_toys.log

    local pids=() outs=() i n seed
    for (( i = 1; i <= njobs; i++ )); do
        # spread the remainder over the first (ntoys % njobs) jobs
        n=$(( ntoys / njobs + (i <= ntoys % njobs ? 1 : 0) ))
        (( n > 0 )) || continue
        seed=$(( 123456 + i ))

        combine -M GoodnessOfFit "$WS" --algo saturated -n ".toys_job${i}" -m 125 \
            -t "$n" -s "$seed" --toysFrequentist --rMin=0 --rMax=10 \
            > "gof_toys_job${i}.log" 2>&1 &

        pids+=("$!")
        outs+=("higgsCombine.toys_job${i}.GoodnessOfFit.mH125.${seed}.root")
    done

    local fail=0 p
    for p in "${pids[@]}"; do
        wait "$p" || fail=1
    done
    (( fail )) && echo "  WARNING: at least one toy job failed; see gof_toys_job*.log"

    # Collect only what actually landed, so a partial failure still yields a
    # usable (smaller) toy distribution rather than aborting the step.
    local found=() f
    for f in "${outs[@]}"; do
        if [[ -s $f ]]; then found+=("$f"); fi
    done

    if (( ${#found[@]} == 0 )); then
        echo "  ERROR: no toy files produced; see gof_toys_job*.log" >&2
        return 1
    fi

    # No hadd: CollectGoodnessOfFit takes the whole set of files, splits observed
    # from toys on iToy, and drops any file that is corrupt or truncated -- which
    # is what a killed toy job leaves behind. The mass key it writes is str(mh),
    # hence 125.0 rather than 125 in the plotGof call below.
    echo "  collecting (${#found[@]}/${#outs[@]} toy files)"
    combineTool.py -M CollectGoodnessOfFit \
        --input higgsCombine.obs.GoodnessOfFit.mH125.root "${found[@]}" \
        -o gof.json > gof_collect.log 2>&1

    plotGof.py gof.json \
        --statistic saturated \
        --mass 125.0 \
        -o gof_plot \
        --title-right="VBS Higgs" > gof_plot.log 2>&1

    # The p-value is the point of the exercise; plotGof draws it, but echo the
    # surviving toy count too since collection silently drops failed jobs.
    python3 -c "import json; j = json.load(open('gof.json'))['125.0']; print('  p-value = %.4f from %d toys' % (j['p'], len(j['toy'])))" || true
    echo "  wrote gof.json, gof_plot.pdf, gof_plot.png"
}

# Unblinded pulls and constraints from the fit to data. Unlike the impact
# ranking these are unaffected by the POI sitting on its boundary, so they are
# the substantive unblinded check in the step-1 package.
step_pulls() {
    cd "$OUT"
    set_nudge
    combine -M FitDiagnostics "$WS" -m 125 --rMin=0 --rMax=10 \
        --cminDefaultMinimizerStrategy 0 "${NUDGE[@]}" -n .data > fitdiag.log 2>&1

    # cmsenv's python, not pixi's: diffNuisances.py imports the CombinedLimit
    # package and ROOT, neither of which exists in the pixi env.
    python3 "$CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py" \
        fitDiagnostics.data.root --abs > pulls.txt 2>&1
    echo "  wrote pulls.txt"
}


# -------------------------------------------------------------------------- main

steps=("$@")
[[ ${#steps[@]} -eq 0 ]] && steps=(impacts gof)

echo "workspace: $WS"
for s in "${steps[@]}"; do
    declare -F "step_$s" >/dev/null \
        || { echo "ERROR: unknown step '$s' (have: impacts gof pulls)" >&2; exit 1; }
    echo "=== $s ==="
    "step_$s"
done

echo "Done. Outputs in $OUT"
