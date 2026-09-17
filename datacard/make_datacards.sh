#!/bin/bash
#
# Build the per-scan ABCD datacards.
#
# Usage:
#   ./make_datacards.sh [--unblind]
#
#   --unblind   region A takes its real observation instead of being pinned to 1.
#               Cards go to unblind/<region>/ so the blind ones are never
#               overwritten. Combine them from there:
#                   cd unblind && ../combine_cards.sh all

BASE=/home/users/aaarora/phys/run3/cmstas-run3-vbsvvh/abcd/output/

OUTDIR=.
UNBLIND=()
if [[ ${1:-} == "--unblind" ]]; then
        OUTDIR=unblind
        UNBLIND=(--unblind)
        echo "Building UNBLINDED cards into $OUTDIR/"
        mkdir -p "$OUTDIR"
fi

# Return the highest version_N directory for a given channel, e.g.
#   latest_version 1LEP_2FJ_RUN3  ->  .../1LEP_2FJ_RUN3/single/version_5
latest_version() {
        local channel=$1
        ls -d "$BASE/$channel/single/version_"* 2>/dev/null \
                | sort -t_ -k2 -n \
                | tail -n1
}

# run_datacard <CHANNEL> <out> <proc> [extra datacard.py args...]
run_datacard() {
        local channel=$1 out=$2 proc=$3
        shift 3
        local dir
        dir=$(latest_version "$channel")
        if [[ -z "$dir" ]]; then
                echo "ERROR: no version_* directory found for $channel" >&2
                return 1
        fi
        local data sig
        data=$(ls "$dir"/predictions_single*_data.parquet 2>/dev/null | tail -n1)
        sig=$(ls "$dir"/predictions_single*.parquet 2>/dev/null | grep -v '_data\.parquet$' | tail -n1)
        if [[ -z "$sig" || -z "$data" ]]; then
                echo "ERROR: could not find prediction files in $dir for $channel" >&2
                return 1
        fi
        echo "for $proc, data=$data, sig=$sig"
        python3 datacard.py \
                --sig "$sig" \
                --data "$data" \
                --out "$out" \
                --config "$dir/regions.yaml" \
                --proc "$proc" \
                "$@"
}

run_datacard 1LEP_2FJ_RUN3 $OUTDIR/1lep_2fj_r3/datacard_scan 1lep_2fj_r3 "${UNBLIND[@]}" &
run_datacard 1LEP_2FJ_RUN2 $OUTDIR/1lep_2fj_r2/datacard_scan 1lep_2fj_r2 "${UNBLIND[@]}" &
run_datacard 1LEP_1FJ_RUN3 $OUTDIR/1lep_1fj_r3/datacard_scan 1lep_1fj_r3 --combination or "${UNBLIND[@]}" &
run_datacard 1LEP_1FJ_RUN2 $OUTDIR/1lep_1fj_r2/datacard_scan 1lep_1fj_r2 --combination or "${UNBLIND[@]}" &
run_datacard 0LEP_3FJ_RUN3 $OUTDIR/0lep_3fj_r3/datacard_scan 0lep_3fj_r3 "${UNBLIND[@]}" &
run_datacard 0LEP_3FJ_RUN2 $OUTDIR/0lep_3fj_r2/datacard_scan 0lep_3fj_r2 "${UNBLIND[@]}" &
