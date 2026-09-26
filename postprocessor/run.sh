#!/bin/bash
#
# Post-process every (channel, run, sample-kind) group of a preselection production.
#
# The preselection now writes merged job groups whose directory names carry only the
# channel, run and submit timestamp:
#
#   <PRESEL_DIR>/merged_<channel>_<r2|r3>_<timestamp>_<channel>/<sample>/output_*.root
#
# i.e. they no longer spell out sig / data / bkg the way the old
# *_r3_1lep_2fj_sig_* groups did, so the old globs cannot address them. Each merged
# group is homogeneous, though, so resolve_group below picks the right directory by
# looking at the sample names inside it:
#
#   sig   -> VBS<VVH>[_OS|_SS]_c2v<x>_c3_<y>_<campaign>   (e.g. VBSWZH_c2v1p5_c3_1p0_UL17)
#   data  -> <PD>_Run<era>..._NANOAOD                     (no SIM suffix)
#   bkg   -> everything else (the *_NANOAODSIM / RunIISummer20UL* MC)
#
# This keeps the script working across re-productions (the timestamps change every time)
# without hardcoding a directory list.

set -u

PRESEL_DIR=${PRESEL_DIR:-/ceph/cms/store/user/aaarora/vbsvvh/preselection/new}
OUTPUT_DIR=${OUTPUT_DIR:-/ceph/cms/store/user/aaarora/vbsvvh/postprocessing/latest/}
CUTFLOW_DIR=${CUTFLOW_DIR:-cutflows}  # e.g. CUTFLOW_DIR=cutflows/resume ./run.sh --skip-existing
mkdir -p ${CUTFLOW_DIR}

# Extra postprocess.py flags for every job, from this script's own arguments, e.g.
#   ./run.sh --skip-existing     # resume: only process preselection files with no parquet yet
# (the cutflow tables of a resumed run cover only the newly processed files, and say so).
EXTRA_ARGS=("$@")

# Per-signal-point cutflow split, matching the current signal sample naming
# (VBSWWH_OS_c2v1p0_c3_10p0_UL18). Keep in sync with SIGNAL_POINT_RE in postprocess.py.
SIGNAL_POINT_RE='VBS[A-Z]+H(?:_[OS]S)?_c2v[0-9p]+_c3_[0-9p]+'

# The preselection half of the combined cutflow is parsed from the condor .stdout logs of
# the matching job group. Only jobs submitted with the preselection's own --cutflow flag
# print a cutflow table; when they were not, the combined table degrades to the
# post-processor stage alone (postprocess.py warns and carries on).
export VBSVVH_PRESEL_DIR=${VBSVVH_PRESEL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/cmstas-run3-vbsvvh/preselection}

# resolve_group <channel> <r2|r3> <sig|data|bkg> -> the one merged_* dir for that group.
# Errors out unless exactly one directory matches, so a half-staged production fails loudly
# instead of silently post-processing nothing.
resolve_group() {
    local channel=$1 run=$2 kind=$3 dir hits=()
    for dir in "${PRESEL_DIR}"/merged_"${channel}"_"${run}"_*_"${channel}"; do
        [ -d "$dir" ] || continue
        local n_sig n_data n_tot
        n_tot=$(find "$dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
        [ "$n_tot" -gt 0 ] || continue
        n_sig=$(find "$dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -cE '^VBS.*_c2v')
        n_data=$(find "$dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -c '_NANOAOD$')
        case $kind in
            sig)  [ "$n_sig"  -gt 0 ] && hits+=("$dir") ;;
            data) [ "$n_data" -gt 0 ] && hits+=("$dir") ;;
            bkg)  [ "$n_sig" -eq 0 ] && [ "$n_data" -eq 0 ] && hits+=("$dir") ;;
        esac
    done
    if [ ${#hits[@]} -ne 1 ]; then
        echo "resolve_group: expected 1 dir for ${channel}/${run}/${kind}, found ${#hits[@]}: ${hits[*]-}" >&2
        return 1
    fi
    printf '%s\n' "${hits[0]}"
}

# submit <channel> <run> <kind> <threads> [extra postprocess.py args...]
submit() {
    local channel=$1 run=$2 kind=$3 threads=$4; shift 4
    local dir tag
    dir=$(resolve_group "$channel" "$run" "$kind") || return 1
    # Log / cutflow basename in the old r3_1lep_2fj_sig style, from the channel name.
    tag="${run}_$(echo "$channel" | tr '[:upper:]' '[:lower:]')_${kind}"
    echo "[run.sh] ${tag}  <-  $(basename "$dir")"
    nohup python3 postprocess.py \
        --channel "$channel" \
        --threads "$threads" \
        --input "$dir"/*/output_*.root \
        --output-dir "${OUTPUT_DIR}" \
        --cutflow "${CUTFLOW_DIR}/${tag}.txt" \
        "$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} >>"${tag}.log" 2>&1 &
}

# # sig
submit 1lep_2FJ r2 sig 16 --cutflow-split "${SIGNAL_POINT_RE}"
submit 1lep_2FJ r3 sig 16 --cutflow-split "${SIGNAL_POINT_RE}"

submit 1lep_1FJ r2 sig 16 --cutflow-split "${SIGNAL_POINT_RE}"
submit 1lep_1FJ r3 sig 16 --cutflow-split "${SIGNAL_POINT_RE}"

submit 0lep_3FJ r2 sig 16 --cutflow-split "${SIGNAL_POINT_RE}"
submit 0lep_3FJ r3 sig 16 --cutflow-split "${SIGNAL_POINT_RE}"


# # data
submit 1lep_2FJ r2 data 16
submit 1lep_2FJ r3 data 16

submit 1lep_1FJ r2 data 16
submit 1lep_1FJ r3 data 16

submit 0lep_3FJ r2 data 16
submit 0lep_3FJ r3 data 16


# # bkg
submit 1lep_2FJ r2 bkg 32
submit 1lep_2FJ r3 bkg 32

submit 1lep_1FJ r2 bkg 32
submit 1lep_1FJ r3 bkg 32

submit 0lep_3FJ r2 bkg 32
submit 0lep_3FJ r3 bkg 32

wait
