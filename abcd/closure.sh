#!/bin/bash

REGION=${REGION:-all}
NSCAN=${NSCAN:-50}

data_pred_file() {
    ls "$1"/predictions_single*_data.parquet 2>/dev/null | tail -1
}

latest() {
    ls -d "$1"/version_* 2>/dev/null \
        | sort -t_ -k2 -n \
        | while read -r d; do [ -n "$(data_pred_file "$d")" ] && [ -f "$d/regions.yaml" ] && echo "$d"; done \
        | tail -1
}

run() {
    BASE=$(latest "output/$1/single")
    if [ -z "$BASE" ]; then
        echo "[skip] $1: no version_* with predictions_single*_data.parquet + regions.yaml" >&2
        return
    fi
    DATA=$(data_pred_file "$BASE")
    echo "[run]  $1 ($2) -> $BASE"
    python3 closure.py -i "$DATA" \
        --channel "$2" \
        --regions "$BASE/regions.yaml" \
        --region "$REGION" \
        -n "$NSCAN" \
        -o "$BASE/closure" > "closure_$3.log" 2>&1 &
}

run 1LEP_2FJ_RUN2 2fj 1lep_boosted_run2
run 1LEP_2FJ_RUN3 2fj 1lep_boosted_run3
run 1LEP_1FJ_RUN2 1fj 1lep_semiresolved_run2
run 1LEP_1FJ_RUN3 1fj 1lep_semiresolved_run3
run 0LEP_3FJ_RUN2 3fj 0lep_boosted_run2
run 0LEP_3FJ_RUN3 3fj 0lep_boosted_run3

wait
echo "All closure jobs finished. Summaries: output/*/single/version_*/closure/closure_summary_*.yaml"
