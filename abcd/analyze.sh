#!/bin/bash

pred_file() {
    ls "$1"/predictions_single*.parquet 2>/dev/null | grep -v '_data\.parquet$' | tail -1
}

latest() {
    ls -d "$1"/version_* 2>/dev/null \
        | sort -t_ -k2 -n \
        | while read -r d; do [ -n "$(pred_file "$d")" ] && echo "$d"; done \
        | tail -1
}

run_analysis() {
    local pred; pred=$(pred_file "$(latest "$2")")
    echo "Using $(basename "$(dirname "$pred")") for $3"
    echo "Predictions: $pred"
    python3 run_analysis.py -c "$1" -i "$pred" -d "${pred%.parquet}_data.parquet" -n 5 > "$4"
}

run_analysis 2fj output/1LEP_2FJ_RUN2/single 1LEP_2FJ_RUN2 config_1lep_boosted_run2.log &
run_analysis 2fj output/1LEP_2FJ_RUN3/single 1LEP_2FJ_RUN3 config_1lep_boosted_run3.log &
run_analysis 1fj output/1LEP_1FJ_RUN2/single 1LEP_1FJ_RUN2 config_1lep_semiresolved_run2.log &
run_analysis 1fj output/1LEP_1FJ_RUN3/single 1LEP_1FJ_RUN3 config_1lep_semiresolved_run3.log &
run_analysis 3fj output/0LEP_3FJ_RUN2/single 0LEP_3FJ_RUN2 config_0lep_boosted_run2.log &
run_analysis 3fj output/0LEP_3FJ_RUN3/single 0LEP_3FJ_RUN3 config_0lep_boosted_run3.log &
