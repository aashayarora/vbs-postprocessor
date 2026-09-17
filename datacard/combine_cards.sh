#!/bin/bash
set -euo pipefail

CHANNEL=${1:-all}
shift || true
EXTRA_REGIONS=("$@")

# Build the combineCards.py argument list for a region directory by
# globbing its scan files: <region>/datacard_scan_Scan<N>.dat
# -> r<tag>s<N>=<region>/datacard_scan_Scan<N>.dat
#
# Bin labels are kept short on purpose: combine builds PDF names like
# pdf_bin<label>_A, and RooFit's factory string overflows (and historically
# corrupts results) past ~1000 chars when many channels are combined. A short
# label keeps the combined string well under that limit. The region<->tag
# mapping is printed below for traceability.
cards_for() {
    local region=$1 tag=$2
    local f n
    for f in "$region"/datacard_scan_Scan*.dat; do
        [[ -e "$f" ]] || continue
        n=$(basename "$f" .dat); n=${n#datacard_scan_Scan}
        printf 'r%ss%s=%s ' "$tag" "$n" "$f"
    done
}

# Discover all regions: directories containing at least one scan file.
REGIONS=()
for d in */; do
    d=${d%/}
    compgen -G "$d/datacard_scan_Scan*.dat" >/dev/null && REGIONS+=("$d")
done

# Validate that a region directory has scan files; exit otherwise.
check_region() {
    local r=$1
    if [[ -d "$r" ]] && compgen -G "$r/datacard_scan_Scan*.dat" >/dev/null; then
        return 0
    fi
    echo "Unknown region: $r"
    echo "Available regions: ${REGIONS[*]}"
    exit 1
}

# Select regions based on the requested channel.
case "$CHANNEL" in
    all) SELECTED=("${REGIONS[@]}"); OUTPUT="datacard_combined.dat" ;;
    r2)  SELECTED=(); for r in "${REGIONS[@]}"; do [[ $r == r2_* ]] && SELECTED+=("$r"); done; OUTPUT="datacard_r2.dat" ;;
    r3)  SELECTED=(); for r in "${REGIONS[@]}"; do [[ $r == r3_* ]] && SELECTED+=("$r"); done; OUTPUT="datacard_r3.dat" ;;
    *)
        # One or more region names: combine just those.
        check_region "$CHANNEL"
        SELECTED=("$CHANNEL")
        for r in "${EXTRA_REGIONS[@]}"; do
            check_region "$r"
            SELECTED+=("$r")
        done
        if [[ ${#SELECTED[@]} -eq 1 ]]; then
            OUTPUT="datacard_${CHANNEL}.dat"
        else
            OUTPUT="datacard_$(IFS=_; echo "${SELECTED[*]}").dat"
        fi
        ;;
esac

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    echo "No regions matched channel: $CHANNEL"
    exit 1
fi

INPUT=""
ri=0
for r in "${SELECTED[@]}"; do
    ri=$((ri+1))
    echo "  r${ri} = ${r}"
    INPUT+="$(cards_for "$r" "$ri")"
done

combineCards.py $INPUT > "$OUTPUT"

combine -M AsymptoticLimits "$OUTPUT" --run blind