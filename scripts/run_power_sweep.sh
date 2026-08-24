#!/bin/sh
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
results="$root/results/power-sweep"
set_cap="$root/scripts/set-power-cap.sh"
bench="$root/scripts/power_bench.py"
prefill_prompts="$root/results/frozen-gptq-mtp4-p8192-g128/prompts.json"
decode_prompts="$root/results/frozen-gptq-mtp4-p512-g128/prompts.json"

cap_file=
for candidate in /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*/power1_cap; do
    if [ -e "$candidate" ]; then
        cap_file=$candidate
        break
    fi
done
[ -n "$cap_file" ] || { echo "B70 power1_cap not found" >&2; exit 1; }

original_uw=$(cat "$cap_file")
original_w=$((original_uw / 1000000))
restore() {
    "$set_cap" "$original_w" >/dev/null 2>&1 || true
}
trap restore EXIT HUP INT TERM

mkdir -p "$results"

if [ "$#" -eq 0 ]; then
    set -- 275 250 230 210 190 175 160
fi

for watts in "$@"; do
    "$set_cap" "$watts"
    sleep 3

    prefill_json="$results/cap-${watts}-prefill-p8192-g8.json"
    prefill_log="$results/cap-${watts}-prefill-p8192-g8.log"
    "$bench" \
        --prompt-file "$prefill_prompts" \
        --output-tokens 8 --repeats 3 --fixed-index --settle 2 \
        --output "$prefill_json" > "$prefill_log"
    jq -c '{workload:"prefill",summary}' "$prefill_json"

    decode_json="$results/cap-${watts}-decode-p512-g512.json"
    decode_log="$results/cap-${watts}-decode-p512-g512.log"
    "$bench" \
        --prompt-file "$decode_prompts" \
        --output-tokens 512 --repeats 3 --fixed-index --settle 2 \
        --output "$decode_json" > "$decode_log"
    jq -c '{workload:"decode",summary}' "$decode_json"
done
