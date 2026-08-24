#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 WATTS" >&2
    exit 2
fi

watts=$1
case "$watts" in
    *[!0-9]*|'') echo "power cap must be an integer number of watts" >&2; exit 2 ;;
esac

if [ "$watts" -lt 160 ] || [ "$watts" -gt 290 ]; then
    echo "refusing cap outside Intel's documented B70 range: 160-290 W" >&2
    exit 2
fi

pci=/sys/bus/pci/devices/0000:03:00.0

# The boot service can run before xe has finished publishing hwmon. Wait only
# for this exact PCI function, then verify its identity before writing.
attempt=0
while [ ! -r "$pci/vendor" ] || [ ! -r "$pci/device" ]; do
    attempt=$((attempt + 1))
    [ "$attempt" -lt 31 ] || { echo "B70 PCI function not ready" >&2; exit 1; }
    sleep 1
done
[ "$(cat "$pci/vendor")" = "0x8086" ] || { echo "unexpected PCI vendor" >&2; exit 1; }
[ "$(cat "$pci/device")" = "0xe223" ] || { echo "unexpected PCI device" >&2; exit 1; }

cap_file=
attempt=0
while [ -z "$cap_file" ]; do
    for candidate in "$pci"/hwmon/hwmon*/power1_cap; do
        if [ -e "$candidate" ]; then
            cap_file=$candidate
            break
        fi
    done
    [ -n "$cap_file" ] && break
    attempt=$((attempt + 1))
    [ "$attempt" -lt 31 ] || break
    sleep 1
done
[ -n "$cap_file" ] || { echo "B70 power1_cap not found" >&2; exit 1; }

microwatts=$((watts * 1000000))
printf '%s\n' "$microwatts" > "$cap_file"
actual=$(cat "$cap_file")
[ "$actual" -eq "$microwatts" ] || {
    echo "requested ${microwatts}uW but driver reports ${actual}uW" >&2
    exit 1
}
echo "B70 power cap: $((actual / 1000000)) W"
