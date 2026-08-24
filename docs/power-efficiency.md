# Power-efficiency sweep

## Selected operating point

The persisted B70 power cap is **210 W**. Intel specifies a 230 W reference TBP and a configurable 160–290 W range for the Arc Pro B70. This board exposed a 275 W initial cap through the Xe hwmon interface.

The choice is the cross-workload knee, not the lowest-power point:

- Decode retains 98.6% of the 275 W rate while using 11.6% less energy per output token.
- Cold 8K prefill retains 82.5% of throughput while using 6.1% less energy per input token.
- Cold-prefill energy/token is best at 210 W. Lower caps are slower and make prefill less efficient.
- Decode efficiency continues improving below 210 W, but that does not compensate for the prefill regression in an interactive agent workload.

Normal local-agent use has been observed around 65 °C with only faint, unobtrusive noise. The automated descending sweep was heat-soaked, so its temperature columns are not used for cross-cap conclusions. A cache-isolated 32K cold run at the final cap reached 64 °C; the 100K saturation run reached 70 °C.

## Method

The sweep tested `275, 250, 230, 210, 190, 175, 160 W`. Every cap used three byte-identical, cache-isolated requests in each workload:

- Prefill: 8,192 input tokens and 8 output tokens.
- Decode: 512 input tokens and 512 output tokens.
- One active request; identical model, server, scheduler and MTP settings.
- Card/package energy counters, TTFT, decode time, temperature, fan and active clock sampled from Xe hwmon/sysfs.

| Cap | 8K prefill | Prefill J/input tok | p512/g512 decode | Decode J/output tok |
|---:|---:|---:|---:|---:|
| 275 W | 1,948 tok/s | 0.1416 | 84.97 tok/s | 2.811 |
| 250 W | 1,838 tok/s | 0.1372 | 85.08 tok/s | 2.835 |
| 230 W | 1,729 tok/s | 0.1347 | 84.56 tok/s | 2.697 |
| **210 W** | **1,608 tok/s** | **0.1329** | **83.77 tok/s** | **2.486** |
| 190 W | 1,446 tok/s | 0.1333 | 82.99 tok/s | 2.267 |
| 175 W | 1,337 tok/s | 0.1347 | 81.52 tok/s | 2.131 |
| 160 W | 1,207 tok/s | 0.1368 | 79.21 tok/s | 2.019 |

Evidence is under `results/power-sweep/`. The harness is `scripts/run_power_sweep.sh`; raw request/telemetry logic is in `scripts/power_bench.py`.

## Persistence

The boot-persistent unit is `b70-power-limit.service`. It calls the root-owned `/usr/local/sbin/b70-power-limit`, verifies PCI ID `8086:e223`, bounds values to Intel's documented 160–290 W range, waits for Xe hwmon, writes the cap, and reads it back.

```bash
systemctl is-enabled b70-power-limit.service
systemctl is-active b70-power-limit.service
cat /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*/power1_cap
```

Expected cap: `210000000` microwatts.

Rollback to the board's pre-sweep 275 W state:

```bash
sudo systemctl disable --now b70-power-limit.service
sudo /usr/local/sbin/b70-power-limit 275
```

Re-enable the selected cap:

```bash
sudo systemctl enable --now b70-power-limit.service
```

The tracked source files are `systemd/b70-power-limit.service` and `scripts/set-power-cap.sh`.
