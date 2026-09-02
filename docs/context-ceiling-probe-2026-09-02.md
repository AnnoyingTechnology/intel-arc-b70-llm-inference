# Context ceiling probe — 2026-09-02

## Outcome

The promoted service uses the model's native 262,144-token limit with an exact
10,300,000,000-byte FP8 KV reservation. The engine reports 263,633 KV-cache
tokens and 1.01x maximum concurrency. The container starts cleanly with the
vision encoder's 16,384-token budget, remains healthy, and passed short text
and processor-maximum-image canaries after promotion.

OpenCode uses the empirically completed usable contract: `context: 253952`,
`input: 245760` and `output: 8192`. The server retains the model's native
262,144-token limit, while the client keeps 8,192 tokens of scheduler/KV
headroom below it and another 8,192 tokens inside the usable window for
generation. Normal agent sessions begin small and grow through automatic prefix
caching.

## Exact evidence

| Configuration and request | Result |
|---|---|
| Final 196,608 layout, cold text, 196,480 prompt + 128 output | Completed in 617.63 s; TTFT 613.59 s; zero cached tokens |
| Final 196,608 layout, cold 4096x4096 image, 196,480 prompt + 128 output | Completed in 635.87 s; TTFT 632.11 s; exactly 16,384 image tokens |
| Native 262,144 layout, 10.3 GB KV, text, 262,016 prompt + 128 output | Accepted, but ultimately parked at `waiting: capacity` with zero running requests; not a usable boundary |
| Native layout with 4,096-token reserve, text, 257,920 prompt + 128 output | Progressed, then also parked at `waiting: capacity`; reserve rejected |
| Native layout with 8,192-token reserve, text, 253,824 prompt + 128 output | **Completed exactly 253,952 tokens** in 1,770.16 s; TTFT 1,750.44 s; 6.44 decode tok/s |

All three completed boundary requests returned `finish_reason=length`, exactly
128 completion tokens and zero reasoning tokens. Their peak reported
VRAM-channel temperature was 70°C. The zero- and 4,096-reserve attempts did not
OOM or restart the service, but their terminal capacity-wait state means they
must not be reported as usable boundary passes.

The initial 10,700,000,000-byte trial started successfully and reported 274,059
KV tokens. Reducing the reservation to 10,300,000,000 bytes returned about
400 MB of transient headroom while retaining 1,489 nominal cache tokens above
the native model limit. Runtime evidence shows that the nominal 1.01x figure
does not include enough effective scheduler/hybrid-cache headroom for a request
whose prompt plus generation consumes the full native limit.

## Low-context A/B

The 262,144/10.3 GB profile was compared with the prior 196,608/8.5 GB profile
at 210 W using the same five cold prompt families and 128 requested output
tokens per family.

| Cold workload | 196K profile | 262K profile | Relative result |
|---|---:|---:|---:|
| ~512 prompt, median TTFT | 0.389410 s | 0.388989 s | -0.11% |
| ~512 prompt, median prefill | 1,302.02 tok/s | 1,311.32 tok/s | +0.71% |
| ~512 prompt, median decode | 83.73 tok/s | 83.79 tok/s | +0.07% |
| ~8K prompt, median TTFT | 4.855429 s | 4.848541 s | -0.14% |
| ~8K prompt, median prefill | 1,686.77 tok/s | 1,688.76 tok/s | +0.12% |
| ~8K prompt, median decode | 87.47 tok/s | 82.48 tok/s | -5.71% |

Enabling the native server window does not collapse low-context performance:
TTFT and prefill are unchanged or slightly better at both sizes, and 512-token
decode is unchanged. The 8K decode difference coincides with lower median MTP
acceptance (0.639 versus 0.691), so it is content/speculation variance rather
than evidence of a max-length allocation penalty.

## Test fixture and interpretation

The deterministic prompt used repeated ` x` units because the active tokenizer
maps each unit to exactly one token. Every request was preflighted through
`/tokenize`; the runner aborted unless the prompt count exactly matched the
target. The maximum-image fixture was a 4096x4096 PNG with SHA-256
`c9dc591dc720f64376c065d101fd0b3f6e89b52fda27f38e50e4b370ebd5b14e`.

Cold full-context latency is not the production workload. OpenCode agent
conversations grow incrementally and reuse cached prefix blocks. The native
server profile is retained because it has no demonstrated low-context penalty,
but the client contract is capped at the exact completed 253,952-token boundary
to preserve the required 8,192-token runtime reserve.

Raw boundary result JSON and the temporary runner are retained on the B70 host
under `/var/tmp/b70-context-ceiling-20260902/`. The four low-context A/B result
files are under `/var/tmp/b70-context-ab-20260902/`.

## Validation and rollback

The promoted service returned exact `FINAL_OK` and `VISION_OK` canaries; the
vision canary exercised all 16,384 image tokens. `/v1/models` advertised
`max_model_len: 262144`, and the container was healthy with zero restarts and
`OOMKilled=false`.

Rollback is `--max-model-len 196608` with
`--kv-cache-memory 8500000000`, followed by a Compose recreate and the same
short text and vision canaries.
