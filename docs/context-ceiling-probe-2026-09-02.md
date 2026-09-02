# Context ceiling probe — 2026-09-02

## Outcome

The promoted service uses the model's native 262,144-token limit with an exact
10,300,000,000-byte FP8 KV reservation. The engine reports 263,633 KV-cache
tokens and 1.01x maximum concurrency. The container starts cleanly with the
vision encoder's 16,384-token budget, remains healthy, and passed short text
and processor-maximum-image canaries after promotion.

OpenCode uses `context: 262144`, `input: 253952` and `output: 8192`. The 8,192
tokens are an output reserve; normal agent sessions begin small and grow through
automatic prefix caching.

## Exact evidence

| Configuration and request | Result |
|---|---|
| Final 196,608 layout, cold text, 196,480 prompt + 128 output | Completed in 617.63 s; TTFT 613.59 s; zero cached tokens |
| Final 196,608 layout, cold 4096x4096 image, 196,480 prompt + 128 output | Completed in 635.87 s; TTFT 632.11 s; exactly 16,384 image tokens |
| Native 262,144 layout, 10.3 GB KV, cold text, 262,016 prompt + 128 output | Accepted and ran for 1,800.40 s without OOM, restart or thermal failure; timed out before first token |

Both completed boundary requests returned `finish_reason=length`, exactly 128
completion tokens and zero reasoning tokens. Their peak reported VRAM-channel
temperature was 70°C. The native-limit attempt peaked at 68°C; its timeout is
not a completed boundary pass and must not be reported as one.

The initial 10,700,000,000-byte trial started successfully and reported 274,059
KV tokens. Reducing the reservation to 10,300,000,000 bytes returned about
400 MB of transient headroom while retaining 1,489 cache tokens above the
native model limit.

## Test fixture and interpretation

The deterministic prompt used repeated ` x` units because the active tokenizer
maps each unit to exactly one token. Every request was preflighted through
`/tokenize`; the runner aborted unless the prompt count exactly matched the
target. The maximum-image fixture was a 4096x4096 PNG with SHA-256
`c9dc591dc720f64376c065d101fd0b3f6e89b52fda27f38e50e4b370ebd5b14e`.

Cold full-context latency is not the production workload and was not pursued
further. OpenCode agent conversations grow incrementally and reuse cached
prefix blocks. The native profile is promoted on demonstrated allocation and
runtime stability, while exact cold completion above 196,608 remains explicitly
unproven.

Raw result JSON and the temporary runner are retained on the B70 host under
`/var/tmp/b70-context-ceiling-20260902/`.

## Validation and rollback

The promoted service returned exact `FINAL_OK` and `VISION_OK` canaries; the
vision canary exercised all 16,384 image tokens. `/v1/models` advertised
`max_model_len: 262144`, and the container was healthy with zero restarts and
`OOMKilled=false`.

Rollback is `--max-model-len 196608` with
`--kv-cache-memory 8500000000`, followed by a Compose recreate and the same
short text and vision canaries.
