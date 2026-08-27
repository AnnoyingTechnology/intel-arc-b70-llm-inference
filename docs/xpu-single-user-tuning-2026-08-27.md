# XPU single-user tuning — 2026-08-27

## Decision

The service is intentionally single-user and already limits execution to one
sequence. `--performance-mode interactivity` is therefore retained. It gives
MTP4's five-token verifier an exact graph instead of padding it to eight tokens.

The Qwen fused gated projection + Q/K RMSNorm + RoPE path is rejected. Its XPU
kernel is fast in isolation but does not improve the compiled service.

Both changes are lossless: they leave weights, activation precision, KV format,
sampling and MTP verification unchanged.

## Exact-graph result

With the default `balanced` mode, this one-sequence MTP4 service captured graph
sizes `[1, 2, 4, 8]`. vLLM dispatches an uncaptured size to the nearest larger
graph, so the five-token MTP verifier used an eight-row graph. `interactivity`
captures every size from 1 through 10, including five.

The A/B used the same 476-token prompt, 512-token forced output, power cap,
model, image and sampling configuration. Each side had a warm-up followed by
six recorded requests.

| Mode | Median decode | Decode energy | Change |
|---|---:|---:|---:|
| `balanced` | 84.60 tok/s | 2.471 J/token | baseline |
| `interactivity` | 85.61 tok/s | 2.449 J/token | +1.19% throughput; -0.92% energy/token |

A three-run cold 32K control changed from 24.605 to 24.718 seconds TTFT and
from 0.15864 to 0.15904 J/input-token. This is a 0.5% regression and is treated
as measurement noise: graph-size selection applies to small decode steps, not
prefill.

The first startup captured six additional graph sizes and took longer. The
persistent vLLM cache retains the compiled artifacts. Reported KV capacity
remained 209,523 tokens.

## Correctness gate

The promoted mode passed:

- all seven deterministic output canaries;
- eight identical repeat hashes;
- exact hash parity with the accepted service baseline;
- finite raw logprobs for lengths 1–128, 133, 197 and 261.

The last gate covers the five-token shape and the neighboring `64*N+5` cases
that exposed the prior graph-classification defect.

## Rejected Qwen fusion

The installed vLLM contains a model-specific fused gated projection + Q/K
RMSNorm + RoPE kernel, but Qwen enables it only on CUDA. The existing kernel ran
unchanged on XPU for the service geometry: FP16, 24 query heads, 4 KV heads,
head dimension 256 and rotary dimension 64.

Across token counts 1, 4, 5, 37, 64, 257, 512 and 8,192, its direct output
matched the reference within 0.01. The isolated operation was about 78–84%
faster; at 8,192 tokens it fell from 7.36 to 1.16 ms.

That result did not survive full serving:

| Configuration | 32K cold TTFT | 512-token decode |
|---|---:|---:|
| Existing compiled path | 24.605 s | 84.60 tok/s |
| Qwen fused kernel enabled on XPU | 24.638 s | 83.77 tok/s |

Prefill was unchanged and decode regressed 0.98%. The compiled serving graph
already removes most of the eager-operation overhead; forcing the standalone
kernel adds no end-to-end value. It must not be enabled or proposed upstream on
the strength of its microbenchmark.

## Remaining lossless work

Single-user operation removes batch-throughput tuning from the search space.
The remaining substantial target is long-context prefill: upstreaming and
broadening the validated head-256 K64 attention policy, then testing neighboring
Xe2 shapes. Decode is close to the card's memory-bandwidth limit; further
lossless decode kernel work needs graph-replay visibility and a measured hotspot.
