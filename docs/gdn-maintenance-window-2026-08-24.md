# GDN maintenance window: rejected scheduler split

## Outcome

The 2026-08-24 maintenance window tested the staged MTP-safe scheduler split
against the pinned production stack. The candidate did **not** correct the
five-token NaN and was rolled back. The active service is again the prior
Compose configuration: Qwen3.8-27B, MTP4, FP8 KV cache, prefix caching, an
8,192-token scheduler budget, a 196,608-token serving limit and the persisted
210 W cap.

No issue, pull request or other upstream submission was made.

## Test sequence

1. The pre-change API neighbor matrix reproduced the established rule: 5, 69
   and 133 tokens returned HTTP 400 because of NaN logprobs; 4/6, 68/70 and
   132/134 returned 20/20 finite top logprobs.
2. Only `b70-vllm-qwen38` was stopped. A disposable container using the exact
   pinned image ran [`gdn_operator_tail_probe.py`](../scripts/gdn_operator_tail_probe.py).
3. All 144 synthetic direct-operator cases were finite: fused and split paths,
   FP16 and BF16 activations, FP32 and model-dtype state, reordered and normal
   input, and the nine neighbor lengths. All virtual-padding assertions passed.
4. Compose mounted and invoked `patch_xpu_gdn_tail.py`; startup logs confirmed
   the scheduler source was patched. The recreated engine was healthy with zero
   restarts and still reported four speculative tokens.
5. The mandatory fixed-oracle check failed: lengths 4 and 6 remained finite,
   while length 5 still returned the same NaN HTTP 400. The longer sweep was
   stopped at this first hard failure.
6. The activation lines were removed and only `b70-vllm-qwen38` was recreated.
   The persistent compile cache and model data were untouched.

The direct synthetic pass does not contradict the API failure. It shows that
random tensors at the public operator boundary are insufficient to reproduce
the bad condition; real checkpoint activations or surrounding execution state
are part of the trigger. Likewise, the failed API gate proves that the proposed
scheduler split is not a correction, but does not by itself establish whether
the condition failed to match or whether the resulting second scheduler step
still reached a bad internal kernel shape.

## Rollback validation

| Gate | Restored-service result |
|---|---:|
| Container health | healthy, restart count 0 |
| Serving contract | `qwen38`, max length 196,608 |
| MTP | MTP4; 1,182/1,776 draft tokens accepted across the mixed recovery suite |
| Power cap | 210 W |
| Original NaN oracle | expected failures only at 5, 69 and 133 |
| Deterministic quality | 7/7 exact canaries; 8/8 repeat stability; prior hashes matched |
| Long-context quality | 30,350-token needle passed; prior hash matched |
| p512/g128, five families | 85.07 tok/s decode; 1,314 prompt tok/s; 0.387 s TTFT |
| p8192/g128, five families | 88.39 tok/s decode; 1,570 prompt tok/s; 5.216 s TTFT |
| 32K tool flow | correct tool and arguments; correct final value |
| 32K tool follow-up | 3.614 s; 29,952/32,913 prompt tokens cached |

The 8K cell is within 3% of the earlier 91.08 tok/s decode median and matches
its prefill rate. The short decode cell varied by 7.5%; this suite's output rate
tracks content-dependent MTP acceptance, while the service configuration and
quality hashes are identical to the rollback point.

Raw recovery evidence:

- `results/repetition-incident-20260824/rollback-quality-20260824.json`
- `results/repetition-incident-20260824/rollback-p512-g128-20260824.json`
- `results/repetition-incident-20260824/rollback-p8192-g128-20260824.json`
- `results/repetition-incident-20260824/rollback-prefix-tool-32k-20260824.json`

## Disposition

Do not mount or invoke `patch_xpu_gdn_tail.py` in production. It remains only as
a rejected experiment and source-audit artifact. The next correction must be
proven against the raw five-token API request before another service recreate.
A useful next isolation step is to capture the real projected GDN inputs and
metadata at the failing first prefill, then replay those through the split and
fused direct-operator paths. Any later candidate must still keep MTP4 and pass
the exhaustive finite-logprob gate before performance or quality promotion.
