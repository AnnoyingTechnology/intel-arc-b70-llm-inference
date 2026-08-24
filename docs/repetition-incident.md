# Long-context token-0 repetition incident

Status on 2026-08-24: reproducible and contained, but the permanent server-side
fix is not yet proven. MTP4 remains enabled. Do not treat a repetition penalty
or disabling MTP as the fix.

## Symptom

At roughly 50K prompt tokens, an OpenCode assistant turn began emitting only
`!` in its reasoning stream. The user stopped each affected generation; the
model did not terminate the repetitions itself. Compacting the session did not
recover it.

The Qwen tokenizer maps `!` to token ID 0. The failure is therefore a pure
token-0 collapse, not a normal prose loop. During affected requests, all four
MTP draft positions had 0% acceptance while the target continued emitting the
same token.

A one-token non-streaming request with `logprobs=true` made the failure
explicit: vLLM returned HTTP 400 because the response contained `nan`, which
its strict JSON serializer rejected. The result was unchanged at temperature
0. The identical probe on the succeeding cold follow-up returned one finite
chosen logprob and 20/20 finite top logprobs. Token 0 is therefore downstream
of a non-finite model/sampling distribution, not a legitimate maximum or a
missing repetition penalty.

## Privacy-preserving reproducer

[`opencode_repetition_probe.py`](../scripts/opencode_repetition_probe.py) can
capture one OpenCode-compatible request through a localhost-only endpoint and
replay it directly against vLLM. Captures are written mode `0600`; replay output
contains only counts, hashes and timing, never transcript text.

The incident request was captured in tmpfs and is deliberately excluded from
this repository. It contained 24 messages, 12 tools, 143,158 serialized message
characters and 49,969 model prompt tokens. The first-failure boundary is the
first 20 messages, or 49,925 model prompt tokens. OpenCode had omitted the
manually aborted assistant output, so no run of repeated `!` was present in the
replayed input.

## Reproduced facts

All rows used the current XPU target, FP8 KV cache, prefix caching and MTP4.
"Qwen sampler" means `temperature=1`, `top_p=0.95`, `top_k=20`, matching the
official model generation configuration. A unique `cache_salt` creates an
isolated prefix-cache namespace without disabling caching.

| Test | Cache state | Result |
|---|---|---|
| Exact 24-message OpenCode payload, current client sampler | existing | 3/3 runs emitted 96/96 `!` |
| Same payload, Qwen sampler | existing | 3/3 runs emitted 96/96 `!` |
| Same payload, repetition penalty 1.01 through 1.20 | existing | every run emitted 96/96 `!` |
| Exact 24-message payload, Qwen sampler | cold unique namespace | 3/3 coherent; 0 repeated `!` |
| Same clean namespace reused | warm | 3/3 coherent and byte-identical |
| Exact 24-message payload, `top_k=20` only (`top_p=1`) | cold unique namespace | 3/3 coherent; 0 repeated `!` |
| First-failure 20-message boundary, Qwen sampler | cold unique namespace | reproducibly emitted only `!` |
| First recovery message, Qwen sampler | cold unique namespace | coherent; 0 repeated `!` |
| First recovery message after caching the failed boundary | warm, 48,256-token hit | emitted 32/32 `!` |
| Next recovery message after caching a successful request | warm, 48,256-token hit | coherent; 0 repeated `!` |
| First-failure boundary with logprobs, Qwen sampler | cold | HTTP 400: response contains `nan` |
| First-failure boundary with logprobs, greedy | cold | HTTP 400: response contains `nan` |
| First recovery message with logprobs, Qwen sampler | cold | chosen 1/1 finite; top 20/20 finite |

The controlled failed-then-follow-up pair is the strongest result:

1. The 49,925-token request was processed cold, emitted 32/32 `!`, and created
   49,920 cache tokens.
2. The 49,940-token follow-up reused 48,256 tokens from that namespace, finished
   in 3.79 seconds, and emitted the same 32/32 `!`.
3. The identical follow-up in a new namespace processed cold, finished in 44.50
   seconds, and produced coherent reasoning.

This demonstrates a bad reusable recurrent prefix state. It also explains why
OpenCode compaction did not repair the session: the follow-up still matched the
damaged in-memory prefix. The API server stayed healthy and logged no OOM,
device error, engine exception or restart.

## Conclusions and remaining isolation

Proven:

- The UI is not required to reproduce the failure.
- The aborted `!` output is not poisoning the serialized chat history.
- A repetition penalty does not address token-0 collapse.
- The failure contains NaN logprob data under both stochastic and greedy
  sampling; sampler tuning cannot repair non-finite upstream values.
- Qwen's sampler plus a clean cache namespace recovers the complete current
  session while retaining MTP4.
- For the recovered 24-message payload, `top_k=20` alone prevents the
  full-vocabulary failure: 3/3 cold passes while retaining `top_p=1`. This is
  an independently proven client guardrail, not the explanation for the
  20-message cold NaN that survives `top_k=20`.
- Reuse of state produced by the failing turn makes a later prompt fail when
  the same prompt succeeds from a cold prefill.

The strongest untested cause is now the forced target dtype. The checkpoint
declares BF16 and inspection of its embedding, normalization, GDN and LM-head
tensors confirms BF16 storage. Compose nevertheless passes `--dtype float16`,
and vLLM logs `Casting torch.bfloat16 to torch.float16`. This cuts the exponent
range of every target activation. The selected Intel `XPUwNa16LinearKernel`
explicitly supports both BF16 and FP16, so BF16 is a viable MTP-preserving A/B,
not a backend switch.

OpenCode's `local-b70/qwen38` options now explicitly send the official
`temperature=1`, `top_p=0.95`, `top_k=20` values. A localhost capture of the
actual configured client confirmed those exact snake-case fields on the wire.
The other OpenCode model entries were not changed.

Not yet proven:

- Whether restoring BF16 eliminates the cold NaN and the poisoned warm prefix
  while retaining the measured speed. It is the first restart A/B.
- Which layer first becomes non-finite. If BF16 does not fix it, instrumentation
  must record finite/min/max summaries after each target layer without logging
  activations or transcript content.
- Whether fine-grained hybrid prefix matching contributes to propagation. The
  very recent path enabled by `--prefix-match-unit 64` remains a second A/B,
  but it cannot by itself explain the initial cold request's NaN because that
  request read zero cached tokens.
- Whether draft-only INT4 contributes. It is a later A/B only if the cache
  granularity test fails; MTP remains enabled.

The current vLLM checkout is
`0.27.2rc1.dev77+gac7509e2b` (2026-08-14). Fine-grained hybrid cache primitives
landed only in June/July 2026, followed by multiple cache correctness fixes.
Related upstream reports describe Qwen tool-use sessions degrading to 0% MTP
acceptance and gibberish, MTP/prefix-cache/FP8 corruption, and Qwen endless
`!` output.

## Immediate recovery

For a stuck session, send the next replay/request with both the Qwen sampler and
a new private cache namespace:

```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "cache_salt": "NEW-UNGUESSABLE-VALUE"
}
```

This was repeatably successful for the complete captured session. It is a
containment measure, not the permanent fix: a static salt can itself accumulate
a bad state, and changing it on every turn would throw away the TTFT benefit of
prefix caching.

## Next controlled A/B

1. Change only `--dtype float16` to `--dtype bfloat16` in Compose.
2. Recreate the single B70 container, clearing only in-memory prefix state.
3. Keep MTP4, FP8 KV, the target, draft overlay, scheduler and 210 W cap intact.
4. Replay the exact 20-message failure boundary cold three times, then its
   follow-up warm three times.
5. Replay the complete 24-message request cold and warm, and rerun the cached
   TTFT, quality and p512/p8192 decode gates.
6. Promote BF16 only if NaNs are gone and quality/performance pass. Otherwise
   restore FP16 and run the same matrix after removing only
   `--prefix-match-unit 64`.

Service recreation is intentionally pending explicit authorization. Expected
impact is roughly two minutes of local inference unavailability and loss of
in-memory prefix entries. Rollback is to restore the single Compose dtype and
recreate the container; model data and the persistent compile cache are not
removed.

## Upstream references

- [Qwen3.5 MTP multi-turn tool-use gibberish and acceptance collapse](https://github.com/vllm-project/vllm/issues/36872)
- [MTP, prefix caching and FP8 KV tool-call corruption](https://github.com/vllm-project/vllm/issues/50188)
- [Qwen endless exclamation-mark output](https://github.com/vllm-project/vllm/issues/39348)
- [Fine-grained partial cache hits for hybrid/Mamba models](https://github.com/vllm-project/vllm/issues/45702)
- [Official Qwen3.8-27B generation configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/generation_config.json)
