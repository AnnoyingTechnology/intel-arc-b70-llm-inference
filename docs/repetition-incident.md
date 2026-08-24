# Long-context token-0 repetition incident

Status on 2026-08-24: the trigger is reduced to a fully synthetic, deterministic
XPU GDN failure. The first scheduler-split candidate failed its mandatory raw
API gate and was rolled back; no server-side correction is deployed. MTP4
remains enabled. Do not treat a repetition penalty or disabling MTP as the fix.

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

## Public synthetic reproducer

The failure no longer needs the private OpenCode capture. On the running XPU
service, raw completion prompts fail exactly when:

```text
prompt_tokens % 64 == 5
```

[`gdn_tail_probe.py`](../scripts/gdn_tail_probe.py) sends only repeated synthetic
one-token units, uses a unique cache namespace for every request, requests one
output token plus logprobs, and never prints response text:

```bash
./scripts/gdn_tail_probe.py
```

The default neighbor matrix is 4/5/6, 68/69/70 and 132/133/134 tokens. On the
affected runtime, only 5, 69 and 133 return HTTP 400 because strict JSON
serialization encounters `nan`; every adjacent control returns 20/20 finite top
logprobs. Exhaustive sweeps over 1–64 and 65–128 produced no other failures.
Five different synthetic one-token units all failed at total length five, so
the trigger is length-dependent rather than content-dependent.

These raw requests have no chat template, tools, prior prefix, resumed GDN
state or speculative-decode metadata. MTP4 is still configured on the server,
but the first target prefill occurs before draft verification. The equality
between the five-token failing tail and the MTP4 verification width is therefore
coincidental.

The XPU implementation internally uses 64-token GDN chunks. The live path is
`torch.ops.vllm.gdn_attention_core_xpu` ->
`torch.ops._xpu_C.gdn_attention` from `vllm-xpu-kernels 0.1.12.3`. This proves
the failure is upstream of sampling in the XPU GDN prefill path. The exact bad
instruction inside the compiled kernel is still under source-level isolation.

## Direct-operator result

During the authorized maintenance window, the serving container was stopped
and [`gdn_operator_tail_probe.py`](../scripts/gdn_operator_tail_probe.py) ran in
a disposable container using the same pinned image. All 144 cases were finite:
fused and split operators, FP16/BF16 activations, FP32/model-dtype recurrent
state, reordered/non-reordered input, and the nine API-neighbor lengths. The
split path also found no non-zero virtual padding between `causal_conv1d` and
`gated_delta_rule`.

This negative result means random projected tensors at the direct operator
boundary do not reproduce the fault. It does not override the deterministic
end-to-end API oracle, which uses the real checkpoint activations and runtime
metadata.

## Private incident replay

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
- Qwen's sampler plus a clean cache namespace recovered the complete current
  session because that rendered prompt did not have the failing remainder;
  MTP4 remained enabled.
- The recovered 24-message payload also passed 3/3 cold runs with only
  `top_k=20` added while retaining `top_p=1`. This supports the official sampler
  configuration but does not establish a second root cause: the exact
  `64*N+5` cold NaN survives `top_k=20`.
- Reuse of state produced by the failing turn makes a later prompt fail when
  the same prompt succeeds from a cold prefill.

The original first-failure boundary had 49,925 prompt tokens:
`49,925 = 780 * 64 + 5`. The next cold request had 49,940 tokens, remainder 20,
and was finite. The complete 24-message request had 49,969 tokens, remainder 49,
and was also finite in a clean namespace. Reusing state written by the failing
request poisoned later prompts. The synthetic modulo result explains all three
observations without relying on transcript content.

OpenCode's `local-b70/qwen38` options now explicitly send the official
`temperature=1`, `top_p=0.95`, `top_k=20` values. A localhost capture of the
actual configured client confirmed those exact snake-case fields on the wire.
The other OpenCode model entries were not changed.

The checkpoint declares BF16 while Compose currently forces FP16. Restoring
BF16 remains desirable as an independent fidelity test, and the draft INT4
helpers now preserve the model dtype for their scales. It is no longer the
leading explanation for this incident: deterministic failures at one exact
64-token remainder, including five-token prompts, are characteristic of the
specialized GDN chunk path rather than general FP16 overflow.

Not yet proven:

- Which operation within the compiled XPU GDN 64-token chunk first becomes
  non-finite.
- Whether rebuilding the same kernel source against current SYCL-TLA corrects
  the partial tail. The installed `0.1.12.x` source pins SYCL-TLA commit
  `cd763790`; XPU-kernels main advances it to `87f68506`. PR 846 in that newer
  revision restores BMG load cache controls and block-height selection which
  already match the old pin, so it is not direct evidence of a fix. A newer
  dependency/compiler A/B remains useful but is no longer the leading change.
- The installed release includes the earlier GDN OOB and SLM-race fixes. The
  later XPU-kernels mixed spec/non-spec fix changes a different execution shape
  and does not modify the failing chunk header.
- Whether the best production correction is a kernel fix or a narrow scheduler
  guard that splits a final `64*N+5` prefill into two safe chunks while keeping
  the four-token MTP lookahead intact.

The detailed static source and version audit is in
[`xpu-gdn-source-audit.md`](xpu-gdn-source-audit.md). The same XE2 GDN source
blob is present in v0.1.12, release/v0.1.13.2 and current main; upgrading the
whole wheel is therefore not a known source fix.

The first narrow scheduler candidate is preserved as the rejected experiment
[`patch_xpu_gdn_tail.py`](../docker/patches/patch_xpu_gdn_tail.py). A controlled
recreate confirmed that it matched and patched the live scheduler while MTP4
remained configured. It did not correct the fault: the fixed-oracle check still
returned finite results at 4 and 6 tokens and the same NaN HTTP 400 at 5. The
mount and invocation were removed and the service was recreated on the prior
configuration. Do not activate this patch in production.

The current vLLM checkout is
`0.27.2rc1.dev77+gac7509e2b` (2026-08-14). Fine-grained hybrid cache primitives
landed only in June/July 2026, followed by multiple cache correctness fixes.
Related upstream reports describe Qwen tool-use sessions degrading to 0% MTP
acceptance and gibberish, MTP/prefix-cache/FP8 corruption, and Qwen endless
`!` output.

## Immediate recovery

For a stuck session, use a new private cache namespace and ensure the newly
rendered prompt does not contain `64*N+5` tokens. Adding text is effective only
if tokenization actually changes the count; the earlier space and `x` controls
merged into existing tokens and did not move the failing remainder.

```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "cache_salt": "NEW-UNGUESSABLE-VALUE"
}
```

This recovered the complete captured session because that rendered prompt had
remainder 49. A new salt alone cannot repair a cold prompt whose length still
has remainder five. It is a containment measure, not the permanent fix: a
static salt can accumulate bad state, and changing it on every turn throws away
the TTFT benefit of prefix caching.

## Next controlled fix gate

1. Keep MTP4, FP8 KV, the target, draft overlay, scheduler budget and 210 W cap.
2. Capture the real first-prefill GDN inputs and metadata for a five-token raw
   request, without retaining response text, then replay them through the fused
   and split direct-operator paths. This closes the gap left by the all-finite
   random-input matrix.
3. Apply only one evidence-backed GDN correction in a derivative image or
   startup patch. Do not reuse the rejected scheduler split unchanged, and do
   not bundle the independent BF16/draft-helper cleanup into this A/B.
4. Run the public 1–128 exhaustive sweep. Every request must return 20/20 finite
   top logprobs.

   ```bash
   ./scripts/gdn_tail_probe.py --lengths 1-128 --expect-fixed
   ```

5. Test 1,669 and 3,333 tokens cold, then reuse each namespace with a known-good
   continuation to prove that recurrent prefix state remains finite.
6. Rerun the cached TTFT, p512/p8192 decode, MTP acceptance and deterministic
   quality gates. MTP must remain enabled and its acceptance must not regress.

The failed maintenance trial and successful rollback gates are recorded in
[`gdn-maintenance-window-2026-08-24.md`](gdn-maintenance-window-2026-08-24.md).
Any future recreate needs a new maintenance authorization. Model data and the
persistent compile cache are not removed by the rollback procedure.

## Upstream references

- [Qwen3.5 MTP multi-turn tool-use gibberish and acceptance collapse](https://github.com/vllm-project/vllm/issues/36872)
- [MTP, prefix caching and FP8 KV tool-call corruption](https://github.com/vllm-project/vllm/issues/50188)
- [Qwen endless exclamation-mark output](https://github.com/vllm-project/vllm/issues/39348)
- [Fine-grained partial cache hits for hybrid/Mamba models](https://github.com/vllm-project/vllm/issues/45702)
- [XPU GDN shared-local-memory race fix](https://github.com/vllm-project/vllm-xpu-kernels/pull/411)
- [XPU GDN out-of-bounds guard](https://github.com/vllm-project/vllm-xpu-kernels/pull/439)
- [XPU-kernels SYCL-TLA update](https://github.com/vllm-project/vllm-xpu-kernels/pull/517)
- [SYCL-TLA Battlemage block-2D load fix](https://github.com/intel/sycl-tla/pull/846)
- [Official Qwen3.8-27B generation configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/generation_config.json)
