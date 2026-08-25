# Deterministic all-NaN GDN output for single-sequence prefill lengths `T % 64 == 5` on Xe2/Battlemage

Submitted as [`vllm-project/vllm-xpu-kernels#548`](https://github.com/vllm-project/vllm-xpu-kernels/issues/548).

> **AI disclosure:** OpenAI Codex GPT-5.6-sol drafted this report after conducting
> the supervised investigation. Anthropic Claude Code Opus 5 and Fable 5 were
> used as independent adversarial reviewers. The hardware owner reviewed and
> approved the facts and submission.

## Summary

On an Intel Arc Pro B70 (Xe2/Battlemage), Qwen3.8-27B raw completion requests
produce all-NaN logprobs deterministically when the initial single-sequence
prefill length satisfies `T % 64 == 5`. Neighboring lengths are finite.

The first trusted non-finite boundary is the output of the layer-4 fused GDN
operation. The operation receives finite hidden/projection inputs and produces
finite `z`, while every active `core_attn_out` value becomes NaN. Exact captured
operands replay finitely through the same operators outside the live compiled
model forward. This localizes the problem to a live XPU
integration/code-generation/runtime interaction without attributing it to a
specific kernel instruction, compiler, Level Zero, or driver defect.

## Reproducer

The probe is public and does not contain user data:

```bash
git clone https://github.com/AnnoyingTechnology/intel-arc-b70-llm-inference.git
cd intel-arc-b70-llm-inference
./scripts/gdn_tail_probe.py --lengths 4-6,68-70,132-134
./scripts/gdn_tail_probe.py --lengths 1-128
```

It targets an OpenAI-compatible endpoint at
`http://127.0.0.1:19622/v1/completions` by default. Each request uses prompt
`" A" * T`, a unique `cache_salt`, one output token, and 20 logprobs. Response
text is not printed.

Observed:

| Prompt tokens | Result |
|---:|---|
| 4 | 20/20 finite top logprobs |
| 5 | HTTP 400: strict JSON serialization encounters `nan` |
| 6 | 20/20 finite top logprobs |
| 68 | finite |
| 69 | NaN |
| 70 | finite |
| 132 | finite |
| 133 | NaN |
| 134 | finite |

An exhaustive 1--128 sweep found no other failing remainder. The failure also
reproduces at 49,925 tokens (`780 * 64 + 5`) and is independent of prompt
content across five synthetic one-token units.

## Failure context

The five-token failure is a pure initial target prefill:

```text
num_actual_tokens=5
num_prefills=1
num_decodes=0
num_spec_decodes=0
non_spec_token_indx=None
query_start_loc=[0, 5]
has_initial_state=[false]
state_index=0
runtime XPU graph mode=NONE
```

MTP4 is configured on the server, but the failure occurs before speculative
verification. GDN shape is 16 K heads, 48 V heads, K=V=128, convolution width
4, TP1, with reordered input.

## Trusted producer-side capture

The capture uses module-registered persistent buffers and a separate custom
operator that explicitly declares the buffers as mutated arguments. Every
copy has an independent marker. Buffers are keyed by actual dispatch length so
later internal graph replays cannot overwrite the five-token record. The host
reads only after the request and an XPU synchronization.

GDN layers 0, 1, and 2 remain finite. At layer 4:

| Boundary, five active rows | Finite values |
|---|---:|
| hidden input | 25,600 / 25,600 |
| projected `qkvz` | 81,920 / 81,920 |
| projected `ba` | 480 / 480 |
| post-GDN `z` | 30,720 / 30,720 |
| post-GDN `core_attn_out` | **0 / 30,720** |

The physical tensors have five rows. In the eight-row persistent capture bank,
the three padding rows of `core_attn_out` remain finite; exactly all five active
rows are NaN.

This locates the first trusted invalid value at `core_attn_out` after the fused
GDN call. It does not prove which internal GDN sub-operation first fails.

## Exact replay and negative controls

Exact captured layer-4 projections, weights, metadata, and state replay
finitely on the same B70 through:

- physical-five and padded-eight layouts;
- fused GDN and split causal-convolution/delta calls;
- eager and `torch.xpu.XPUGraph` execution;
- optimized and generic/reference implementations;
- the exact live shared/as-strided hybrid-cache layout.

The live recurrent-state layout is:

```text
conv_state: [146,7,10240], stride [1703936,10240,1], FP16, offset 0
ssm_state:  [146,48,128,128], stride [851968,16384,128,1], FP32,
            storage offset 35840
shared raw storage pitch: 3,407,872 bytes per block
```

Poisoning the active state storage with `0xFF` remains finite in the fresh
prefill replay with `has_initial_state=false`.

The n5 live failure is also unchanged by pre/post-GDN synchronization,
retaining split intermediates and inputs, routing all prefills through the
generic/native causal and delta paths, or backporting #537. A scheduler split
was tested separately, failed its end-to-end gate, and was removed.

Some lower-level debug-copy experiments perturbed or failed to observe the
compiled execution and are deliberately excluded from this report.

## Installed package provenance

The installed package reports `vllm-xpu-kernels 0.1.12.3`; upstream has no
public `v0.1.12.3` tag. Local provenance resolves the installed source to:

```text
e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88
```

This is unpublished packaging from the `release/v0.1.12.2` lineage. The
installed source contains the relevant fixes from:

- #344: padded-input handling;
- #411: Xe2 synchronization race causing NaNs;
- #437: causal-convolution out-of-bounds writes;
- #439: `v_head_id` out-of-bounds write.

#537 mixed speculative/non-speculative handling is absent, but a controlled
backport/rebuild preserved 4-finite / 5-NaN / 6-finite. The relevant GDN source
is otherwise identical through current upstream main. We therefore cannot
currently attribute this to a missing known-fix backport.

## Environment

```text
GPU: Intel Arc Pro B70, 32 GiB, PCI 8086:e223
kernel/Xe driver: 7.1.8+deb14.1-amd64
container: vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
local image ID: sha256:68c925c6142e55c429d8ff37c4e1531f9ea50039935d54487bc4ec9f43671f18
vLLM: 0.27.2rc1.dev77+gac7509e2b
PyTorch: 2.13.0+xpu
vllm-xpu-kernels package: 0.1.12.3
vllm-xpu-kernels source: e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88
target: Frozenlock Qwen3.8-27B AutoRound/GPTQ W4A16, FP16 compute
KV cache: FP8, 8.5 GB, SHA-256 prefix caching
speculation: MTP4
max sequences: 1
max batched tokens: 8,192
max model length: 196,608
```

The immutable service definition is
[`docker/compose.yaml`](https://github.com/AnnoyingTechnology/intel-arc-b70-llm-inference/blob/main/docker/compose.yaml).
The complete evidence trail, capture hashes, replay matrix, and rejected
hypotheses are in
[`docs/gdn-64n5-investigation-pause-2026-08-25.md`](https://github.com/AnnoyingTechnology/intel-arc-b70-llm-inference/blob/main/docs/gdn-64n5-investigation-pause-2026-08-25.md).

## Artifact hashes

The exact synthetic capture files are retained and can be uploaded on request:

```text
df984ad578bb60dd71b13f4cc50fdd08c4a00db32bd2055e49f413982055e8a7  immediate-layer4-through-gdn-n5.pt
2caec82998886fc916e275f2553abfc6e530f1568b36a6b09303aa799671e1c8  immediate-projections-n4-n5-n6.pt
28c8d24d84b3e9b293dc7027973f3192fb06a353e7161b6980fbf14000693b3a  layer-0-stage.pt
132cf90b8c9563fa37ba09418d2d14885e513ca142d0f290dc431e61542bd944  layer-1-stage.pt
61155b393e335c076602c43be2a165d1593b0514d386566a1479e525127cef1a  layer-2-stage.pt
3cf4aa33450ae4f1e4b422b081dd5924dcf3227f0fe2889b9d4c652e192e0bde  layer-4-stage.pt
```

## Requested help

Please involve Intel/XPU maintainers to identify a non-perturbing internal GDN
capture and determine whether this matches a known Xe2 compiler/runtime issue.
We can run focused builds and validate proposed patches on the affected B70.
Current evidence supports neither a length-specific workaround nor a specific
source fix.
