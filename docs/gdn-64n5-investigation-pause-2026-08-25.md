# XPU GDN `64*N+5` investigation pause

Status: root-cause work paused on 2026-08-25 after approximately 24 hours of
controlled experiments. The stock kernel and scheduler are restored. A
validated one-token prompt guard is deployed with MTP4 retained; it is
containment, not a source correction.

This is the authoritative handoff. It separates trusted observations from
hypotheses and supersedes the older "next step" sections in the incident and
source-audit notes.

## Decision

The failure is localized well enough for an actionable
`vllm-project/vllm-xpu-kernels` issue and Intel/XPU maintainer involvement. It
is not localized enough to claim a particular kernel instruction, compiler,
Level Zero, or driver defect, and no source patch is proposed.

Submitted upstream as
[`vllm-project/vllm-xpu-kernels#548`](https://github.com/vllm-project/vllm-xpu-kernels/issues/548).

Further local work is paused. The only unfinished experiment is a narrower
producer-queue capture described below. It must not delay external
investigation and must not be represented as evidence until validated.

## Deterministic end-to-end oracle

The privacy-safe probe is [`scripts/gdn_tail_probe.py`](../scripts/gdn_tail_probe.py).
It sends a raw synthetic completion request, gives every request a unique
prefix-cache namespace, requests one output token and 20 logprobs, and never
prints generated text.

```bash
./scripts/gdn_tail_probe.py --lengths 4-6,68-70,132-134
./scripts/gdn_tail_probe.py --lengths 1-128
```

Those commands express the historical failure oracle. With the active
containment, use `--expect-tail-guard`.

Observed on the pinned service:

- `prompt_tokens % 64 == 5` returns HTTP 400 because the logprob response
  contains `nan`;
- 4/6, 68/70, and 132/134 return 20/20 finite top logprobs;
- exhaustive sweeps over 1--128 tokens found no other failing remainder;
- 5, 69, 133, and the original 49,925-token incident all reproduce;
- five different synthetic one-token units retained the same length rule.

The failure is in the initial target prefill. For the five-token dispatch the
runtime metadata was:

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

MTP4 remains configured, but speculative verification has not begun at the
failure point. Repetition penalties, sampler settings, OpenCode, and generated
text are downstream of the defect.

## Trusted live boundary

The trustworthy capture uses module-registered persistent buffers, a separate
explicit custom operator declaring those buffers as mutated arguments, and a
write marker after every copy. The host watcher synchronizes and reads the
buffers only after the API request. Capture banks are keyed by the actual
dispatch length, preventing later MTP graph replays from overwriting the
five-token record.

Model-stage captures show GDN layers 0, 1, and 2 remaining finite. At GDN layer
4 the input and both projections are finite; `z` remains finite after GDN, but
every active value of `core_attn_out` is NaN:

| Layer-4 boundary, five active rows | Finite values |
|---|---:|
| hidden input | 25,600 / 25,600 |
| projected `qkvz` | 81,920 / 81,920 |
| projected `ba` | 480 / 480 |
| post-GDN `z` | 30,720 / 30,720 |
| post-GDN `core_attn_out` | **0 / 30,720** |

The physical live tensors have five rows. The persistent bank has eight rows;
all 18,432 padding values in `core_attn_out` remain finite, while exactly the
five active rows are NaN. `core_attn_out` is therefore the first trusted
non-finite boundary, not necessarily the first failing instruction inside the
fused operation.

## Replay and negative controls

The exact captured layer-4 projections, weights, metadata, and recurrent state
replay finitely outside the live compiled model forward. The matrix includes:

- physical-five and padded-eight layouts;
- fused `gdn_attention` and split causal-convolution/delta calls;
- eager execution and `torch.xpu.XPUGraph`;
- the stock optimized implementation and a generic/reference-path candidate;
- the exact shared hybrid-cache layout from the live execution.

The live state layout was reproduced as:

```text
conv_state: shape=[146,7,10240], stride=[1703936,10240,1], FP16, offset=0
ssm_state:  shape=[146,48,128,128], stride=[851968,16384,128,1], FP32,
            storage_offset=35840
shared raw storage pitch: 3,407,872 bytes per block
```

Filling the active live-layout convolution and SSM state with `0xFF` still gave
a finite fresh five-token replay when `has_initial_state=false`. This rejects
the proposed explanation that a prefill path uniquely reads poisoned state at
length five.

The following live experiments also preserved the exact 4-finite / 5-NaN /
6-finite result:

- synchronize immediately before or after GDN;
- retain split intermediates and input references;
- route all prefills through the generic/native causal and delta paths;
- cherry-pick and rebuild mixed speculative/non-speculative fix #537;
- split the scheduler around the five-token remainder.

The scheduler candidate was rejected and rolled back. The reference path was a
diagnostic, not a numerically acceptable fix. These results do not identify a
particular runtime layer; they establish a live compiled-forward interaction
that the standalone replay does not reproduce.

## Installed provenance and known fixes

The installed wheel reports `vllm-xpu-kernels 0.1.12.3`, although upstream has
no public `v0.1.12.3` tag. Local source and binary provenance resolves it to:

```text
vllm-xpu-kernels source commit:
e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88
lineage: unpublished packaging from release/v0.1.12.2
```

The installed lineage contains the relevant changes from:

- #344, padded-input handling;
- #411, Xe2 synchronization-race correction;
- #437, causal-convolution out-of-bounds correction;
- #439, `v_head_id` out-of-bounds correction.

#537 is absent from the installed source, but its controlled backport did not
change this pure-prefill failure. The relevant GDN source remains identical in
current upstream main. This is therefore not presently explained as a missing
known-fix backport.

## Runtime and model

| Field | Value |
|---|---|
| GPU | Intel Arc Pro B70, 32 GiB, PCI `8086:e223` |
| Linux/Xe driver | `7.1.8+deb14.1-amd64` |
| Container image | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` |
| Local image ID | `sha256:68c925c6142e55c429d8ff37c4e1531f9ea50039935d54487bc4ec9f43671f18` |
| vLLM | `0.27.2rc1.dev77+gac7509e2b` |
| PyTorch | `2.13.0+xpu` |
| Target | Frozenlock Qwen3.8-27B AutoRound/GPTQ W4A16 |
| Target revision | `b4c61732c4f2d8af323d75ba5702b5c7f3361539` |
| MTP sidecar revision | `e28c5f952bdd5d814297a07d85a064a87af26a3f` |
| Serving | FP16 compute, FP8 KV, MTP4, prefix caching, TP1 |
| GDN shape | 16 K heads, 48 V heads, K=V=128, convolution width 4 |

The exact Compose command and immutable image pin are in
[`docker/compose.yaml`](../docker/compose.yaml); model revisions and generated
overlay hashes are in [`architecture.md`](architecture.md).

## Artifact manifest

These artifacts are synthetic and contain no OpenCode transcript. The minimal
32 MiB replay set and complete paused source-tree diff are preserved under the
Git-ignored `/home/julien/Documents/B70/.artifacts/gdn-64n5/`; the original
`/tmp` copies are no longer the only copy.

| Artifact | SHA-256 |
|---|---|
| `/tmp/b70-projection-decisive-20260825/immediate-layer4-through-gdn-n5.pt` | `df984ad578bb60dd71b13f4cc50fdd08c4a00db32bd2055e49f413982055e8a7` |
| `/tmp/b70-projection-decisive-20260825/immediate-projections-n4-n5-n6.pt` | `2caec82998886fc916e275f2553abfc6e530f1568b36a6b09303aa799671e1c8` |
| layer-0 stage capture | `28c8d24d84b3e9b293dc7027973f3192fb06a353e7161b6980fbf14000693b3a` |
| layer-1 stage capture | `132cf90b8c9563fa37ba09418d2d14885e513ca142d0f290dc431e61542bd944` |
| layer-2 stage capture | `61155b393e335c076602c43be2a165d1593b0514d386566a1479e525127cef1a` |
| layer-4 stage capture | `3cf4aa33450ae4f1e4b422b081dd5924dcf3227f0fe2889b9d4c652e192e0bde` |

The capture/replay sources retained with this handoff are:

- `docker/patches/patch_layer4_projection_decisive.py`;
- `docker/patches/patch_gdn_stage_capture.py`;
- `docker/patches/patch_model_stage_capture.py`;
- `scripts/analyze_decisive_gdn.py`;
- `scripts/analyze_gdn_stage_capture.py`;
- `scripts/analyze_model_stage_capture.py`;
- `scripts/replay_layer4_causal_conv.py`;
- `scripts/replay_layer4_projection.py`.

## Observer-invalid branch

Several attempts copied intermediates from inside the outer GDN custom
operator into newly added debug outputs. They contradicted the trusted
producer-side capture by reporting already-NaN projections in the same
dispatch. Initializing with finite sentinels, checking disjoint pointers, and
adding a post-operation synchronization did not make those copies trustworthy.
Expanding the outer custom operator to 32 mutated arguments also changed
compiled memory planning and moved the apparent failure upstream.

Those artifacts are observer-perturbed or observer-invalid. They do not support
a claim that causal convolution, a hidden state read, or the projections are
the first failing operation, and they must not be attached as primary evidence
to the upstream issue.

## Independent review

Claude Code Opus 5 was used repeatedly as an adversarial source and experiment
reviewer. It challenged capture lifetime, graph overwrite, padding analysis,
and custom-op mutation contracts. Its valid objections led to length-keyed
persistent banks and explicit write markers. Its last assessment agreed that
the producer-side layer-4 boundary is trustworthy and the deeper debug-copy
branch is not.

Fable 5 was run manually with online and upstream source access but without
access to this local repository. Its strongest proposal was a hidden-operand
read selected by `num_speculative_tokens + 1 == 5`. Local static inspection
found no such count-based routing in the exact installed source, and the
poisoned live-layout replay remained finite, so that mechanism is rejected.
Its useful contribution was identifying the hidden state layout as a replay
gap that needed to be closed.

Neither reviewer supplied a source-level root cause that survived the local
evidence. This is recorded to prevent future investigators from treating an
independent hypothesis as a confirmed diagnosis.

## Validated containment after the pause

`docker/patches/patch_gdn_prompt_padding.py` runs after rendering and changes
only token prompts whose final length is `64*N+5`. Raw completions append one
tokenizer-verified space token. Chat prompts insert it before the final
`<|im_end|>`, so the assistant-generation prefix remains unchanged. Earlier
tokens and `cache_salt` are preserved, retaining aligned prefix-cache reuse.

Validation on the real service:

- exhaustive raw 1--128: 128/128 matched, no NaNs;
- 4/5/6, 68/69/70, 132/133/134: all finite, affected lengths reported one
  additional inference token;
- exact chat canary at rendered 68/69/70: `OK` at all three lengths;
- guarded tool selection and guarded tool-result ingestion both correct;
- unaffected five-family p512/g128: 92.48 tok/s median, 0.383 s median TTFT.

An initial version appended the token after the assistant-generation marker.
Although finite, it caused an immediate empty stop at the guarded chat length.
The exact-output canary rejected that placement. It was never accepted as the
final configuration.

## Latest published stack test (2026-08-26)

The first external contributor to issue #548 reported no reproduction on a
newer stack and suggested the defect might have been silently fixed. We tested
the newest published components available at the time, without the prompt
guard:

| Component | Tested value |
|---|---|
| vLLM nightly image | `vllm/vllm-openai-xpu:nightly@sha256:5f417989045f2e16379bbc6975410edb22d2c86f107577c3507a1359425e7eb1` |
| vLLM | `0.26.1rc1.dev1219+g46638857f.xpu` |
| XPU-kernels source | `a397c58eb7781e6fe0d6b3fb7c25d21b5f658784` (then-current `main`) |
| XPU-kernels CI wheel | full-config `vllm_xpu_kernels-0.1.dev1+ga397c58eb-cp312-cp312-linux_x86_64.whl` |
| XPU-kernels wheel SHA-256 | `99e0560ee1afea40f320feef8a809e131ae8a60565f76bc5455961a4227e4ddf` |
| PyTorch | `2.13.0+xpu` |
| Derived local image | `sha256:8ff7dc99d59fd056579bfa096efecf604f4e25a5a46acdb0842ac6c9bf2a63ec` |

The wheel's development-version spelling differs from the contributor's
`0.1.14.dev16+ga397c58`, but the source commit is exact and the tested artifact
is upstream CI's full-config build.

Results:

- unguarded 4/5/6, 68/69/70 and 132/133/134 reproduced finite/NaN/finite;
- unguarded 1--128 matched `T % 64 == 5` exactly, with no unexpected length;
- the 1--128 JSONL is retained under Git-ignored
  `.artifacts/latest-stack/gdn-tail-1-128.jsonl`, SHA-256
  `07cbeea1f48900087dc46a5190dd5a7f5e153dd29f89e4571d597c5943e6f85d`.

This disproves the proposed explanation that upgrading XPU-kernels alone has
silently fixed the defect. Another B70 contributor independently reproduced it
on the same revisions. Their still-unaudited Fable 5 analysis points to vLLM
graph dispatch rather than kernel arithmetic; that is consistent with the live
versus replay boundary, but remains a hypothesis rather than a validated root
cause.

## Exact pause point

The only unfinished focused experiment replaces five diagnostic ATen `copy_`
calls in the GDN wrapper with `vllmGetQueue().memcpy`, without changing an
inference kernel. Its purpose is to determine whether the invalid observer was
using a copy queue not ordered with the producer queue. The reduced kernel
build unexpectedly began reconfiguring a oneDNN dependency and was stopped.

The candidate build is incomplete and must not be deployed. If local work
resumes, first confirm the stock service is healthy, then decide whether this
single capture is worth completing in parallel with the upstream issue. Do not
resume broad instrumentation or another scheduler workaround.

## Service state at pause

```text
container: b70-vllm-qwen38
image/kernel/scheduler: pinned stock versions above
state: stopped intentionally; not classified as healthy
MTP: 4 speculative tokens enabled
available containment: post-tokenization one-token prompt guard (not a fix)
diagnostic mounts: none
scheduler workaround: absent
latest candidate: b70-vllm-qwen38-latest, stopped; XPU-kernels a397c58
```
