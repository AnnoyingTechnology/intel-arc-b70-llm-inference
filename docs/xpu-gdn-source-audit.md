# XPU GDN `64*N+5` source audit

Status on 2026-08-24: the public API reproducer is conclusive, but the exact
faulting instruction inside the compiled XE2 kernel is not. The synthetic
direct-operator matrix was finite and the first scheduler split failed its API
gate, so it was rolled back. No kernel binary has been replaced. This audit was
cross-checked against the upstream source with Claude Opus 5.

## Established boundary

Raw single-sequence target prefills fail only when their scheduled length is
`64*N+5`. Lengths 4/6, 68/70 and 132/134 return finite logprobs; 5/69/133
return HTTP 400 because a logprob is NaN. Sweeps over 1--128 tokens found no
other failure, and five different repeated one-token inputs retained the same
length boundary.

The requests have no chat template, tools, previous messages, cached prefix or
speculative metadata. The first target prefill therefore establishes all of
the following:

- the failure precedes sampling and MTP verification;
- prompt content, repetition penalties and OpenCode are not the cause;
- the active path is `gdn_attention_core_xpu` -> `_xpu_C.gdn_attention` -> the
  XE2 64-token chunk implementation in `libgdn_attn_kernels_xe_2.so`;
- a failed prefill can write non-finite recurrent state that later prefix-cache
  hits reuse.

The deployed Qwen3.8-27B GDN shape is 16 key heads, 48 value heads, 128 key and
value dimensions, convolution width 4 and tensor parallelism 1. Upstream's
default tests use 32 value heads and distribute the larger token cases over 32
sequences. With their fixed seed, none of those sequences has a five-token
chunk tail.

## Version forensics

The live package reports `vllm-xpu-kernels 0.1.12.3`. The XE2 GDN kernel source
blob is identical (`88d232e3893e1693db007b4e06a524df7358f978`) at all of these
refs:

| Ref | XE2 GDN source blob |
|---|---|
| `v0.1.12` | `88d232e3893e1693db007b4e06a524df7358f978` |
| `origin/release/v0.1.13.2` | `88d232e3893e1693db007b4e06a524df7358f978` |
| `origin/qiming/fix_gdn` | `88d232e3893e1693db007b4e06a524df7358f978` |
| `origin/main` | `88d232e3893e1693db007b4e06a524df7358f978` |

The later mixed spec/non-spec patch rewrites the C++ dispatch interface, not
this non-spec prefill kernel. The earlier SLM-race and out-of-range value-head
fixes are already present in the deployed source.

Main advances SYCL-TLA from `cd763790` to `87f68506`. The latter includes PR
846, but that change restores non-Xe3P load cache controls and block-height
selection which already match the old `cd763790` behavior on BMG. The newer
dependency remains a valid binary A/B because other compiler/header changes
exist, but PR 846 is not direct evidence that main fixes this failure.

An official main wheel was downloaded to tmpfs for a future controlled A/B:

| Field | Value |
|---|---|
| Workflow run | `32692290527` |
| Artifact | `9508328924`, `vllm-xpu-kernels--20260824-050903` |
| Commit | `baaa05bb4e92901219a5a072dd63f2474896f6d1` |
| Wheel | `vllm_xpu_kernels-0.1.dev1+gbaaa05bb4-cp38-abi3-manylinux_2_28_x86_64.whl` |
| SHA-256 | `7b886fa814469aef8904118729f31f2fe77559f3c5219bd0ecf799a904387483` |
| Expiry | 2026-09-23 05:43:33 UTC |

The main and installed GDN shared libraries have the same SONAME and dynamic
dependencies. Their low-level exported GDN symbol is unchanged, so replacing
only that library in a derivative image is a feasible dependency/compiler A/B
without mixing the newer Python operator interface into the pinned vLLM image.
This compatibility has been checked statically, not at runtime.

## Source narrowing

For a fresh five-token prefill, the XE2 forward-output kernel sees one chunk
and `has_prev_state == false`. The fused previous-state `W*S`/`Q*S` branch and
the intervening global `u` write/read cannot cause that first failure. At 69
tokens, the failing five-token tail is the second chunk and does have previous
state. The defect must therefore be in tail-dependent work common to both
branches or in an earlier shared stage:

- preparation of the gate cumsum and 64-slot padding;
- `Q*K^T`, triangular masking and the `O2` block-2D store;
- the state update using the partial chunk; or
- a block-2D load/store or register reorder whose surface height is five.

An initial hypothesis blamed unmasked padded rows in the `O2` epilogue. The
neighbor controls falsify that as a root-cause explanation: the same padded
rows exist for four- and six-token tails, a row guard cannot affect any valid
row, and both controls are finite. The guard may be safe as an instrumentation
probe, but it is not a justified fix.

The second source pass established several useful negative results:

- Block-2D bounds are built from the runtime tensor shape; hardware clamps
  partial loads and stores for every tail size.
- Register `reorder()` and the MMA layouts are static permutations independent
  of `current_chunk_size`.
- Gate preparation does transform padding slots, but backward subtraction
  excludes later padding from the last valid cumsum. A magnitude explanation
  is also monotonic in the wrong direction: a four-token tail has more padding
  than the failing five-token tail and is finite.
- The padded inverse block is identity and algebraically decoupled from the
  valid block.

No visible C++ or SYCL-TLA rule found so far distinguishes exactly five rows
from four and six. The remaining leading class is therefore below the
statically visible algorithm: XE2 DPAS/block-2D code generation or a compiler
fast-math scheduling artifact for that exact partial surface. The forward
output kernel is still worth isolating because it is common to fresh and
stateful failures and uniquely uses `sycl::native::exp`, but that is a hardware
A/B hypothesis rather than a root-cause claim.

Secondary source hazards are real but cannot explain the fresh five-token
case: a local-space-only barrier orders a global `u` write before a global read
on the previous-state path, and two kernels dereference `has_initial_state`
without a null guard. They should not be bundled into the first A/B.

## Controlled correction order

1. Capture the real projected tensors and scheduling/GDN metadata from the
   first five-token prefill, then replay them in the direct probe. The random
   synthetic matrix did not reproduce the failure.
2. Apply one evidence-backed correction while keeping MTP4. The existing
   scheduler guard is rejected and must not be activated unchanged.
3. Require the exhaustive public 1--128 sweep to return finite top logprobs,
   then test cold and stateful tails at 1,669 and 3,333 tokens.

   ```bash
   ./scripts/gdn_tail_probe.py --lengths 1-128 --expect-fixed
   ```

4. Rerun cached TTFT, sustained decode, MTP acceptance and the deterministic
   quality gate. Any scheduler-based correction for one of every 64 possible
   prompt-length remainders must not materially change normal throughput.
5. Build compile-time localization probes separately: contain padded lanes in
   the forward-output epilogues, clamp prepared padding-gate magnitude, then
   replace `sycl::native::exp` with `sycl::exp`. These are probes, not claimed
   fixes; each needs its own off/on binary and exact neighbor matrix.
6. Separately A/B the official main GDN shared library. This tests the newer
   build dependency/compiler while holding vLLM and its operator interface
   fixed; source parity makes it a lower-priority probe.

The direct operator test must use a single prefill sequence at the real
16/48/128/128 shape, sweep 4/5/6, 68/69/70 and 132/133/134 in FP16 and BF16,
and assert finiteness after `causal_conv1d`, after `gated_delta_rule`, and in
the recurrent state. The end-to-end API sweep remains the acceptance test
because it uses the checkpoint's real projections and gate parameters.

[`gdn_operator_tail_probe.py`](../scripts/gdn_operator_tail_probe.py) implements
that synthetic fused/split matrix and checks the zero-padding contract between
the two operators. It ran in a disposable XPU container while the serving
container was stopped: all 144 combinations were finite and all padding checks
passed. Real model projections or surrounding runtime state are therefore
needed for the next direct reproducer.

## Rejected scheduler-split audit

Before deployment, the staged patch was reviewed against the live scheduler
and GDN metadata builder, including an explicit Claude Opus 5 pass. Its intended
split does not produce a
`64+5` continuation: it subtracts the four-token lookahead, so 5, 69, 1,669
and the incident's final 773-token step become 1+4, 65+4, 1,665+4 and 769+4.
Every emitted query length is therefore finite under the measured rule.

The four-token second step remains a non-spec prefill. Before the prompt is
complete, the request has no scheduled speculative token IDs; the GDN metadata
builder takes its non-spec branch and classifies every query length greater
than one as prefill. `max_num_seqs=1` also prevents a different request's spec
decode from sharing that batch. This closes the apparent MTP4 classification
risk without disabling MTP.

The finite 68-token control gives direct stateful coverage: its second internal
chunk is a four-token tail with `has_prev_state=true`, reading and writing the
same recurrent-state slot as the patch's second scheduler step. Finite 65 gives
the corresponding stateful one-token-tail coverage for the first step. The
concrete patched scheduler method also executed correctly for eight prompt,
decode and configuration cases against the live source anchor. Those static and
unit-level arguments were insufficient: after a real recreate, the five-token
API request still produced NaN. The experiment does not reveal whether the
guard failed to match in the live request or whether a resulting scheduler step
still mapped to a bad internal GDN shape. Its production efficacy is falsified.

The guard intentionally applies only to the final scheduled prompt chunk.
Intermediate chunks must therefore remain a multiple of 64. Compose now pins
`max_num_batched_tokens` literally to 8,192 instead of allowing an environment
override; any future batch-budget or long-prefill-threshold change must retain
that divisibility or broaden and revalidate the guard.

## Runtime boundary

Applying any candidate requires recreating only `b70-vllm-qwen38`. Expected
impact is about two minutes of inference unavailability and loss of in-memory
prefix entries. Rollback is to remove the candidate mount or restore the
pinned image and recreate the same container; model data and the persistent
compile cache remain untouched.

The 2026-08-24 recreate was completed and rolled back after the first mandatory
gate failed. The running service remains on the pinned package with MTP4 enabled
and without the scheduler patch. See
[`gdn-maintenance-window-2026-08-24.md`](gdn-maintenance-window-2026-08-24.md).
