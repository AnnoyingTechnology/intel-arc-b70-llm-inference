# XPU GDN `64*N+5` source audit

Status on 2026-08-24: the public API reproducer is conclusive, but the exact
faulting instruction inside the compiled XE2 kernel is not. No kernel binary
has been replaced and the live service has not been restarted. This audit was
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

The leading source hypothesis is the `O2` epilogue in
`chunk_gated_delta_rule_kernels_xe2.hpp`: it evaluates the complete 64-row MMA
fragment but masks only `m < n`; it does not explicitly zero
`m >= current_chunk_size` before `reorder()` and the bounded block-2D store.
That makes correctness depend on padded DPAS rows and the five-row block-2D
surface being discarded perfectly. Adding the missing row guard is a narrow
compile-time experiment with no effect on complete chunks or MTP.

This is not yet the proven instruction. In particular, source inspection alone
does not explain why the neighboring four- and six-row surfaces are finite.
The exact five-row invariant makes a surface/layout interaction more plausible
than ordinary numerical overflow. The kernel needs stage-by-stage finiteness
instrumentation or isolated patched binaries to promote the hypothesis to a
root cause.

Secondary source hazards are real but cannot explain the fresh five-token
case: a local-space-only barrier orders a global `u` write before a global read
on the previous-state path, and two kernels dereference `has_initial_state`
without a null guard. They should not be bundled into the first A/B.

## Controlled correction order

1. Enable the staged scheduler guard, which keeps MTP4 and changes only the
   bad final prefill shape from `64*N+5` to `64*N+1` followed by four tokens.
2. Require the exhaustive public 1--128 sweep to return finite top logprobs,
   then test cold and stateful tails at 1,669 and 3,333 tokens.
3. Rerun cached TTFT, sustained decode, MTP acceptance and the deterministic
   quality gate. The extra scheduler step applies to one of every 64 possible
   prompt-length remainders and must not materially change normal throughput.
4. Separately A/B the official main GDN shared library. This tests the newer
   build dependency/compiler while holding vLLM and its operator interface
   fixed; source parity means it is lower priority than the scheduler guard.
5. Build the explicit padded-row kernel guard and a stage-localizing unit test.
   If it fixes the exact matrix, prefer and upstream that kernel correction,
   then remove the scheduler workaround.

The direct operator test must use a single prefill sequence at the real
16/48/128/128 shape, sweep 4/5/6, 68/69/70 and 132/133/134 in FP16 and BF16,
and assert finiteness after `causal_conv1d`, after `gated_delta_rule`, and in
the recurrent state. The end-to-end API sweep remains the acceptance test
because it uses the checkpoint's real projections and gate parameters.

## Runtime boundary

Applying any candidate requires recreating only `b70-vllm-qwen38`. Expected
impact is about two minutes of inference unavailability and loss of in-memory
prefix entries. Rollback is to remove the candidate mount or restore the
pinned image and recreate the same container; model data and the persistent
compile cache remain untouched.

No recreate is authorized by this document. The running service remains on
the pinned package with MTP4 enabled.
