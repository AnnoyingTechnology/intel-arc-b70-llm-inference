# Claude Opus 5 independent GDN audit — 2026-08-25

This records Claude Code Opus 5's independent source review before its local
session quota was exhausted. It is an audit checkpoint, not a final diagnosis.
Its proposed untiled causal-convolution defect was not established as the cause
of the `64*N+5` failure; later trusted capture and replay are summarized in the
[authoritative pause handoff](gdn-64n5-investigation-pause-2026-08-25.md).

## Corrections to the first audit

- The retained-tensor dictionary is a valid storage-lifetime probe. Holding the
  first `Tensor` for a pointer tuple pins that storage, while a new pointer tuple
  creates another retained entry. The earlier objection was withdrawn.
- `replay_gdn_multilayer_graph.py` does create fresh graph cases after a
  separate eager warm-up. The earlier statement that it reused warm tensors was
  withdrawn.
- The split retention probe still cannot pin temporaries allocated and destroyed
  entirely inside C++, including `conv_states_tmp` and gated-delta-rule private
  temporaries. Its negative result therefore does not falsify private-temporary
  lifetime problems.

## Independent source findings

Opus identified an ambiguity in the first persistent-stage captures: floating
capture buffers were initialized to NaN, so an all-NaN dump could mean either
that a kernel produced NaN or that the graph never executed the intended copy.
It also noted that the original analyzer compared the physical `n+63` rows for
q/k/v/b/a but only active rows for z. Those measurements were not comparable.

Opus found a concrete source defect in the untiled Xe2 causal-convolution
kernel: `chunk_causal_conv1d_kernel` has no guard when `token_id` is at or past
the last `query_start_loc` endpoint. Such a workgroup falls through with the
last batch selected, sees a zero-length sequence, takes the decode-state update
branch, and may write three rows of a live convolution-state slot from an
out-of-request token. The tiled Xe2 kernel contains an explicit guard. This is
a real correctness defect, but Opus cautioned that it did not yet explain the
entire observed `64*N+5` API rule because long requests use the tiled prefill
path.

Opus could not derive the `64*N+5` rule from the kernel source. The 144-case
standalone operator matrix remaining finite argues against a pure shape/codegen
defect. It recommended proving capture writes with explicit markers and
capturing neighbouring 4/6 controls before compiling a diagnostic kernel.

## Evidence received after the audit

A shared post-request buffer labelled for the five-token API request contained
`query_start_loc=[0,4]`. This proved that multiple internal MTP graph replays
overwrite a single persistent buffer; the dump was not necessarily the
five-token dispatch. Capture banks keyed by the actual `num_actual_tokens`,
with explicit write markers, were therefore introduced. The resulting evidence
must supersede the ambiguous shared-buffer results in the final diagnosis.

Claude Code could not write this file itself because its session quota was
exhausted after composing the audit. This document preserves its reported
independent conclusions; a final patch review still requires a fresh Opus 5
session after the quota resets.
