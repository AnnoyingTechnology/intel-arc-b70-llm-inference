# Historical Fable 5 prompt: Intel Xe2 GDN `64*N+5` NaN

> This file is the evidence dossier supplied to Fable 5, not Fable's answer and
> not the final investigation record. Fable had online/upstream-source access
> but no access to this local repository. Its leading hidden-state/spec-length
> mechanism was later rejected by exact local source inspection and a finite
> poisoned live-layout replay. See the
> [authoritative pause handoff](gdn-64n5-investigation-pause-2026-08-25.md).

You are the last independent reviewer on a difficult numerical/kernel investigation. You have no tools in this session. Everything needed is below. Do not confirm the investigators by default. Reconstruct the execution, identify contradictions or invalid inferences, and try to supply the insight that makes the next experiment decisive.

## Required answer

Give:

1. the strongest diagnosis supported by the evidence, explicitly distinguishing proof from hypothesis;
2. the most likely exact failure mechanism, including component and source construct;
3. one decisive, minimally perturbing experiment, with exact tensors/markers and where each write belongs;
4. the result table that would distinguish the remaining hypotheses;
5. if the evidence is already sufficient for a principled source fix, the smallest upstream-suitable fix and regression test; otherwise say precisely what is missing;
6. any way the existing investigators have fooled themselves.

Do not propose disabling MTP, a scheduler split, a token-length exception, repetition penalties, or a production workaround. Do not ask for files or suggest broad instrumentation/builds. The goal is the underlying defect.

## Hardware and deterministic end-to-end oracle

- Intel Arc B70, Xe2/Battlemage, 32 GiB VRAM.
- vLLM target Qwen3.8-27B, AutoRound W4A16/GPTQ weights, model compute FP16, FP8 KV, MTP4 enabled.
- Raw `/v1/completions`, unique cache salt, one output token plus logprobs.
- Only prompt lengths `64*N+5` fail: 4 finite, 5 HTTP 400 due NaN logprob, 6 finite; 68 finite, 69 NaN, 70 finite; 132 finite, 133 NaN, 134 finite. Exhaustive 1..128 found no other remainder.
- Content-independent across five synthetic one-token units.
- The original real incident had 49,925 prompt tokens = `780*64+5`.
- Pure first target prefill: no decode/spec tokens, `num_actual_tokens=5`, `num_prefills=1`, `num_decodes=0`, `num_spec_decodes=0`, `non_spec_token_indx=None`, `query_start=[0,5]`, `has_initial_state=[false]`, state index 0, runtime XPU graph mode `NONE`.
- Shape: 16 K heads, 48 V heads, K=V=128, convolution width 4, TP1, `reorder_input=true`.
- The first actual bad base-model layer is layer 4, a linear-attention/GDN layer: model-stage persistent buffers show layers 1,2,3 entirely finite, layer-4 entry/input norm finite, and layer-4 attention output entirely NaN. A separately captured MTP layer labelled 0 is not part of this ordering and must be ignored.

## Installed binary provenance and known fixes

- Package reports `vllm-xpu-kernels 0.1.12.3`, but no public `v0.1.12.3` tag exists.
- Exact installed source commit established locally: `e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88`, unpublished packaging from `release/v0.1.12.2`.
- Known fixes present in that source/binary lineage: PR #344 padded-input handling, #411 Xe2 synchronization race, #437 causal-convolution OOB writes, #439 `v_head_id` OOB write.
- #537 mixed speculative/non-speculative handling is absent. A cherry-pick/rebuild of #537 preserved 4-finite/5-NaN/6-finite, so it does not cure this pure-prefill failure.
- The relevant installed GDN kernel source blob is identical through current upstream main.

## Narrow live captures

A persistent, module-registered buffer bank keyed by actual dispatch length 4/5/6 was used. Every device copy is immediately followed by an independent integer marker write, all enqueued in execution order; no host synchronization occurs within the forward. A watcher synchronizes only after the API request and copies the persistent bank to CPU. For the layer-4 n5 dispatch:

- hidden/projection input: all finite, marker 1;
- `projected_states_qkvz`: all 81,920 active FP16 values finite, marker 2;
- `projected_states_ba`: all 480 active FP16 values finite, marker 3;
- post-GDN `z`: all 30,720 active FP16 values finite, marker 4;
- post-GDN `core_attn_out`: all 30,720 active values NaN, marker 5.

The physical projection and output tensors have 5 rows in the actual call; persistent capture buffers have 8 rows but the active five and padding were analyzed separately. A replay tried both physical-5 and padded-8 layouts.

An earlier split-operator persistent capture reported finite projections/states and active q/k/v/b/a all NaN after causal convolution, then NaN core/state. However, later immediate debug tensors allocated inside the compiled forward and copied to CPU sometimes reported the projections themselves as NaN even though the registered persistent projection bank reported them finite. Those newly allocated observer buffers are therefore not trusted: their storage can be affected by graph/allocator reuse, and D2H ordering is observer-sensitive. The claim “causal convolution is first bad” is not accepted without a lower-level marker-stamped capture.

## Exact offline replay and the central contradiction

The exact finite projection bytes, static weights (`conv_weights`, `A_log`, `dt_bias`), zero/finite initial recurrent state, and exact metadata from the real failure were replayed directly on the same B70:

- physical-5 and padded-8;
- fused `gdn_attention` and split causal+delta;
- eager and `torch.xpu.XPUGraph`;
- stock optimized kernels.

Every q/k/v/b/a/z/core value and recurrent-state value was finite.

A second source candidate changed **every prefill**, not n5 specifically:

- Xe2 causal path replaced by the generic/native causal implementation;
- matching generic/native gated-delta implementation used;
- no scheduler or length exception;
- candidate binary SHA-256 `f6abdd7b7ce9e125dcda6a30181927df6003fc77c5878543f2ebea841509b49d`.

Live API with this candidate remained exactly 4 finite / 5 NaN / 6 finite. The service was then restored.

The exact n5 captured projections/state were also replayed directly through **that same reference-path candidate** beside the healthy service. Physical-5/padded-8 and eager/XPU-graph were all entirely finite. Therefore neither the optimized Xe2 causal source nor value-dependent reference GDN mathematics alone explains the live result. The failure requires live integration/storage/dispatch/lifetime context.

Note: this generic path double-applies some transformations relative to the optimized path (the native delta normalizes q/k and applies sigmoid to b even when fed outputs already transformed by another path). It is numerically unsuitable as a fix, but finite input remains finite; its live-vs-replay contradiction is still significant.

Additional A/Bs that did not cure n5: pre-GDN device synchronization, post-GDN synchronization, retaining split intermediate tensors, retaining split input references, and a validated generic/reference delta substitution. Do not assume a missing stream synchronization is proved merely from asynchronous symptoms.

## vLLM Python integration

The model forward is effectively:

```python
projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
projected_states_ba, _ = self.in_proj_ba(hidden_states)
core_attn_out = torch.zeros((num_tokens, 48, 128), dtype=hidden_states.dtype, device="xpu")
z = torch.empty_like(core_attn_out)
torch.ops.vllm.gdn_attention_core_xpu(
    core_attn_out, z, projected_states_qkvz, projected_states_ba, self.prefix
)
core_attn_out = self.norm(core_attn_out.reshape(-1, 128), z.reshape(-1, 128))
out, _ = self.out_proj(core_attn_out.reshape(num_tokens, -1))
```

`gdn_attention_core_xpu` is a Python custom op registered as:

```python
direct_register_custom_op(
    op_name="gdn_attention_core_xpu",
    op_func=eager_break_during_capture(_gdn_attention_core_xpu_impl),
    mutates_args=["core_attn_out", "z"],
    fake_impl=_gdn_attention_core_xpu_fake,
)
```

Its implementation retrieves the layer object and attention metadata through global `forward_context`, then calls `_xpu_C.gdn_attention`. That C++ op also mutates `self.kv_cache[0]` (conv state) and `self.kv_cache[1]` (SSM state), but those tensors are hidden global/module state rather than explicit arguments of the outer Python custom op. The C++ dispatcher schema marks `core_attn_out`, `z`, `conv_state`, and `ssm_state` mutable, while projection inputs are read-only. The SYCL kernels use `c10::xpu::getCurrentXPUStream(...).queue()` through `vllmGetQueue`; this is not an obviously separate queue.

Question this integration carefully: runtime graph mode is `NONE`, but the surrounding model is torch-compiled and this custom op is the deliberate compile boundary. Determine whether hidden mutable state, alias analysis, async lifetime, or compile memory planning can create the exact live-only behavior despite Python references.

## Installed C++ path

`gdn_attention` calls `causal_conv1d`, retains its five returned tensors in a local `std::vector<torch::Tensor>`, then immediately calls `gated_delta_rule` with them. The C++ local references remain alive until the wrapper returns.

For n4/n5/n6, `conv1d_tile_size=8`, so all use the **untiled** causal kernel. For 69/133 the causal implementation is tiled. Gated delta uses 64-token chunks in the optimized path, making `64*N+5` map exactly to a final delta chunk of five.

Untiled causal output allocation for one sequence adds 63 virtual rows: q/k/v/b/a are zero-initialized with `non_spec_token + 63` rows/columns. Only the active tokens are written. It launches separate kernels:

1. `chunk_causal_conv1d_kernel`: produces q/k/v from convolution and optionally fused q/k L2 normalization;
2. `chunk_reorder_zba_kernel`: independently produces z/b/a from the same projection inputs.

For model dimensions, the untiled q/k/v workgroup has 160 work-items = five 32-lane subgroups. The fused L2 path computes q and k subgroup reductions, lane 0 writes two FP32 sums per subgroup into local memory, executes `sycl::group_barrier(item.get_group())`, every thread sums all five subgroup slots, then applies q/k reciprocal square roots. v lanes participate in the barrier but contribute zero q/k sums. There is no second barrier because local memory is read-only afterward. This is the only conspicuous environment-sensitive construct in the untiled q/k/v kernel, but it cannot explain 69/133 because those causal calls are tiled.

The prior “missing token-end guard” observation is a real defensive source gap, but not reachable here: the n5 launch grid contains exactly token IDs `[0,5)`, equal to the last query endpoint. Do not use it as this root cause.

The optimized delta kernel computes:

```cpp
current_chunks = ceil(seq_len / 64)
current_chunk_size = min(64, seq_len - chunk_id*64)
g_last = a[chunk_offset + current_chunk_size - 1]
barrier(local_space)
for e < current_chunk_size:
    g_slm[e] = a[chunk_offset+e]
    g_multi_slm[e] = native_exp(g_last - a[chunk_offset+e])
    g_exp_slm[e] = native_exp(a[chunk_offset+e])
for e in [current_chunk_size,64): set all three SLM arrays to zero
barrier(local_space)
```

It then uses SYCL-TLA/DPAS block-2D loads, triangular masking, a 64x64 inverse/update and partial-surface output/state stores. Static source inspection found no visible branch unique to size five, but this is the one algorithmic path shared by every `64*N+5` failure.

## Existing independent Opus 5 assessment

Opus independently rejected the previous causal-first claim as unproved. It ranked:

1. same-length dispatch/input identity or live allocation handoff;
2. live-only untiled q/k/v fused-L2/SLM fault for n5;
3. optimized delta final-five-chunk defect as the best explanation of the complete modulo rule.

It recommends wiring existing optional in-op `debug_q/debug_k/debug_v/debug_b/debug_a/debug_marker` arguments. The C++ wrapper currently copies each causal output into device debug buffers after `causal_conv1d` and before `gated_delta_rule`, then stamps shape/dispatch markers; no host sync is needed. Opus also asks for a first-statement device copy of qkvz/ba and a dispatch ID so same-length internal calls cannot be confused.

Challenge that design too: a debug `copy_` may add dependencies or perturb allocator timing; debug buffers allocated within the compiled forward have already been unreliable. If persistent module-registered buffers passed as explicit custom-op arguments or another mechanism is required, state it exactly. The desired experiment must identify the first failing operation without merely hiding a race.

## Constraints on the next move

- One decisive n5 experiment; n4/n6 only controls. Use 69/133 only if indispensable to distinguish tiled causal from final-five delta.
- No broad instrumentation or full rebuild churn. A focused existing `_xpu_C` incremental build is available.
- Every capture write needs a marker and dispatch identity. Capture the narrowest exact tensors/metadata.
- Restore the known working MTP4 service after a failed diagnostic.
- Patch only after a concrete defect is reproduced. Any patch must be principled, minimal, regression-tested, and upstream-suitable.
- Do not submit upstream.

Now reason from first principles. Prefer one strong conclusion and one experiment over a menu.
