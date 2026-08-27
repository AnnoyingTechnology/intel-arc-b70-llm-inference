# Xe2 head-256 prefill-attention tuning — 2026-08-27

## Outcome

The retained candidate doubles the Xe2 head-256 FlashAttention K tile from 32 to 64 while leaving the Q tile, subgroup layout, output tile, model, quantization, FP8 KV cache, scheduler and MTP4 unchanged.

```diff
 struct chunk_policy_head256 {
-  using ShapeQK = Shape<_256, _32, _32>;
-  using ShapePV = Shape<_256, _32, _32>;
+  using ShapeQK = Shape<_256, _64, _32>;
+  using ShapePV = Shape<_256, _32, _64>;
   using ShapeOut = Shape<_256, _256>;
   using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
 };
```

The reproducible source change is [`vllm-xpu-kernels-xe2-head256-k64.patch`](../docker/patches/vllm-xpu-kernels-xe2-head256-k64.patch).

## Scope: no capability trade

This work deliberately excludes weight, activation, KV-cache and output-head quantization changes. The retained candidate performs the same FP16-query/FP8-KV attention with a different workgroup tile. It can change floating-point reduction order, so bitwise identity is not promised; reference comparisons remained far inside the existing tolerance.

Target-head FP8/INT4 was identified as the only obvious way to reduce the decode path's near-saturated memory traffic. It changes verified logits and is therefore outside this optimization scope. No lossy candidate was built or tested.

## Exact runtime

- B70 power cap: 210 W
- vLLM: `46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3`
- `vllm-xpu-kernels`: `a397c58eb7781e6fe0d6b3fb7c25d21b5f658784`
- target: Frozenlock Qwen3.8-27B GPTQ INT4
- draft: same-base BF16 MTP sidecar with the existing draft-only INT4 overlay
- FP8 KV cache, MTP4, 8,192-token scheduler, XPU graph
- attention geometry: 24 Q heads, 4 KV heads, head dimension 256
- live cache page: 1,664 tokens, divisible by both tested K tiles

Qwen has 16 full-attention target layers; the MTP layer adds a seventeenth attention invocation during prefill.

## Results

The microbenchmark replays the exact live cache layout and scheduler chunks. Each figure is the sum of per-chunk medians multiplied by 17 attention invocations.

| Cold-prefill shape | Stock K32 | Retained K64 | Attention gain |
|---|---:|---:|---:|
| 8K | 0.408898 s | 0.404615 s | 1.05% |
| 32K | 5.749366 s | 5.383578 s | 6.36% |
| 100K | 53.377168 s | 48.267177 s | 9.57% |

The long chunks rise from about 39.3 to 43.6–43.7 TFLOP/s. The advantage grows with context because attention's quadratic work increasingly dominates fixed model work.

Controlled full-model results used unique cold prefixes, the same 617,000-character fixture, MTP4, FP8 KV and the 210 W cap:

| Request | Stock | K64 | End-to-end gain |
|---|---:|---:|---:|
| Warmed-graph 32K TTFT | 24.8985 s | 24.5801 s | 1.28% |
| 100K TTFT | 115.1867 s | 111.4571 s | 3.24% |
| 100K prompt rate | 867.16 tok/s | 896.21 tok/s | 3.35% |
| 100K prefill energy | 0.241945 J/token | 0.234081 J/token | 3.25% |

The 100K candidate saves 3.73 seconds. Attention itself reaches the requested near-10% gain, but the whole model does not: W4A16 GEMMs and GDN are unchanged.

## Correctness gates

- The upstream FP8 paged-attention test passed for head dimension 256, block size 64, FP16 query, FP8 KV, paged causal attention and no sliding window.
- [`validate_xe2_prefill_attention.py`](../scripts/validate_xe2_prefill_attention.py) passed exact 24Q/4KV/head-256/page-1664 cases `(q=5,kv=65)`, `(49,1665)` and `(257,4097)` against a PyTorch reference.
- Maximum absolute differences were `0.000801086`, `0.000244141` and `0.000152588`; the established absolute and relative tolerances are `0.015`.
- Full-model 32K and 100K requests returned finite outputs; MTP decode continued to function.
- After promotion, all seven deterministic service canaries and the eight-repeat output hash matched the accepted baseline exactly.
- The raw finite-logprob sweep passed all 131 lengths: 1–128, 133, 197 and 261. Prompt padding remained disabled.

The candidate is lossless by design, not bitwise identical. The altered tile changes only parallel decomposition and floating-point accumulation order.

## Deployment

The validated binaries are installed in local image `b70-vllm-latest-xpu:a397c58-head256-k64`, image ID:

```text
sha256:dbe76bb9ba1a55c5ab163f0e1ee961f29d0bc5bd2706f542083c158d8b4c53c5
```

Compose serves that image as `b70-vllm-qwen38-latest` on the established port 19622. The previous stock-kernel container is stopped as `b70-vllm-qwen38-latest-stock-20260827`; it retains the graph-dispatch fix and is a functional rollback. The disposable bind-mounted test container was removed.

## Rejected candidate

Q128/K64 reduced the Q tile and subgroup count. Its modeled 32K attention time was 5.523474 seconds, 2.60% slower than Q256/K64, so it was rejected without a 100K or full-model run.

Q512 and K128 were not built. Q256/K64 already compiles at 256 registers with about five spilled registers on BMG; enlarging either dimension is more likely to reduce occupancy than improve reuse. This is a reason to defer them, not a measured result.

## Reproduction

Apply the committed patch to `vllm-xpu-kernels` commit `a397c58eb7781e6fe0d6b3fb7c25d21b5f658784`. Build with oneAPI 2026.0 and the runtime-compatible 26.27 `ocloc`/2.38 IGC pair. A builder image can be produced from [`Dockerfile.xpu-kernels-build`](../docker/Dockerfile.xpu-kernels-build).

The focused build used only these two generated specializations:

```text
# VLLM_CHUNK_PREFILL_CONFIG
256,true,true,false,false,false

# VLLM_PAGED_DECODE_CONFIG
8,256,64,false,false,false
```

Relevant CMake settings:

```text
BUILD_SYCL_TLA_KERNELS=ON
FA2_KERNELS_ENABLED=ON
BASIC_KERNELS_ENABLED=OFF
GDN_KERNELS_ENABLED=OFF
MOE_KERNELS_ENABLED=OFF
MQA_LOGITS_KERNELS_ENABLED=OFF
MHC_KERNELS_ENABLED=OFF
XPU_SPECIFIC_KERNELS_ENABLED=OFF
XPUMEM_ALLOCATOR_ENABLED=OFF
```

The decode specialization is required even for pure prefill because the current XPU wrapper defaults to mixed-batch mode and calls a masked decode companion. Omitting it makes the focused library fail at runtime.

The builder's bundled `ocloc` was incompatible with the installed runtime and had to be replaced with the exact runtime image's compiler pair. Reuse the matching oneDNN source through `FETCHCONTENT_SOURCE_DIR_ONEDNN`; otherwise the focused build needlessly fetches and rebuilds unrelated dependencies.

Benchmark with [`benchmark_xe2_prefill_attention.py`](../scripts/benchmark_xe2_prefill_attention.py). Raw compiler trees, binaries, traces and request payloads remain ignored. The retained local binaries have these hashes:

```text
80e3e5fdf75b249a9de93fa620f66d48b9becb57d3d5e9492158771f0d50cbdf  _vllm_fa2_C.abi3.so
d893b467c7424ad1ab6f05a6c73b850d8b44385ce246a43fa1a321704c72e29c  libattn_kernels_xe_2.so
```

## Remaining lossless work

1. Make the head-256 policy conditional on page size and benchmark enough neighboring Xe2 shapes for an upstream-quality selection rule.
2. Port Qwen's fused gated split + Q/K RMSNorm + RoPE kernel to XPU. The generic vLLM fusion cannot match Qwen's gated QKV layout, while the model-specific Triton path is currently CUDA-only.
3. Revisit W4A16 GEMM only with oneDNN/Xe2 profiler evidence; it remains the largest short-context cost but has no isolated pathological shape.
4. Do not pursue decode kernels without a way to reduce weight traffic: the measured target and draft heads already stream about 589–596 GB/s against the card's listed 608 GB/s.
