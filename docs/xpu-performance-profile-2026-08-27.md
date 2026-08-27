# XPU performance profile — 2026-08-27

## Scope

This profile answers one question: where could a 24-hour optimization effort improve Qwen3.8-27B with MTP4 on the B70?

Runtime under test:

- vLLM `46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3`
- `vllm-xpu-kernels` `a397c58eb7781e6fe0d6b3fb7c25d21b5f658784`
- Frozenlock target GPTQ INT4, BF16 MTP sidecar, INT4 draft linears and LM head
- FP8 KV cache, 8,192-token scheduler chunk, MTP4, XPU graph
- persisted 210 W card cap

The production container was idle before capture, stopped for profiling, and restored healthy on port 19622 afterward. Stopping it necessarily cleared its in-memory prefix cache.

## Captures

PyTorch XPU traces were recorded from an otherwise identical isolated server. Raw traces remain outside Git because they total about 19 MB compressed; the relevant measurements are preserved below.

| Capture | Workload | Wall result | Trace SHA-256 |
|---|---|---:|---|
| Decode | p476/g512, low-acceptance `assistant` prompt | 74.23 tok/s; 6.884 s decode | `54feabbc9a4f46ce22cce9658301cbb68f1841370cc0b2301f30bdfd4881bd0f` |
| Prefill 8K | 8,190 uncached tokens | 1,495 tok/s; 5.476 s TTFT | `a7db2c62016468a402454bccd43eb66d98b3b971f48124db8172f8fdc583a725` |
| Prefill 32K | 32,561 uncached tokens | 1,305 tok/s; 24.960 s TTFT | `7d82b9b3498255b036c570b24b4364d57fe9aa4113c2631cafbc6eb4fd4dfeab` |

Profiler overhead makes the wall figures diagnostic rather than replacements for the established benchmark results. Device-time proportions are the useful result.

## Prefill attribution

Prefill is fully visible to the XPU profiler.

| XPU operation | 8K | 32K |
|---|---:|---:|
| oneDNN W4A16 INT4 GEMMs | 3.794 s / 73.2% | 14.135 s / 57.2% |
| Xe2 FlashAttention prefill | 0.629 s / 12.1% | 7.926 s / 32.1% |
| Xe2 GDN | 0.424 s / 8.2% | 1.492 s / 6.0% |
| Everything else | 6.5% | 4.7% |

The 8K scheduler split was 4,992 + 3,136 + 62 tokens. The 32K split was 6,656 × 4 + 3,328 + 2,560 + 49 tokens.

Large-M W4A16 GEMMs sustain a consistent 108.8–127.4 TFLOP/s across the model's traced shapes. No isolated pathological projection appears: the largest 32K contributions are the `[M,5120]×[5120,34816]` gate/up projection and `[M,17408]×[17408,5120]` down projection. Replacing mature oneDNN GEMM is therefore a high-effort target; a 10% GEMM improvement would save about 7.3% at 8K and 5.7% at 32K.

Attention grows from 12.1% at 8K to 32.1% at 32K and continues growing with context. The exact hot specialization is batch 1, 24 query heads, 4 KV heads, head dimension 256, causal paged attention, FP16 queries, and FP8 KV. The installed Xe2 implementation uses one fixed head-256 policy: Q tile 256, K tile 32, 32 subgroups per workgroup, and two pipeline stages. A B70-specific policy sweep has a plausible payoff without writing a new attention algorithm. A 15% attention-kernel gain would save 4.8% at 32K and likely more near 100K.

GDN is only 6–8% of prefill and already contains Intel's Xe2 optimizations. Even an unrealistic 2× GDN rewrite would save 3–4%; it is not the first target.

## Decode attribution and limitation

XPU graph replays hide the kernels inside captured command lists. The trace reports 6.814 s self CPU time, including 4.504 s in `zeEventHostSynchronize`, but only 1.772 s of individually named XPU kernels. The synchronization time includes graph-hidden GPU execution and must not be called removable CPU overhead.

Two output projections remain visible and are already actionable:

| Projection | Calls | XPU time | Per call | Decode wall share |
|---|---:|---:|---:|---:|
| INT4 draft LM head, M=1 | 612 | 673.8 ms | 1.101 ms | 9.8% |
| BF16 target LM head, M=5 | 150 | 647.1 ms | 4.314 ms | 9.4% |
| Combined | | 1.321 s | | 19.2% |

The server's cumulative MTP telemetry, including preceding warmups, reported mean acceptance length 3.11 and 52.7% average draft-token acceptance. The capture itself emitted 512 tokens through 150 M=5 target passes, about 3.41 output tokens per verifier pass. This low-acceptance prompt family's 74.23 tok/s is below the diversified 80–90 tok/s service range.

The draft INT4 head reads about 656 MB of packed weights and scales per call, yielding about 596 GB/s of payload. The BF16 target head reads about 2.54 GB per call, yielding about 589 GB/s. Their matching streaming rates indicate bandwidth scaling, not a uniquely slow INT4 kernel. A replacement M=1 INT4 kernel has little room unless it reduces weight traffic.

Quantizing only the target LM head is the clearest decode experiment. FP8/INT8 could at most remove roughly half of its measured 9.4% wall share; INT4 could at most remove roughly three quarters. Those are ceilings, not predictions. The target head determines verified logits, so any quantization requires the full quality gate. INT4 would also free about 1.9 GB of VRAM; FP8 would free about 1.27 GB.

An attempted no-XPU-graph decode profile was rejected as unfaithful: with graph disabled, this build allocated 31.63 GiB while constructing the base model and OOMed before loading completed, even at an 8K model limit and 1 GB KV reservation. No conclusions were drawn from it.

## Ranked next work

1. **Long-prefill kernel work:** microbenchmark the exact Xe2 FlashAttention specialization and sweep Q/K tiles, subgroup layout, and pipeline stages. Gate changes at 8K, 32K, and 100K with bitwise-finite and quality checks.
2. **Decode, low code risk:** A/B an FP8 target LM head, then INT4 only if FP8 leaves useful performance on the table. Run the existing capability/canary suite because this changes verified logits.
3. **Decode instrumentation:** obtain graph-replay kernel visibility before changing main-model or MTP-body kernels. The present trace cannot rank graph-hidden candidates honestly.
4. **Cheap existing fusion:** test vLLM's currently disabled Q/K RMSNorm + RoPE fusion. It is a configuration A/B, not a custom-kernel project, and its ceiling is much smaller than GEMM or long-context attention.
5. **Do not prioritize:** another GDN rewrite, scratchpad-allocation cleanup without a measured CPU bottleneck, or a speculative M=1 INT4 rewrite.

## 24-hour decision

A focused 24-hour effort is justified, but the best custom-kernel target is the existing Xe2 FlashAttention policy for this exact head-256/GQA/FP8-KV shape, not GDN. The best immediate decode experiment is target-LM-head quantization, not a new arithmetic kernel. Expect single-digit total gains from either path; the profile provides no evidence for a credible 2× improvement.
