# Intel Arc Pro B70 local inference

This repository records the quality-gated optimization of Qwen3.8-27B for one
32 GiB Intel Arc Pro B70. The controlled llama.cpp/Vulkan baseline produced
12.02 tok/s, and the fastest same-GGUF SYCL control reached 29.93 tok/s. The
quality-gated vLLM/XPU service now sustains 85.61 tok/s on a long decode cell
and reached 118.60 tok/s on a favorable short-output cell: **7.1x the
controlled Vulkan baseline in sustained use and up to 9.9x on the standardized
short cell**.

This was not a one-flag speedup. It required selecting the viable backend and
weight path, constructing a same-base speculative sidecar, quantizing only the
draft, sizing the KV cache for the full context contract, profiling exact XPU
shapes, tuning attention and W4A16 dispatch, and rejecting changes that failed
end-to-end, energy or correctness gates. The reproducible result uses a
transparent INT4 target, MTP4, FP8 KV, automatic prefix caching and a persisted
210 W efficiency cap.

## Result

The validated patched service exposes `qwen38` at the established `http://127.0.0.1:19622/v1` endpoint. Existing OpenCode sessions therefore require no provider change or restart.

| Measurement at 210 W | Result |
|---|---:|
| Sustained decode, p476/g512 | **85.61 tok/s** median |
| Diversified short decode, ~p512/g128 | **91.98 tok/s** median |
| Cold prefill, 8,190 tokens | **1,665 tok/s**, 4.919 s TTFT |
| Cold prefill, 32,563 tokens | **1,418 tok/s**, 22.963 s TTFT |
| Cold prefill, 99,889 tokens, before the final W4A16 selector | **896 tok/s**, 111.46 s TTFT |
| Cached 32K repeated turn | **1.07 s TTFT** after 31,616 cached tokens |
| 32K tool-result follow-up | **3.52 s total**, 29,952 cached tokens |

The earlier 275 W profile reached 118.60 tok/s on a favorable 128-token forced-output cell. It is not a blanket sustained-rate claim: MTP acceptance varies with generated content, and longer diversified outputs settle around the 80–90 tok/s band.

The 210 W cap is the measured cross-workload efficiency knee. Relative to 275 W, it retains 98.6% of sustained decode rate while reducing decode energy by 11.6% per output token. Cold 8K prefill retains 82.5% of throughput while improving energy by 6.1% per input token; going below 210 W makes cold prefill less efficient as well as slower. In normal local-agent use, the card has been observed around 65 °C with only faint, unobtrusive noise, unlike full-TDP operation.

## Bandwidth reference and remaining headroom

At the measured 3.4–3.5 mean acceptance length, one MTP4 cycle streams about
15.23 GB of target weights plus four approximately 0.87 GB draft passes. That
places the sustained cell's absolute weight-only roof near 112 tok/s; 85.61
tok/s is about 76% of that deliberately unattainable upper bound. After
allowing for unavoidable recurrent state, activations and dispatch costs, the
service is estimated to operate at **82–90% of its practical bandwidth-limited
decode ceiling**. As a second reference, the visible draft and target output
heads stream 589–596 GB/s against the B70's nominal 608 GB/s, or 97–98%.

These are roofline estimates, not a claim that every graph-hidden kernel has
been proved optimal. They do show why another lossless 2x decode improvement is
not credible without first finding a new source of recoverable traffic or
graph overhead.

## Optimization ladder

| Accepted decision | Measured contribution | Integrity boundary |
|---|---:|---|
| Vulkan/GGUF to the best same-GGUF SYCL/MTP2 path | 12.02 → 29.93 tok/s, **2.49x** | Same GGUF target; backend and MTP changed together |
| GGUF/SYCL to vLLM XPU GPTQ W4A16 with BF16 MTP4 draft | 29.93 → 82.98 tok/s, **2.77x** | System-path comparison, not a single-variable A/B |
| Symmetric INT4 G128 for only the draft LM head and five draft linears | 82.98 → 118.83 tok/s, **+43.2%** on p512/g128 | Unchanged target verifies every emitted token |
| SHA-256 automatic prefix caching | 32K TTFT 21.87 → 2.62 → 1.07 s, **up to -95.1%** | Reuses only identical prefix blocks |
| Xe2 head-256 Q256/K64 attention policy | 100K TTFT 115.19 → 111.46 s, **-3.24%** | Weights, logits, precision, KV and MTP unchanged |
| Exact five-row single-user MTP graph | 84.60 → 85.61 tok/s, **+1.19%** | Removes graph padding only |
| Contained oneDNN W4A16 32x64 selector for four exact prefill projections | 8K TTFT **-5.68%**; 32K **-6.73%** | Six production-shape output tensors bit-identical |
| Seven-point power-cap sweep selecting 210 W | **-11.6% decode energy/token** while retaining 98.6% of 275 W decode | Same model and workload at every cap |
| FP8 KV with an explicit 8.5 GB reservation | 209,523-token cache capacity; **196,608-token serving contract** | Exact context boundary and quality gates passed |

The percentages above are deliberately not multiplied together: several rows
use different controlled workloads. MTP4 also receives no invented standalone
gain; it helped the selected XPU path, but MTP4 was slower than MTP2 on the
rejected GGUF/SYCL path because acceptance fell.

Every promoted kernel change passed exact or tolerance-bounded tensor checks,
neighboring shapes, deterministic canaries, repeat hashes, energy measurement
and full-service tests. Plausible changes were recorded and rejected when they
failed to clear the end-to-end threshold: a 16,384-token scheduler chunk was
2.0% slower at 100K, a Qwen fusion that was 78–84% faster in isolation regressed
serving decode by 0.98%, and draft scale refinement regressed decode by 0.34%.

## Correctness breakthrough

The optimization work also exposed a vLLM legacy-runner bug: an MTP4
five-token prefill tail was mistaken for uniform decode and sent through the
wrong FULL graph, where it consumed invalid reserved recurrent state. Adding
the missing `has_prefill` condition fixed lengths 5/69/133, the exhaustive
1–128 sweep and the original 49,925-token failure without changing the prompt
or XPU arithmetic. The validated temporary
[runtime patch](docker/patches/patch_uniform_decode_prefill.py),
[authoritative investigation](docs/gdn-64n5-investigation-pause-2026-08-25.md)
and [upstream issue #548](https://github.com/vllm-project/vllm-xpu-kernels/issues/548)
retain the evidence.

Cold long-context prefill remains the principal throughput limitation.
Qwen3.8-27B has 16 full-attention layers without a sliding window, so prefix
caching can reuse unchanged multi-turn history but cannot make first ingestion
free. Decode has no demonstrated remaining lossless change above the 3%
promotion threshold; the next valid investigation is graph-replay attribution,
not an unprofiled replacement kernel. Long-session cache eviction and
cache-preserving compaction remain potentially large user-visible latency work.

## What might still be left

The obvious gains are exhausted, but the investigation is not declared
finished. These bounded hypotheses remain, in priority order:

1. **Decode graph-replay attribution.** Captured command lists hide much of the
   decode execution. Aggregate timing and traffic may expose an operation or
   launch boundary with at least 3% recoverable wall time. No such hotspot has
   yet been demonstrated.
2. **Prefix-cache eviction resilience.** A two-request 21–23K side conversation
   made a warm approximately 157K main session miss its entire usable prefix.
   Controlled A/Bs of positive GDN/Mamba retention intervals might turn that
   catastrophic zero hit into bounded replay without reducing the 196,608-token
   contract.
3. **Cache-preserving OpenCode compaction.** One real 137,459-token compaction
   missed the complete warm prefix and took 443 seconds. A guarded side-by-side
   implementation exists and passed its software tests, but still needs a
   synthetic long-session inference gate before promotion.
4. **Final 100K W4A16 accounting.** The new selector is proven end-to-end at 8K
   and 32K and acts on chunk shapes also used during longer ingestion. A 100K
   A/B would establish the combined attention-plus-W4 result; benefit there is
   plausible, not yet measured.
5. **New evidence for draft acceptance or upstream kernels.** Better
   activation-aware draft calibration could theoretically raise the decode
   roof, and future oneDNN/XPU releases may improve exact production shapes.
   The current cheap draft refinement regressed by 0.34%, however, so neither
   path merits more work without a materially different candidate.

Target-head or target-weight requantization, more aggressive KV quantization,
same-traffic output-head kernels, another GDN rewrite, larger scheduler chunks
and full-TDP operation are not remaining lossless opportunities under this
project's contract.

## Quick operations

```bash
cd /home/julien/Documents/B70/docker
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:19622/health
docker compose logs -f --tail=100
```

Power-cap state:

```bash
systemctl status b70-power-limit.service
cat /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*/power1_cap
```

Xe/Battlemage telemetry caveat: after the final inference on 2026-08-28, the
B70 was genuinely idle at roughly 6–12 W and 0% engine use, but its package
and PCIe readings remained frozen at 65/66 °C before jumping to 26/28 °C. The
stock fan took 13m19s to stop. This matches the upstream
[B70 stale-package-temperature report and forcewake RFC](https://lkml.iu.edu/2608.2/11582.html),
which postdates the installed kernel. Treat a flat package temperature as
potentially stale rather than evidence of hidden inference; the firmware fan
cooldown is real, while the RFC fixes reporting and is not yet proven to
shorten that cooldown.

Stop only the inference service:

```bash
cd /home/julien/Documents/B70/docker
docker compose down
```

The API is deliberately loopback-only and the container uses `restart: "no"`. LM Studio must not infer on the B70 concurrently.

## Model and quality contract

The selected overlay is `/srv/models/vllm/Frozenlock--Qwen3.8-27B-GPTQ-MTP-BF16`. Image, model revisions, generated hashes and runtime-patch hashes are pinned in [architecture.md](docs/architecture.md).

The profile passed seven deterministic canaries, repeat stability, JSON/tool/code checks, comparison with the retained Unsloth GGUF reference, and a 30K needle retrieval. The same target and 8.5 GB KV reservation passed an exact 196,608-token mechanical boundary before prefix caching changed the hybrid page layout; the final runtime reports 209,523-token capacity, but that expensive boundary request was not repeated. These are bounded regression gates, not proof of universal equivalence.

The planned Huihui abliterated candidate must match the base model's quality and performance before it can replace the resident model. See [huihui-plan.md](docs/huihui-plan.md).

## Documentation

Start with:

- [Benchmarks and quality](docs/benchmarks-and-quality.md): complete performance evidence, quality gates and caveats.
- [Architecture](docs/architecture.md): exact engine, overlay, versions and hashes.
- [Operations](docs/operations.md): lifecycle, requests, validation and rollback.
- [Research and pitfalls](docs/research-and-pitfalls.md): rejected paths and the lessons that shaped the final system.

Optimization reports:

- [Active optimization roadmap](docs/active-optimization-roadmap-2026-08-27.md): roofline position, ≥3% promotion threshold and current execution order.
- [XPU performance profile](docs/xpu-performance-profile-2026-08-27.md): measured decode/prefill kernel attribution and ranked targets.
- [W4A16 prefill selector tuning](docs/xpu-w4a16-prefill64-tuning-2026-08-28.md): exact-shape strategy search, bit-identical result, end-to-end gain and bounded build.
- [Xe2 head-256 attention tuning](docs/xpu-head256-attention-tuning-2026-08-27.md): lossless K64 policy, exact benchmarks and reproducible patch.
- [XPU single-user tuning](docs/xpu-single-user-tuning-2026-08-27.md): exact MTP graph gain and rejected Qwen fusion.
- [Power efficiency](docs/power-efficiency.md): seven-point sweep, selection, persistence and rollback.
- [Prefill and prefix caching](docs/prefill-and-prefix-cache.md): TTFT scaling, tool-flow result and remaining bottleneck.
- [Optimization resume and OOM postmortem](docs/optimization-resume-2026-08-28.md): compiler incident, 26 GiB/no-swap guardrail and final promoted service.

Correctness and investigation archive:

- [Long-session prefix-cache eviction](docs/prefix-cache-eviction-incident-2026-08-26.md): observed full miss, compaction cliff and focused A/B plans.
- [Token-0 repetition incident](docs/repetition-incident.md): sanitized reproducer, evidence and recovery.
- [XPU GDN source audit](docs/xpu-gdn-source-audit.md): version forensics and kernel narrowing.
- [Rejected GDN maintenance trial](docs/gdn-maintenance-window-2026-08-24.md): failed scheduler A/B and rollback gates.
- [GDN `64*N+5` investigation](docs/gdn-64n5-investigation-pause-2026-08-25.md): authoritative evidence, root cause and artifact manifest.
- [GDN diagnostic code map](docs/gdn-diagnostic-code-map.md): trusted, negative-control and superseded instrumentation.
- [Upstream issue #548](https://github.com/vllm-project/vllm-xpu-kernels/issues/548): submitted Intel/XPU escalation; the [archived body](upstream/vllm-xpu-kernels-64n5-nan-issue.md) is versioned here.
- [Huihui plan](docs/huihui-plan.md): abliterated-model A/B and promotion gate.
- [References](docs/references.md): upstream sources and revisions.
