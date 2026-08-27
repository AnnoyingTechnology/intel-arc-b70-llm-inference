# Active lossless optimization roadmap — 2026-08-27

Status: **actively pursuing**.

## Contract

- Optimize the established single-user Qwen3.8-27B MTP4 service at 210 W.
- Do not change target weights, target logits, KV precision, sampling semantics
  or the 196,608-token context contract.
- Promote only a repeatable end-to-end gain of at least 3% on the workload the
  change targets, with correctness, quality, energy and neighboring-shape gates.
- Drop sub-3% changes after recording the result. Do not compound marginal
  tweaks whose individual effects cannot be distinguished from noise.
- Work highest expected gain and lowest implementation cost first.

## Roofline position

The B70 provides 608 GB/s nominal memory bandwidth. One MTP4 cycle reads about
15.23 GB of target weights plus four approximately 0.87 GB draft passes. At the
measured 3.4–3.5 mean acceptance length, the weight-only ceiling is about
112 tok/s. The current 85.61 tok/s fixed decode cell reaches about 76% of that
absolute ceiling and an estimated 82–90% of the practical ceiling after
unavoidable state, activation and dispatch costs.

Nominal XMX arithmetic gives loose prefill ceilings around 3,650 tok/s at 8K,
3,330 at 32K and 2,685 at 100K. The measured 1,566, 1,325 and 896 tok/s reach
43%, 40% and 33% respectively. These ratios understate implementation quality:
the nominal roof assumes continuous peak XMX utilization for attention, GDN,
normalization and dequantization. Large W4A16 GEMMs already sustain 109–127
TFLOP/s and the output heads sustain 589–596 GB/s.

## Execution order

### 1. Same-footprint draft calibration

Current draft MTP linears and draft LM head use one-time symmetric INT4 G128
max-absolute quantization. It is uncalibrated. Improve rounding and clipping at
the same byte footprint, starting with cheap weight-error searches and
escalating to activation-aware calibration only if the cheap candidates move
acceptance.

The goal is to raise mean MTP acceptance from about 3.45 toward 3.8 without
slowing a draft pass. At fixed cycle cost, that raises the decode roof by about
10%. Target verification remains authoritative; deterministic outputs and the
speculative sampler still require explicit gates.

Stop when no candidate projects at least 3% end-to-end gain.

### 2. Contained oneDNN W4A16 update

Check the pinned oneDNN revision against current Xe2 INT4/BF16 fixes. If a
relevant change is absent, test it in an isolated image using the existing 8K
and 32K cold-prefill cells. W4A16 GEMMs occupy 73% of 8K and 57% of 32K device
time, so a 5% kernel improvement can clear the 3% service threshold at 8K.

Do not replace oneDNN without a measured pathological shape or a relevant
upstream correction.

### 3. Xe2 attention dispatch

Extend the validated Q256/K64 head-256 result into a sequence-length and
page-aware dispatch policy. Test neighboring shapes only where the isolated
kernel projects at least 3% whole-prefill gain. This is the principal 100K
prefill path and the strongest upstream candidate, but it requires more build
and regression work than the first two items.

### 4. Decode graph visibility

Expose aggregate graph-replay timing and memory traffic only if the preceding
work fails. Write no custom decode kernel until one operation or launch boundary
accounts for at least 3% recoverable wall time. The visible output heads are
already at 97–98% of nominal bandwidth.

## Explicit non-targets

- The rejected Qwen projection/QK-norm/RoPE fusion.
- Another GDN rewrite.
- Same-traffic output-head kernels.
- Target-head, target-weight or more aggressive KV quantization.
- Sub-3% graph, scheduler or kernel tweaks.
- Full-TDP operation as a permanent substitute for optimization.
