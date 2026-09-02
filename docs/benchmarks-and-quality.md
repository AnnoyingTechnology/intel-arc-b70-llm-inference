# Benchmarks and quality evidence

## Current 210 W profile

The persisted cap is 210 W. Decode is content- and MTP-acceptance-dependent, so both a diversified short-output cell and a sustained-output cell are retained:

| Prompt/output | Decode | Cold prefill / TTFT | Runs |
|---:|---:|---:|---:|
| ~512/128, five prompt families | 91.98 tok/s median | 1,243 tok/s / 0.383 s | 5 |
| 8,156/128, five prompt families | 91.08 tok/s median | 1,566 tok/s / 5.207 s | 5 |
| 476/512, single-user exact-graph profile | 85.61 tok/s median | n/a | 6 |
| 8,190/8, cold W4A16 selector | n/a | 1,665 tok/s / 4.919 s | 3 |
| 32,563/8, cold W4A16 selector | n/a | 1,418 tok/s / 22.963 s | 3 |
| 32,565/8, cold, K64 attention | n/a | 1,325 tok/s / 24.580 s | 1 |
| 99,889/16, cold, K64 attention | n/a | 896 tok/s / 111.457 s | 1 |

The very short 512-token prefill cell is dominated by fixed HTTP/tokenizer/scheduler overhead and is not used as the headline prefill figure. The 32K and 100K decode cells have too few generated tokens for meaningful sustained decode conclusions.

Power-cap sweep and prefix-cache results are documented separately in [`power-efficiency.md`](power-efficiency.md) and [`prefill-and-prefix-cache.md`](prefill-and-prefix-cache.md).

The contained W4A16 selector promotion reduced three-run cold TTFT by 5.68%
at 8K and 6.73% at 32K while reducing prefill energy by 5.78% and 6.73%.
All six exact production-shape output digests matched stock bit for bit; the
service also passed 7/7 canaries, 8/8 repeat stability, exact baseline hashes,
and the 131-length finite-logprob sweep. See
[`xpu-w4a16-prefill64-tuning-2026-08-28.md`](xpu-w4a16-prefill64-tuning-2026-08-28.md).

### Native-window low-context and boundary gates

A five-family cold A/B compared the prior 196,608/8.5 GB profile with the
native 262,144/10.3 GB server profile. At approximately 512 prompt tokens,
median TTFT changed by -0.11%, prefill by +0.71%, and decode by +0.07%. At
approximately 8K, median TTFT changed by -0.14% and prefill by +0.12%. The 8K
decode median changed by -5.71% alongside lower MTP acceptance (0.639 versus
0.691); the invariant prefill path and 512-token decode show no systemic
low-context collapse from enabling the larger server window.

The exact usable boundary is 253,952 total tokens. A 253,824-token prompt plus
128 forced output tokens completed with `finish_reason=length`, 1,750.44 s
TTFT, 1,770.16 s total time, 6.44 decode tok/s and a 70°C peak. Requests with
zero and 4,096 tokens of reserve below the native 262,144 limit progressed but
ultimately parked at scheduler capacity; OpenCode therefore keeps an 8,192-token
runtime reserve. Full evidence and raw-result location are in
[`context-ceiling-probe-2026-09-02.md`](context-ceiling-probe-2026-09-02.md).

The K64 attention promotion changes only the Xe2 head-256 workgroup tile. Its post-promotion gate matched all seven deterministic outputs and eight repeat hashes, and passed the 131-length finite-logprob sweep. Full measurements and the source patch are in [`xpu-head256-attention-tuning-2026-08-27.md`](xpu-head256-attention-tuning-2026-08-27.md).

The single-user profile uses vLLM's built-in `interactivity` mode so MTP4's
five-token verifier uses an exact graph rather than an eight-row padded graph.
It improves the controlled 512-token decode median by 1.19% and reduces decode
energy/token by 0.92%; 32K prefill is unchanged within 0.5% noise. It passed
7/7 canaries, 8/8 repeat stability, exact baseline hashes and the 131-length
finite-logprob sweep. See
[`xpu-single-user-tuning-2026-08-27.md`](xpu-single-user-tuning-2026-08-27.md).

## Original 275 W selected profile

These bring-up results predate the 210 W cap and prefix-cache promotion. They are single-stream, cold unique-prefix requests with prefix caching disabled. Decode rate is measured after the first generated token. The 128-token cells are a standardized comparison, not a blanket sustained-rate claim.

| Prompt/output tokens | Median decode | Range | Prompt rate or TTFT | MTP acceptance |
|---:|---:|---:|---:|---:|
| 512/128, permanent Compose | 118.60 tok/s | 118.48–118.67 | 1,764 prompt tok/s median | 510/540 = 94.4% |
| 8192/128 | 114.88 tok/s | 110.57–115.02 | 1,964 prompt tok/s median | 506/528 = 95.8% |
| 512/256, permanent Compose | 84.46 tok/s | 82.35–86.72 | 1,757 prompt tok/s median | 901/1528 = 59.0% |
| 8192/256, permanent Compose | 79.99 tok/s | 75.91–86.98 | 1,961 prompt tok/s median | 908/1480 = 61.4% |
| 196480/128, exact boundary | 62.56 tok/s | one run | 301.217 s TTFT | 101/103 = 98.1% |

The exact-boundary request contained 196,480 prompt tokens and generated 128 tokens with `finish_reason=length`, totaling the configured 196,608 tokens without OOM or device loss. It predates prefix-cache promotion. The final aligned hybrid-cache layout reports 209,523-token capacity, but the expensive boundary request was not repeated at the user's direction.

Primary evidence:

- `results/compose-final-draft-int4-mtp4-p512-g128/run/results.json`
- `results/frozen-gptq-draft-int4-mtp4-p8192-g128/run/results.json`
- `results/compose-final-draft-int4-mtp4-p512-g256/run/results.json`
- `results/compose-final-draft-int4-mtp4-p8192-g256/run/results.json`
- `results/context-196608/p196480-g128.json`
- `results/context-196608/p196480-g128.sse.jsonl`
- `results/context-196608/exact-p196480.json`

## Quality gate

The promoted profile passed:

- Seven exact deterministic canaries covering exact output, copying, arithmetic, JSON, factual recall, logic and Python evaluation.
- Exact normalized-output and hash comparison with the Unsloth GGUF baseline.
- Eight identical repeat outputs and hashes.
- A 30,362-prompt-token needle retrieval twice, including through permanent Compose.
- The exact 196,608-token boundary test.
- The exact 253,952-token usable boundary test under the native server profile.

Final permanent-service evidence:

- `results/context-196608/quality-compose-final-nonthinking.json`
- `results/context-196608/quality-short-nonthinking.json`
- `results/quality-compose-draft-int4-32k-nonthinking.json`
- `results/quality-compose-final-qwen38-mtp4-nonthinking.json`

Unsloth reference artifact:

`/srv/models/lm-studio/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf`

- Revision: `4ca720788d1e01f1bff70c033e0d0028fd02e502`.
- SHA-256: `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`.

## Comparative results

| Target/backend | Prompt/output | Decode | Disposition |
|---|---:|---:|---|
| Unsloth UD-Q4_K_XL, llama.cpp Vulkan, no MTP | short | 12.02 tok/s | quality reference; slow |
| Same GGUF, Vulkan MTP2 | short | 16.43 tok/s | same target artifact |
| Same GGUF, llama.cpp SYCL MTP2 | short | 29.93 tok/s | fastest GGUF control |
| Same GGUF, SYCL MTP4 | short | 24.14 tok/s | acceptance loss made it slower |
| SergiioB GPTQ target, vLLM XPU MTP4 | 512/128 | 86.05 tok/s median | rejected: 6/7 non-thinking canaries; code output regressed |
| Frozenlock AutoRound through INC, no MTP | 512/128 | 30.48 tok/s median | correct but slow |
| Frozenlock through INC, MTP3/4 | short | invalid | deterministic corruption, 0/7 |
| Frozenlock XPU GPTQ + BF16 draft MTP4 | 512/128 | 82.98 tok/s median | quality-passing rollback |
| Same rollback | 8192/128 | 81.98 tok/s median | quality-passing rollback |
| Selected target + draft-only INT4 MTP4 | 512/128 | 118.83 tok/s isolated | promoted after quality gate |

## Interpretation and limits

The selected result exceeds the target 70–90 tok/s band for the standardized 128-token cells and remains near it for diversified 256-token outputs. MTP acceptance falls for longer generations, explaining the lower sustained rates.

These tests catch the concrete corruption and quantization regressions found during bring-up. They do not prove universal capability, logit, safety or long-context equivalence. A base-versus-Huihui decision requires the broader, paired evaluation described in [`huihui-plan.md`](huihui-plan.md).

The rejected 2026-08-24 GDN scheduler trial was rolled back before promotion.
The restored service then passed 7/7 exact canaries, eight-repeat stability,
prior-output hash parity, a 30,350-token needle and a 32K tool flow. Its recovery
performance cells measured 85.07 tok/s at p512/g128 and 88.39 tok/s at
p8192/g128, with 1,570 prompt tok/s at 8K. See
[`gdn-maintenance-window-2026-08-24.md`](gdn-maintenance-window-2026-08-24.md).

## GDN prompt-guard containment gate

The 2026-08-25 containment appends one token only for raw completion prompts
with length `64*N+5`; chat inserts that token before the final `<|im_end|>` so
the assistant-generation prefix is unchanged. Its deployment gates passed:

- exhaustive raw 1--128 sweep: 128/128 finite, with 5 and 69 reported as 6 and
  70 inference tokens;
- raw neighbors 132/133/134: all finite, with 133 reported as 134;
- exact deterministic chat output at rendered lengths 68/69/70: `OK` for all
  three, with only 69 entering inference as 70;
- guarded tool call and guarded tool-result ingestion: `lookup_record` selected
  correctly and `cobalt-orbit-731` recovered; inference lengths 326 and 454;
- unaffected five-family p512/g128 performance: 92.48 tok/s median decode and
  0.383 s median TTFT at 210 W, consistent with the prior 91.98 tok/s cell.

The first attempted placement after the assistant-generation marker produced
an empty one-token stop despite finite logits. The semantic canary rejected it
before handoff. Only the final-message placement is deployed.
