# Research and pitfalls

## LM Studio installation provenance

Two LM Studio tracks had coexisted. The pre-reboot session used a newer AppImage. After reboot, the desktop entry launched the installed Debian package at `0.4.6+1`. That package was upgraded in place to `0.4.21+2`, but its already-running 0.4.6 process retained the old mapped executable until exit. This explained the apparent version contradiction; it was not another hidden install.

Cleanup completed on 2026-08-24:

- Retained only Debian package `lm-studio` `0.4.21+2` under `/opt/LM-Studio`.
- Removed all home-directory LM Studio AppImages and downloaded LM Studio `.deb` installers.
- Removed the stale alias, broken legacy symlink and obsolete desktop entry.
- Added `/home/julien/.cache/lm-studio/bin` to PATH through `~/.profile` and `~/.bashrc` so `lms` resolves normally.
- Kept a validated user desktop override at `~/.local/share/applications/lm-studio.desktop` with the correct executable and icon.
- Stopped LM Studio and closed its ports 1234 and 39221.

LM Studio remains a GGUF reference/fallback, not the selected high-performance service. `lms runtime ls` can start its background service, so do not use it as a passive filesystem query.

## Why the initial 12 tok/s was misleading

The first 12.26 tok/s LM Studio run used Vulkan, four parallel slots, about 98K total context, FP16 K/V, the multimodal projector and MTP2. It was neither a clean one-sequence benchmark nor the B70's optimized backend ceiling.

Controlled results placed the same Unsloth GGUF at 12.02 tok/s without MTP, 16.43 with Vulkan MTP2, and 29.93 with SYCL MTP2. The breakthrough required leaving the GGUF/llama.cpp path for vLLM's Intel XPU W4A16 path.

## Quantization and speculative-decoding failures

- Frozenlock AutoRound selected vLLM's Intel `inc` backend. Non-speculative output passed, but multi-token speculative verification generated deterministic garbage and failed all canaries. Do not use `inc` with MTP on this pinned stack.
- Reinterpreting the documented AutoRound `auto_gptq` packing as GPTQ selected the working XPU W4A16 kernel without requantizing the target.
- The SergiioB GPTQ target was fast but failed a deterministic code canary (`30` instead of `14`) in non-thinking mode. It is retained only as the same-base source of 15 BF16 MTP tensors; its full target directory was removed.
- MTP4 is not automatically faster than MTP2. With the GGUF/SYCL path, lower acceptance made MTP4 slower.
- Draft-only INT4 is safe only in the speculative sense: target verification prevents an unaccepted draft token from being emitted. It is not proof of identical logits or universal behavioral equivalence.
- FP8 KV saves the memory needed for long context but may have accuracy trade-offs. Exact canaries and a full-window mechanical test passed; a broad long-context reasoning evaluation has not.

## Context sizing

The earlier 100K and 196,608 settings were conservative serving caps, not the
model limit. The service now reserves exactly 10,300,000,000 bytes for FP8 KV;
the final prefix-cache and vision layout reports 263,633 tokens of capacity at
the model's native 262,144-token limit.

Do not confuse allocation success with cold-request usability. Exact 196,608
text and maximum-image requests completed on the final layout, while a cold
262,144-total request remained healthy for 1,800 seconds but did not reach its
first token before timeout. The production client therefore reserves 8,192
tokens for output and relies on normal incremental agent growth and prefix
caching. Capacity figures must always be read from the final runtime rather
than projected proportionally across cache layouts.

## Prefill lessons

- The initial Compose file explicitly disabled prefix caching. That made each OpenCode tool result reprocess the entire conversation. SHA-256 prefix caching reduced a repeated 32K request from 21.87 s cold to 2.62 s and then 1.07 s.
- Prefix caching fixes repeated history, not a new project's first ingestion. Cold rates measured 1,566 tok/s at 8K, 1,300 tok/s at 32K and 851 tok/s at 100K under the final 210 W profile.
- Doubling `max_num_batched_tokens` from 8,192 to 16,384 did not fix long prefill. The controlled 100K run regressed from 117.35 to 119.78 s and used more energy/token.
- The pinned runtime already selects its XPU GDN custom operation. Qwen3.8 still has 16 full-attention layers without sliding-window attention; long cold prefill remains a real upstream/kernel and algorithmic optimization target.
- Prefix cache is in memory. Server recreation, model replacement, or changes near the beginning of the system prompt/tool schema invalidate reuse.
- A long OpenCode tool-use session exposed a deterministic XPU GDN prefill defect. Raw synthetic completions fail with NaN logprobs exactly when `prompt_tokens % 64 == 5`; exhaustive sweeps over 1–128 found only lengths 5 and 69. The original 49,925-token boundary has the same remainder. Reusing its damaged recurrent state then poisoned later prompts. See [`repetition-incident.md`](repetition-incident.md) and [`gdn_tail_probe.py`](../scripts/gdn_tail_probe.py).
- Raw five-token failures have no chat template, tools, prefix hit, resumed state or speculative metadata. `--prefix-match-unit 64`, MTP4, the private transcript and general FP16 overflow are therefore ruled out as the initial trigger. The live path is the 64-token XPU GDN custom operation from `vllm-xpu-kernels 0.1.12.3`; the exact kernel instruction and permanent correction remain under isolation.
- A 144-case direct fused/split operator matrix stayed finite, including the bad API lengths, so random operator inputs are not a sufficient reproducer. The first prompt-only scheduler split applied successfully but still returned NaN at five tokens and was rolled back. Do not promote a source-level argument or unit test without the raw end-to-end finite-logprob gate.
- OpenCode option names still matter independently: `topP`/`topK` were forwarded as unknown custom fields, while `top_p` remained 1 and `top_k` was absent. Snake-case `top_p`/`top_k` were verified on the wire. The official `temperature=1`, `top_p=0.95`, `top_k=20` sampler is retained as normal model configuration, but no sampler can repair upstream NaNs.

## Power-sweep pitfalls

- Compare energy/token as well as tok/s. Decode efficiency kept improving toward 160 W, but cold-prefill efficiency bottomed at 210 W and worsened below it.
- A descending cap sweep heat-soaks the card. Do not interpret its temperature maxima as an A/B thermal curve without randomized order and full cool-downs.
- This board's initial Xe hwmon cap was 275 W, while Intel publishes 230 W reference TBP and a 160–290 W configurable range. Record both the product specification and the actual board control.
- Power limits revert after reboot unless applied by a privileged boot mechanism. The final systemd helper validates the exact PCI identity and read-back value.

## Reproducibility rules

- Keep the image digest and model revisions pinned.
- Verify patch hashes before reusing results.
- Benchmark one request at a time, with exact prompt/output token counts, cold unique prefixes and prefix caching disabled.
- Record MTP proposed/accepted counters, not only wall-clock throughput.
- Never promote throughput alone; run deterministic quality gates first and after permanent Compose recreation.
- Do not compare differently quantized targets and infer backend performance from the result.
- Do not treat a short-canary pass as comprehensive capability equivalence.
