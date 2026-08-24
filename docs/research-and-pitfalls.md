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

The earlier 100K setting was a conservative untested cap, not a hardware limit. The service now explicitly reserves 8.5 billion bytes for FP8 KV. Before prefix caching, the engine reported 214,214 tokens and completed the exact 196,608-token boundary; the final cache layout reports 209,523 tokens of capacity.

The model advertises 262,144 native tokens. That boundary was not selected because its projected KV requirement is about 10.8 GiB, while only roughly 8–9 GiB can safely remain after weights, draft structures, graphs and runtime allocations. Reaching native context would require a separately tested memory reduction or offload path; do not increase `--max-model-len` based only on model metadata.

With automatic prefix caching, hybrid GDN/attention page alignment changes reported capacity from the earlier 214,214 tokens to 209,523. This remains above the configured 196,608 window, but capacity figures must always be read from the final runtime configuration rather than copied across cache modes.

## Prefill lessons

- The initial Compose file explicitly disabled prefix caching. That made each OpenCode tool result reprocess the entire conversation. SHA-256 prefix caching reduced a repeated 32K request from 21.87 s cold to 2.62 s and then 1.07 s.
- Prefix caching fixes repeated history, not a new project's first ingestion. Cold rates measured 1,566 tok/s at 8K, 1,300 tok/s at 32K and 851 tok/s at 100K under the final 210 W profile.
- Doubling `max_num_batched_tokens` from 8,192 to 16,384 did not fix long prefill. The controlled 100K run regressed from 117.35 to 119.78 s and used more energy/token.
- The pinned runtime already selects its XPU GDN custom operation. Qwen3.8 still has 16 full-attention layers without sliding-window attention; long cold prefill remains a real upstream/kernel and algorithmic optimization target.
- Prefix cache is in memory. Server recreation, model replacement, or changes near the beginning of the system prompt/tool schema invalidate reuse.
- A long OpenCode tool-use session exposed a cache-correctness failure: the same follow-up was coherent after cold prefill but collapsed to token ID 0 (`!`) after reusing 48,256 tokens cached by the failed turn. Repetition penalties through 1.20 did nothing. See [`repetition-incident.md`](repetition-incident.md); `--prefix-match-unit 64` is now a suspect pending a controlled MTP-preserving A/B, not an established safe optimization.
- The failing request's logprob response contains NaN under both Qwen sampling and greedy decoding; its cold follow-up has finite logprobs. The checkpoint is BF16, but the service currently forces FP16 despite Intel W4A16 supporting both activation dtypes. Restoring BF16 is the first pending end-to-end A/B; fine-grained prefix matching is second because it cannot explain a zero-cache-hit cold NaN.
- OpenCode option names matter: `topP`/`topK` were forwarded as unknown custom fields, while `top_p` remained 1 and `top_k` was absent. Snake-case `top_p`/`top_k` were verified on the wire. On the complete recovered payload, adding only `top_k=20` changed 3/3 cold failures to 3/3 passes; it remains a guardrail because the original cold NaN ignores it.

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
