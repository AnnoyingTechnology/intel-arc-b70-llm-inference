# Intel Arc Pro B70 local inference

This Intel Arc Pro B70 was quite painful to get rolling properly, so here is what GPT-5.6-sol brute-forced with me. A vanilla LM Studio/llama.cpp Vulkan setup produced a measly ~10 tok/s even with MTP; the final quality-gated vLLM/XPU configuration reaches 80–119 tok/s depending on workload.

This repository documents the reproducible Qwen3.8-27B service for one 32 GiB B70: a transparent INT4 target, MTP4 speculative decoding, FP8 KV cache, automatic prefix caching, and a persisted 210 W efficiency cap.

> **Paused investigation (2026-08-25):** raw prompts fail with NaN logprobs exactly when `prompt_tokens % 64 == 5`. A trusted producer-side capture localizes the first non-finite boundary to layer-4 GDN: its inputs, projections, and `z` are finite, while all five active `core_attn_out` rows are NaN. Exact operands replay finitely outside the live compiled forward, and known fixes #344/#411/#437/#439 are present. The stock MTP4 service is restored; no correction is deployed. See the [authoritative handoff](docs/gdn-64n5-investigation-pause-2026-08-25.md) and [upstream issue draft](upstream/vllm-xpu-kernels-64n5-nan-issue.md).

## Result

The active service exposes `qwen38` through an OpenAI-compatible API at `http://127.0.0.1:19622/v1` and through OpenCode as `local-b70/qwen38`.

| Measurement at 210 W | Result |
|---|---:|
| Sustained decode, p512/g512 | **83.77 tok/s** median |
| Diversified short decode, ~p512/g128 | **91.08 tok/s** median |
| Cold prefill, 8,156 tokens | **1,566 tok/s**, 5.21 s TTFT |
| Cold prefill, 32,565 tokens | **1,300 tok/s**, 25.05 s TTFT |
| Cold prefill, 99,889 tokens | **851 tok/s**, 117.35 s TTFT |
| Cached 32K repeated turn | **1.07 s TTFT** after 31,616 cached tokens |
| 32K tool-result follow-up | **3.52 s total**, 29,952 cached tokens |

The earlier 275 W profile reached 118.60 tok/s on a favorable 128-token forced-output cell. It is not a blanket sustained-rate claim: MTP acceptance varies with generated content, and longer diversified outputs settle around the 80–90 tok/s band.

The 210 W cap is the measured cross-workload efficiency knee. Relative to 275 W, it retains 98.6% of sustained decode rate while reducing decode energy by 11.6% per output token. Cold 8K prefill retains 82.5% of throughput while improving energy by 6.1% per input token; going below 210 W makes cold prefill less efficient as well as slower. In normal local-agent use, the card has been observed around 65 °C with only faint, unobtrusive noise, unlike full-TDP operation.

## What made it fast

- Frozenlock Qwen3.8-27B AutoRound INT4 weights are served through Intel's XPU GPTQ W4A16 path; no opaque replacement quantization was accepted.
- A same-base BF16 MTP sidecar supplies speculative decoding. Only the draft LM head and five draft linears are converted to symmetric INT4 G128 at startup; the unchanged target verifies every emitted token.
- MTP4 raises short-output decode well above the GGUF paths measured on this host: 12.0 tok/s with Vulkan, 29.9 tok/s with SYCL/MTP2, and 80–119 tok/s depending on the selected vLLM workload.
- FP8 KV with an explicit 8.5 GB reservation provides 209,523 tokens of cache capacity and a configured 196,608-token serving contract.
- SHA-256 automatic prefix caching fixes repeated agent/tool ingestion. It reduced an identical 32K turn from 21.87 s cold to 2.62 s and then 1.07 s as the cache tail became reusable.
- The scheduler remains at Intel's reference 8,192-token chunk. A measured 16,384-token A/B made 100K cold prefill 2.0% slower and consumed more energy.

Cold long-context prefill remains the principal limitation. Qwen3.8-27B has 16 full-attention layers without a sliding window, and the current single-XPU path drops from about 1,566 tok/s at 8K to 851 tok/s at 100K. Prefix caching helps unchanged multi-turn history; it cannot make the first ingestion free.

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

- [Power efficiency](docs/power-efficiency.md): sweep, selection, persistence and rollback.
- [Prefill and prefix caching](docs/prefill-and-prefix-cache.md): TTFT scaling, tool-flow result and remaining bottleneck.
- [Architecture](docs/architecture.md): exact engine, overlay, versions and hashes.
- [Operations](docs/operations.md): lifecycle, requests, validation and rollback.
- [Benchmarks and quality](docs/benchmarks-and-quality.md): complete evidence and caveats.
- [Research and pitfalls](docs/research-and-pitfalls.md): rejected paths and lessons.
- [Token-0 repetition incident](docs/repetition-incident.md): exact sanitized reproducer, evidence, recovery and pending A/B.
- [XPU GDN source audit](docs/xpu-gdn-source-audit.md): version forensics, kernel narrowing and controlled correction order.
- [Rejected GDN maintenance trial](docs/gdn-maintenance-window-2026-08-24.md): direct probe, failed scheduler A/B and rollback gates.
- [GDN `64*N+5` investigation pause](docs/gdn-64n5-investigation-pause-2026-08-25.md): authoritative evidence boundary, provenance, artifact manifest, rejected hypotheses and restart point.
- [GDN diagnostic code map](docs/gdn-diagnostic-code-map.md): trusted, negative-control, superseded and observer-invalid instrumentation.
- [Upstream issue #548](https://github.com/vllm-project/vllm-xpu-kernels/issues/548): submitted Intel/XPU escalation; the [archived body](upstream/vllm-xpu-kernels-64n5-nan-issue.md) is versioned here.
- [Huihui plan](docs/huihui-plan.md): abliterated-model A/B and promotion gate.
- [References](docs/references.md): upstream sources and revisions.
