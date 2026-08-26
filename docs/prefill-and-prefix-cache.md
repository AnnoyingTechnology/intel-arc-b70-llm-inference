# Prefill and prefix caching

## Outcome

The severe repeated-tool latency came from an explicit `--no-enable-prefix-caching` setting. The service now uses:

```text
--enable-prefix-caching
--prefix-caching-hash-algo sha256
--prefix-match-unit 64
--max-num-batched-tokens 8192
```

On a controlled 32K agent-like conversation, cold TTFT was 21.87 s. Repeating the same request took 2.62 s with 29,952 cached tokens, then 1.07 s with 31,616 cached tokens. A synthetic two-step tool flow correctly called `lookup_record`, accepted its result, and answered with the exact value; the 32,911-token follow-up completed in 3.52 s with 29,952 cached tokens.

The XPU hybrid cache aligns physical attention pages to the GDN/Mamba state page, yielding 1,664-token physical blocks. The 64-token match unit permits finer logical prefix matching, but only complete reusable physical/state checkpoints can be restored. This explains why the cached tail improves over successive turns rather than making every appended token free immediately.

Evidence:

- `results/prefix-cache/cap-275-agentlike-p32k-g8-match64.json`
- `results/prefix-cache/cap-210-tool-followup-p32k.json`
- `scripts/prefix_tool_bench.py`

## Cold-prefill scaling at 210 W

| Prompt tokens | TTFT | Effective prompt rate | Cache |
|---:|---:|---:|---:|
| 8,156 | 5.21 s median | 1,566 tok/s | cold, five prompt families |
| 32,565 | 25.05 s | 1,300 tok/s | cold, unique prefix |
| 99,889 | 117.35 s | 851 tok/s | cold, unique prefix |

The 100K result is intentionally cache-isolated. It shows that first-time project ingestion remains expensive even though subsequent tool turns are now much faster.

Qwen3.8-27B has 64 layers, with full attention every fourth layer: 16 full-attention layers and no sliding window. Their cost grows with context, while GDN layers and fixed per-request overhead add their own work. Prefix caching reuses unchanged history; it does not change the complexity of a truly new 100K prompt.

## Scheduler A/B

The final scheduler budget remains 8,192 tokens, matching Intel's current B70/Qwen reference commands.

At 210 W and approximately 100K cold input:

| Scheduler budget | TTFT | Prompt rate | Prefill J/input tok |
|---:|---:|---:|---:|
| **8,192** | **117.35 s** | **851.2 tok/s** | **0.2465** |
| 16,384 | 119.78 s | 833.9 tok/s | 0.2498 |

The larger chunk was 2.0% slower, used 1.3% more energy/token, and doubled first-time graph-specialization startup. It was rejected and the service was restored to 8,192.

## Remaining optimization boundary

The service already uses vLLM's XPU GDN custom operation from `vllm-xpu-kernels 0.1.12.3`. No server flag exposed a faster single-XPU long-prefill backend in the pinned build. Upstream XPU work on Qwen GDN kernels remains active; future image/kernel releases should be retested with the same 8K/32K/100K fixtures before adoption.

For OpenCode, keep stable system/project context at the beginning of the conversation and append tool turns. Restarting the server clears the in-memory prefix cache; changing early prompt content or tool schemas also invalidates the reusable prefix.

## Long-session eviction warning

Automatic prefix caching does not reserve memory per OpenCode session. On
2026-08-26, one approximately 21K side-agent turn generated two cold API
requests and made a warm approximately 157K main session miss its entire usable
prefix. The next main request reused the rebuilt prefix, proving that the cold
prefill was real.

The current hybrid-cache setting, `prefix_cache_retention_interval=0`, keeps
only semantic GDN/Mamba replay checkpoints. Under LRU pressure, losing the
latest usable recurrent checkpoint can invalidate reuse of the whole hybrid
prefix rather than merely its tail. This mechanism is strongly supported by
the installed source but still needs an event-level reproduction and eviction
trace. Do not experiment with the active 162K session. See
[`prefix-cache-eviction-incident-2026-08-26.md`](prefix-cache-eviction-incident-2026-08-26.md)
for exact accounting, evidence limits and the focused A/B plan.
