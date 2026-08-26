# Long-session prefix-cache eviction incident — 2026-08-26

## Status

Investigation required. Do not treat automatic prefix caching as protection for
an important long-running session. This incident is independent of the fixed
GDN `64*N+5` dispatch bug.

No request, restart or configuration change was made while documenting the
incident. Only existing OpenCode accounting, vLLM metrics, logs, configuration
and installed source were read.

## Observed sequence

The main OpenCode session was approximately 157K tokens immediately before the
incident and has since grown to approximately 162K. A second OpenCode instance
received one user question. Its agent made a tool call, so that single UI turn
produced two API requests before it was stopped.

OpenCode's stored assistant accounting shows:

| Session/request | Total tokens | New input | Cache read | Interpretation |
|---|---:|---:|---:|---|
| Main, earlier turn | 154,124 | 6,322 | 145,920 | Long prefix reused |
| Main, last turn before side session | 157,057 | 1,869 | 154,240 | Long prefix reused |
| Side, initial request | 20,980 | 20,831 | 0 | Cold request |
| Side, automatic post-tool request | 23,188 | 22,888 | 0 | Second cold request |
| Main, first resumed request | 160,015 | 157,712 | 0 | Entire usable prefix missed |
| Main, immediate continuation | 161,504 | 2,696 | 157,696 | Rebuilt prefix reused |

The main session therefore suffered a real cache miss and cold prefill. Its
immediate next request reused the rebuilt prefix. Prompt contents were not read
or recorded.

## Runtime facts

The running server reported:

```text
kv_cache_size_tokens=209523
kv_cache_memory_bytes=8500000000
kv_cache_max_concurrency=1.0656934306569343
block_size=1664
mamba_block_size=1664
prefix_match_unit=64
prefix_cache_retention_interval=0
enable_prefix_caching=True
```

Compose sets `--max-num-seqs 1`. This limits simultaneously executing requests;
it does not reserve cache capacity for one session. The approximately 157K main
prefix and two independent 21–23K side prefixes approached the nominal 209.5K
pool before physical-block rounding, generation headroom and hybrid-cache
constraints.

The cumulative metrics showed a high global prefix hit rate, which does not
contradict this incident: aggregate hits conceal one expensive full miss.
`num_preemptions_total=2` was also present, but it has not been attributed to
these requests.

## Source-backed mechanism

The installed vLLM implementation uses a shared block pool with LRU eviction.
Inactive prefixes are reclaimable; they are not session reservations. Cached
block chains are returned to the free queue, and allocation evicts cached
mappings as needed. Equal-age chain tails have higher eviction priority.

This model combines full-attention and GDN/Mamba cache groups. Their reusable
prefix lengths must be reconciled. A full-attention group retaining KV blocks is
insufficient when the recurrent group no longer has a state from which replay
can begin.

The decisive current setting is:

```text
prefix_cache_retention_interval=0
```

Installed-source semantics are:

- `0`: retain only semantic checkpoints, including the latest replay boundary
  and shared-prefix junctions;
- a positive multiple of the 8,192-token scheduler block: additionally retain
  periodic GDN/Mamba checkpoints;
- `None`: retain checkpoints densely.

Consequently, memory pressure can evict the latest usable recurrent checkpoint.
The hybrid coordinator can then reject the entire long prefix even if some
attention KV blocks remain. This explains the observed zero hit followed by
normal reuse after rebuilding much better than ordinary gradual tail eviction.

This is a strong source-supported diagnosis, not direct observation of the
specific eviction decision. The event-level block IDs and eviction trace were
not captured.

## Unresolved questions

1. Why did the side session's automatic post-tool request report zero reuse of
   its own approximately 21K prefix? This doubled the pressure. Determine
   whether OpenCode changed an early request component, vLLM found no common
   hybrid checkpoint, or another cache-key input differed.
2. Did the main prefix lose one decisive GDN checkpoint, or did capacity
   pressure remove a broader chain? Capture cache-manager decisions to prove it.
3. Does `prefix_cache_retention_interval=8192`, `32768` or another positive
   value provide graceful fallback without reducing useful capacity enough to
   cause earlier eviction?
4. Can one important session be pinned, protected by quota, or offloaded without
   compromising hybrid GDN correctness? Automatic prefix caching currently
   offers no session reservation.
5. Are the two cumulative preemptions related? Correlate request timestamps with
   scheduler logs before drawing a conclusion.

## Thinking-level cache invalidation

Changing OpenCode's `reasoningEffort` also prevents reuse of the existing
prefix for the next request. This is not merely a sampling parameter in this
deployment: Qwen's chat template renders `low` and `xhigh` as different text at
the beginning of the system block, renders no extra reasoning instruction for
`medium`, and changes the thinking wrapper for `off`. The first 64-token cache
block therefore differs, so vLLM cannot match the later 162K suffix.

The old variant's cache is not explicitly deleted. Switching effort creates a
separate prefix branch; switching back can reuse the old branch only if LRU
pressure has not evicted it. With the current nearly full hybrid cache and
sparse recurrent checkpoints, assume that changing effort can force a complete
cold prefill and can endanger the original warm branch.

This was verified by rendering the same synthetic message locally with the
deployed tokenizer/template for `off`, `low`, `medium` and `xhigh`. No inference
request was sent.

## OpenCode compaction cold-prefill

An observed local-B70 session reached 180,980 tokens and entered compaction.
This boundary is expected: the configured 196,608-token context minus the
16,384-token output allowance is 180,224 tokens. The configured 10,000-token
compaction reserve is smaller and therefore does not determine the boundary.

The preceding ordinary assistant request reused 176,000 cached tokens. The
automatic compaction request then processed 137,459 input tokens with zero
cache read and took 443.031 seconds (7 minutes 23.031 seconds). This is direct
local evidence of the latency cliff, not an estimate.

OpenCode 1.18.4 already puts its compaction instruction after the selected
conversation history. Prompt placement is not the cache failure. Its compaction
request also does all of the following:

- passes `system: []` instead of the normal agent system prompt;
- passes `tools: {}` instead of the normal tool schemas;
- serializes historical tool output with a 2,000-character cap;
- preserves the triggering message's reasoning variant.

For Qwen, system text and tool schemas are rendered at the beginning of the
token stream. Removing them changes the first cache block. Truncating any early
tool result creates another divergence. The appended instruction therefore
cannot reuse the warm approximately 181K prefix despite being appended.

The `experimental.session.compacting` plugin hook can replace the appended
prompt or add context. It cannot restore the original system/tool envelope or
disable historical tool-output truncation, so it cannot implement
cache-preserving compaction alone.

A proper cache-preserving compaction path should rebuild the exact ordinary
request prefix—same system text, tool definitions, reasoning variant, cache
salt and unmodified selected history—then append the summary request. Tool
execution should be disabled through a non-tokenized request control such as
`tool_choice: none`, while retaining identical tool definitions for template
rendering. After the summary is stored, the next ordinary turn will necessarily
start from the new short summary and be cold, but that prefill is cheap. The
expensive summary generation should have reused nearly all selected history.

This requires an OpenCode change, not a vLLM setting. Validate it first with a
synthetic long session and require the compaction request's `cached_tokens` to
cover the unchanged selected head. Also test long tool outputs, thinking-level
stability, summary correctness and the post-compaction continuation. A separate
fast compaction model is the available configuration-level alternative, but it
avoids rather than fixes B70 prefix reuse and may move private history to another
provider.

### Upstream status and safe shape

The problem is already tracked upstream:

- [issue #43249](https://github.com/anomalyco/opencode/issues/43249) identifies
  the divergent compaction prefix and shared cache key;
- [PR #42506](https://github.com/anomalyco/opencode/pull/42506) replays the
  ordinary system, tools and typed history before appending the summary prompt;
- the earlier [PR #25100](https://github.com/anomalyco/opencode/pull/25100)
  measured an 84.9% compaction cache hit but was closed only by stale cleanup;
- [PR #40800](https://github.com/anomalyco/opencode/pull/40800) explains why
  current OpenCode serializes compaction history: sliced structured history can
  expose orphaned tool calls to strict providers.

PR #42506 has a successful Qwen/Ollama cache-hit report, but its current form
changes the default for every provider and does not request `tool_choice: none`.
The upstream-safe version should be opt-in, for example
`compaction.preserve_prefix_cache: true`, while preserving the serialized path
as the default and fallback. The first scope should be same-model,
OpenAI-compatible requests with a clean turn-boundary split, no overflow
recovery and no reordered prior-compaction history. It should retain the full
tool definitions and send `tool_choice: none`.

This restriction is deliberate. vLLM includes tool definitions in the prompt
when `tool_choice` is `none` unless started with
`--exclude-tools-when-tool-choice-none`; this deployment does not set that
option. Thus tool calls can be disabled without changing the cached Qwen token
prefix. Anthropic, Bedrock and Gemini lower `tool_choice: none` differently and
must keep the existing serialized fallback until separately proven safe.

An offline check with the deployed Qwen tokenizer produced identical rendered
prompts for `tool_choice: auto` and `tool_choice: none`: 303 tokens in both
cases. No inference request was sent.

An adversarial review by Claude Opus 5 agreed that exact prefix reuse is
feasible, but rejected a universal default because it can reopen the orphaned
tool-call and signed-reasoning/provider-ordering failures fixed by #40800. It
also identified request-building duplication as a future cache-drift risk: the
ordinary and compaction paths should share one prefix-assembly helper.

The reviewed implementation is public at
[AnnoyingTechnology/opencode:cache-compaction](https://github.com/AnnoyingTechnology/opencode/tree/cache-compaction),
commit `e41db562e`. It keeps serialized compaction as the default and adds the
guarded `compaction.preserve_prefix_cache` option for same-model,
OpenAI-compatible requests. The branch includes generated schema/SDK changes,
prefix-equality tests and fallback tests. Validation passed: 119 tests, two
expected skips, and the repository-wide typecheck. It is linked from
[PR #42506](https://github.com/anomalyco/opencode/pull/42506#issuecomment-5425656358);
do not open a competing PR. Let its author adopt or revise the implementation,
then tag a maintainer only if the thread stalls.

No live inference request, service restart or local OpenCode replacement was
performed during this upstream review.

## Focused reproduction plan

Run only after the active OpenCode work is disposable or preserved elsewhere.
Do not use the 162K session as a test fixture.

1. Build synthetic main and side conversations with fixed, non-sensitive
   prefixes. Record API `cached_tokens`, request timestamps and cache metrics.
2. Reproduce the sequence: grow main near 157K, idle it, submit the same
   two-stage 21–23K tool flow, then resume main.
3. Add narrow cache-manager logging for allocations, freed/evicted block hashes,
   group identity, retained Mamba boundaries and reconciled hit length. Avoid
   tensor or prompt capture.
4. Establish the baseline repeatedly with retention `0`.
5. A/B positive retention intervals that are multiples of 8,192. Measure:
   resumed-main cached tokens and TTFT; side-flow reuse; effective cache
   capacity; recurrent-state overhead; decode and prefill regressions.
6. Select a setting only if it converts catastrophic zero-hit recovery into a
   bounded replay while preserving the 196,608-token serving contract and the
   existing correctness/performance gates.

## Candidate mitigation, not yet validated

A positive retention interval should preserve periodic GDN/Mamba replay points.
`8192` offers the finest fallback but retains the most recurrent state;
`32768` retains less state with a coarser replay boundary. `None` is dense and
may consume too much of a 32 GiB device. These are test candidates, not current
recommendations.

Until measured, avoid starting an unrelated large OpenCode conversation while
the main session's warm 162K prefix matters.
