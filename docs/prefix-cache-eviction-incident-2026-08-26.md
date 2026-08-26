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
