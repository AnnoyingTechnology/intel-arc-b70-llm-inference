#!/usr/bin/env python3
"""Split the vLLM XPU GDN prefill shape that produces non-finite output.

The pinned ``vllm-xpu-kernels`` prefill operation is deterministic at prompt
lengths ``64*N+5``: the final partial GDN chunk returns NaN.  During prompt
processing only, leave the four known MTP-lookahead tokens for a second
scheduler step.  The first step then ends at ``64*N+1`` and the second has
length four; both are proven-finite shapes.

This keeps MTP4 enabled and changes no model, cache, sampler or kernel setting.
It is deliberately a startup patch with a strict source anchor so an upstream
scheduler change fails closed instead of applying an ambiguous rewrite.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MARKER = "B70_XPU_GDN_TAIL_64N_PLUS_5"

OLD = '''        remaining = request.num_tokens - num_computed_tokens - num_new_tokens
        if 0 < remaining < self.num_prefill_lookahead:
            num_new_tokens -= self.num_prefill_lookahead - remaining
        return max(num_new_tokens, 0)
'''

NEW = '''        remaining = request.num_tokens - num_computed_tokens - num_new_tokens
        if (
            num_computed_tokens < request.num_prompt_tokens
            and remaining == 0
            and self.num_prefill_lookahead == 4
            and num_new_tokens > self.num_prefill_lookahead
            and num_new_tokens % 64 == 5
        ):
            # B70_XPU_GDN_TAIL_64N_PLUS_5: the pinned XPU GDN chunk kernel
            # produces NaN for a final partial chunk of five tokens. Leave the
            # four known MTP-lookahead tokens for a second, finite prefill step.
            # The prompt-only condition excludes every MTP4 decode group.
            num_new_tokens -= self.num_prefill_lookahead
            remaining = self.num_prefill_lookahead
        if 0 < remaining < self.num_prefill_lookahead:
            num_new_tokens -= self.num_prefill_lookahead - remaining
        return max(num_new_tokens, 0)
'''


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    if text.count(OLD) != 1:
        raise RuntimeError("scheduler prefill-lookahead anchor changed; refusing to patch")
    return text.replace(OLD, NEW, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "v1/core/sched/scheduler.py"
    )
    original = path.read_text()
    patched = patch_text(original)
    if patched == original:
        print(f"already patched {path}")
        return
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
