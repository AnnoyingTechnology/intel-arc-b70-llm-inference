#!/usr/bin/env python3
"""Diagnostic only: synchronize before layer-4 GDN consumes projections."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_GDN_ENTRY_SYNC_DIAGNOSTIC_V1"


def main() -> None:
    spec = importlib.util.find_spec("vllm._xpu_ops")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate vllm._xpu_ops")
    path = Path(spec.origin)
    text = path.read_text()
    if MARKER in text:
        return
    anchor = """    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens
"""
    replacement = """    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens

    # B70_GDN_ENTRY_SYNC_DIAGNOSTIC_V1: diagnostic, never a proposed fix.
    if ".layers.4." in self.prefix:
        torch.xpu.synchronize()
"""
    if text.count(anchor) != 1:
        raise RuntimeError("unexpected GDN metadata anchor")
    path.write_text(text.replace(anchor, replacement, 1))
    print(f"patched {path}")


if __name__ == "__main__":
    main()
