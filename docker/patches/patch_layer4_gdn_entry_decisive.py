#!/usr/bin/env python3
"""Capture exact layer-4 tensors at the Python/C++ GDN handoff."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_LAYER4_GDN_ENTRY_DECISIVE_V1"
IMPORT_ANCHOR = """from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
"""
IMPORT_REPLACEMENT = """from collections.abc import Callable
from typing import TYPE_CHECKING

# B70_LAYER4_GDN_ENTRY_DECISIVE_V1
import json
import os
import threading
import time
from pathlib import Path

import torch

_B70_GDN_ENTRY_DIR = Path(
    os.getenv("B70_GDN_ENTRY_DIR", "/b70-gdn-entry-captures")
)
_B70_GDN_ENTRY_RECORD = None
_B70_GDN_ENTRY_WATCHER = None


def _b70_gdn_entry_watcher():
    trigger = _B70_GDN_ENTRY_DIR / "trigger.json"
    while True:
        try:
            if not trigger.exists():
                time.sleep(0.05)
                continue
            request = json.loads(trigger.read_text())
            trigger.unlink()
            torch.xpu.synchronize()
            record = _B70_GDN_ENTRY_RECORD
            if record is None:
                continue
            torch.save(
                {
                    "label": request.get("label", "layer4-n5-entry"),
                    "metadata": record["metadata"],
                    "tensor_metadata": {
                        name: {
                            "shape": list(value.shape),
                            "stride": list(value.stride()),
                            "dtype": str(value.dtype),
                            "storage_offset": value.storage_offset(),
                        }
                        for name, value in record["refs"].items()
                    },
                    "tensors": {
                        name: value.detach().cpu()
                        for name, value in record["refs"].items()
                    },
                },
                _B70_GDN_ENTRY_DIR
                / f"{request.get('label', 'layer4-n5-entry')}.pt",
            )
        except Exception as error:
            (_B70_GDN_ENTRY_DIR / "watcher-error.txt").write_text(repr(error))
            time.sleep(0.1)


def _b70_start_gdn_entry_watcher():
    global _B70_GDN_ENTRY_WATCHER
    if _B70_GDN_ENTRY_WATCHER is not None:
        return
    _B70_GDN_ENTRY_DIR.mkdir(parents=True, exist_ok=True)
    _B70_GDN_ENTRY_WATCHER = threading.Thread(
        target=_b70_gdn_entry_watcher, daemon=True
    )
    _B70_GDN_ENTRY_WATCHER.start()
"""

CALL_ANCHOR = """    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
"""
CALL_REPLACEMENT = """    global _B70_GDN_ENTRY_RECORD
    _b70_entry = None
    if (
        num_actual_tokens == 5
        and num_prefills == 1
        and num_decodes == 0
        and num_spec_decodes == 0
        and self.prefix.endswith(".layers.4.linear_attn")
    ):
        _b70_entry = getattr(self, "_b70_layer4_gdn_entry", None)
        if _b70_entry is None:
            device = projected_states_qkvz.device
            _b70_entry = {
                "projected_qkvz": torch.full_like(
                    projected_states_qkvz, float("nan")
                ),
                "projected_ba": torch.full_like(
                    projected_states_ba, float("nan")
                ),
                "ssm_pre": torch.full(
                    (1, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                    float("nan"),
                    dtype=self.kv_cache[1].dtype,
                    device=device,
                ),
                "core_post": torch.full_like(core_attn_out, float("nan")),
                "z_post": torch.full_like(z, float("nan")),
                "state_index": torch.zeros((1,), dtype=torch.int32, device=device),
                "query_start": torch.zeros((2,), dtype=torch.int32, device=device),
                "has_initial_state": torch.zeros(
                    (1,), dtype=torch.bool, device=device
                ),
                "entry_marker": torch.zeros(
                    (1,), dtype=torch.int32, device=device
                ),
                "post_marker": torch.zeros(
                    (1,), dtype=torch.int32, device=device
                ),
            }
            self._b70_layer4_gdn_entry = _b70_entry
        _b70_state_index = non_spec_state_indices_tensor.reshape(-1).to(
            torch.int64
        )
        _b70_entry["projected_qkvz"].copy_(projected_states_qkvz)
        _b70_entry["projected_ba"].copy_(projected_states_ba)
        _b70_entry["ssm_pre"].copy_(
            self.kv_cache[1].index_select(0, _b70_state_index)
        )
        _b70_entry["state_index"].copy_(
            non_spec_state_indices_tensor.reshape(-1)
        )
        _b70_entry["query_start"].copy_(
            non_spec_query_start_loc.reshape(-1)
        )
        _b70_entry["has_initial_state"].copy_(
            has_initial_state.reshape(-1)
        )
        _b70_entry["entry_marker"].fill_(1)
        _B70_GDN_ENTRY_RECORD = {
            "refs": _b70_entry,
            "metadata": {
                "prefix": self.prefix,
                "num_actual_tokens": num_actual_tokens,
                "physical_rows": projected_states_qkvz.shape[0],
                "num_prefills": num_prefills,
                "num_decodes": num_decodes,
                "num_spec_decodes": num_spec_decodes,
                "num_k_heads": self.num_k_heads,
                "num_v_heads": self.num_v_heads,
                "head_k_dim": self.head_k_dim,
                "head_v_dim": self.head_v_dim,
                "tp_size": self.tp_size,
                "reorder_input": not self.gqa_interleaved_layout,
            },
        }
        _b70_start_gdn_entry_watcher()

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
"""

CALL_END_ANCHOR = """        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )
"""
CALL_END_REPLACEMENT = CALL_END_ANCHOR + """    if _b70_entry is not None:
        _b70_entry["core_post"].copy_(core_attn_out)
        _b70_entry["z_post"].copy_(z)
        _b70_entry["post_marker"].fill_(2)
"""


def main() -> None:
    spec = importlib.util.find_spec("vllm._xpu_ops")
    if spec is None or spec.origin is None:
        raise SystemExit("vllm._xpu_ops not found")
    path = Path(spec.origin)
    text = path.read_text()
    if MARKER in text:
        print(f"already patched {path}")
        return
    for anchor, replacement, label in (
        (IMPORT_ANCHOR, IMPORT_REPLACEMENT, "imports"),
        (CALL_ANCHOR, CALL_REPLACEMENT, "call entry"),
        (CALL_END_ANCHOR, CALL_END_REPLACEMENT, "call exit"),
    ):
        if text.count(anchor) != 1:
            raise RuntimeError(f"unexpected {label} anchor")
        text = text.replace(anchor, replacement, 1)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
