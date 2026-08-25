#!/usr/bin/env python3
"""Capture the exact layer-4 n5 inputs on both sides of causal convolution."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_LAYER4_GDN_INPUTS_DECISIVE_V1"

IMPORT_ANCHOR = """from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
"""

IMPORT_REPLACEMENT = """from collections.abc import Callable
from typing import TYPE_CHECKING

# B70_LAYER4_GDN_INPUTS_DECISIVE_V1
import json
import os
import threading
import time
from pathlib import Path

import torch

_B70_GDN_INPUT_DIR = Path(
    os.getenv("B70_GDN_INPUT_DIR", "/b70-gdn-input-captures")
)
_B70_GDN_INPUT_RECORD = None
_B70_GDN_INPUT_WATCHER = None


def _b70_gdn_input_watcher():
    trigger = _B70_GDN_INPUT_DIR / "trigger.json"
    while True:
        try:
            if not trigger.exists():
                time.sleep(0.05)
                continue
            request = json.loads(trigger.read_text())
            trigger.unlink()
            torch.xpu.synchronize()
            record = _B70_GDN_INPUT_RECORD
            if record is None:
                continue
            refs = record["refs"]
            payload = {
                "label": request.get("label", "layer4-n5"),
                "metadata": record["metadata"],
                "tensor_metadata": {
                    name: {
                        "shape": list(value.shape),
                        "stride": list(value.stride()),
                        "dtype": str(value.dtype),
                        "storage_offset": value.storage_offset(),
                    }
                    for name, value in refs.items()
                },
                "tensors": {
                    name: value.detach().cpu() for name, value in refs.items()
                },
                "static": {
                    "conv_weights": record["module"].conv1d.weight.detach()
                    .view(
                        record["module"].conv1d.weight.size(0),
                        record["module"].conv1d.weight.size(2),
                    )
                    .cpu(),
                    "conv_bias": (
                        None
                        if record["module"].conv1d.bias is None
                        else record["module"].conv1d.bias.detach().cpu()
                    ),
                    "A_log": record["module"].A_log.detach().cpu(),
                    "dt_bias": record["module"].dt_bias.detach().cpu(),
                },
            }
            torch.save(
                payload,
                _B70_GDN_INPUT_DIR
                / f"{payload['label']}-layer4-n5.pt",
            )
        except Exception as error:
            (_B70_GDN_INPUT_DIR / "watcher-error.txt").write_text(repr(error))
            time.sleep(0.1)


def _b70_start_gdn_input_watcher():
    global _B70_GDN_INPUT_WATCHER
    if _B70_GDN_INPUT_WATCHER is not None:
        return
    _B70_GDN_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    _B70_GDN_INPUT_WATCHER = threading.Thread(
        target=_b70_gdn_input_watcher, daemon=True
    )
    _B70_GDN_INPUT_WATCHER.start()
"""

CALL_ANCHOR = """    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
"""

CALL_REPLACEMENT = """    global _B70_GDN_INPUT_RECORD
    _b70_debug = None
    if (
        num_actual_tokens == 5
        and num_prefills == 1
        and num_decodes == 0
        and num_spec_decodes == 0
        and self.prefix.endswith(".layers.4.linear_attn")
    ):
        _b70_debug = getattr(self, "_b70_layer4_gdn_input_debug", None)
        physical_rows = projected_states_qkvz.shape[0]
        virtual_rows = physical_rows + 63
        if _b70_debug is None:
            device = projected_states_qkvz.device
            dtype = projected_states_qkvz.dtype
            _b70_debug = {
                "projected_qkvz": torch.empty_like(projected_states_qkvz),
                "projected_ba": torch.empty_like(projected_states_ba),
                "q": torch.full(
                    (virtual_rows * self.num_k_heads * self.head_k_dim,),
                    float("nan"),
                    dtype=dtype,
                    device=device,
                ),
                "k": torch.full(
                    (virtual_rows * self.num_k_heads * self.head_k_dim,),
                    float("nan"),
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.full(
                    (virtual_rows * self.num_v_heads * self.head_v_dim,),
                    float("nan"),
                    dtype=dtype,
                    device=device,
                ),
                "b": torch.full(
                    (virtual_rows * self.num_v_heads,),
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                ),
                "a": torch.full(
                    (virtual_rows * self.num_v_heads,),
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                ),
                "ssm_pre": torch.empty(
                    (1, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                    dtype=self.kv_cache[1].dtype,
                    device=device,
                ),
                "ssm_post": torch.empty(
                    (1, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                    dtype=self.kv_cache[1].dtype,
                    device=device,
                ),
                "core_post": torch.empty_like(core_attn_out),
                "z_post": torch.empty_like(z),
                "state_index": torch.empty((1,), dtype=torch.int32, device=device),
                "query_start": torch.empty((2,), dtype=torch.int32, device=device),
                "has_initial_state": torch.empty(
                    (1,), dtype=torch.bool, device=device
                ),
                "pre_marker": torch.zeros((1,), dtype=torch.int32, device=device),
                "conv_marker": torch.zeros(
                    (16,), dtype=torch.int32, device=device
                ),
                "post_marker": torch.zeros(
                    (1,), dtype=torch.int32, device=device
                ),
            }
            self._b70_layer4_gdn_input_debug = _b70_debug
        _b70_state_index = non_spec_state_indices_tensor.reshape(-1).to(
            torch.int64
        )
        _b70_debug["projected_qkvz"].copy_(projected_states_qkvz)
        _b70_debug["projected_ba"].copy_(projected_states_ba)
        _b70_debug["ssm_pre"].copy_(
            self.kv_cache[1].index_select(0, _b70_state_index)
        )
        _b70_debug["state_index"].copy_(
            non_spec_state_indices_tensor.reshape(-1)
        )
        _b70_debug["query_start"].copy_(
            non_spec_query_start_loc.reshape(-1)
        )
        _b70_debug["has_initial_state"].copy_(
            has_initial_state.reshape(-1)
        )
        _b70_debug["pre_marker"].fill_(1)
        _B70_GDN_INPUT_RECORD = {
            "module": self,
            "refs": _b70_debug,
            "metadata": {
                "prefix": self.prefix,
                "num_actual_tokens": num_actual_tokens,
                "physical_rows": physical_rows,
                "virtual_rows": virtual_rows,
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
        _b70_start_gdn_input_watcher()

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
"""

CALL_END_ANCHOR = """        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )
"""

CALL_END_REPLACEMENT = """        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
        debug_q=None if _b70_debug is None else _b70_debug["q"],
        debug_k=None if _b70_debug is None else _b70_debug["k"],
        debug_v=None if _b70_debug is None else _b70_debug["v"],
        debug_b=None if _b70_debug is None else _b70_debug["b"],
        debug_a=None if _b70_debug is None else _b70_debug["a"],
        debug_marker=(
            None if _b70_debug is None else _b70_debug["conv_marker"]
        ),
    )
    if _b70_debug is not None:
        _b70_debug["ssm_post"].copy_(
            self.kv_cache[1].index_select(0, _b70_state_index)
        )
        _b70_debug["core_post"].copy_(core_attn_out)
        _b70_debug["z_post"].copy_(z)
        _b70_debug["post_marker"].fill_(3)
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
