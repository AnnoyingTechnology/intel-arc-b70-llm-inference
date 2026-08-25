#!/usr/bin/env python3
"""Install a finite-sentinel, marker-stamped layer-4 n5 in-op GDN capture.

The capture buffers are registered during model construction. The live custom
op performs no allocation or synchronization between causal_conv1d and
gated_delta_rule. A single post-operation synchronization keeps all source
tensors alive until the diagnostic copies have completed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_LAYER4_GDN_INOP_PERSISTENT_V3"


def package_file(module: str, relative: str = "") -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise RuntimeError(f"cannot locate {module}")
    if relative:
        if not spec.submodule_search_locations:
            raise RuntimeError(f"cannot locate package directory for {module}")
        return Path(next(iter(spec.submodule_search_locations))) / relative
    if spec.origin is None:
        raise RuntimeError(f"cannot locate source for {module}")
    return Path(spec.origin)


def patch_model() -> None:
    path = package_file(
        "vllm",
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    )
    text = path.read_text()
    if MARKER in text:
        return

    anchor = """        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)
"""
    replacement = anchor + r'''

        # B70_LAYER4_GDN_INOP_PERSISTENT_V3
        self._b70_gdn_inop_debug = None
        self._b70_gdn_inop_armed = False
        self._b70_gdn_inop_dispatch_id = 0
        self._b70_gdn_inop_host_metadata = None
        if prefix.endswith(".layers.4.linear_attn"):
            dtype = vllm_config.model_config.dtype
            specs = {
                "input_qkvz": (8 * 16384, dtype),
                "input_ba": (8 * 96, dtype),
                "q": (68 * 16 * 128, dtype),
                "k": (68 * 16 * 128, dtype),
                "v": (68 * 48 * 128, dtype),
                "b": (48 * 68, torch.float32),
                "a": (48 * 68, torch.float32),
                "core": (8 * 48 * 128, dtype),
                "z": (8 * 48 * 128, dtype),
            }
            buffers = {}
            sentinel = 101.0
            for name, (size, buffer_dtype) in specs.items():
                buffer_name = f"_b70_gdn_inop_{name}"
                self.register_buffer(
                    buffer_name,
                    torch.full((size,), sentinel, dtype=buffer_dtype),
                    persistent=False,
                )
                buffers[name] = getattr(self, buffer_name)
                sentinel += 1.0
            self.register_buffer(
                "_b70_gdn_inop_marker",
                torch.zeros((32,), dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "_b70_gdn_inop_metadata",
                torch.full((32,), -1, dtype=torch.int32),
                persistent=False,
            )
            buffers["marker"] = self._b70_gdn_inop_marker
            buffers["metadata"] = self._b70_gdn_inop_metadata
            self._b70_gdn_inop_debug = buffers
'''
    if text.count(anchor) != 1:
        raise RuntimeError("unexpected qwen GDN init anchor")
    text = text.replace(anchor, replacement, 1)
    compile(text, str(path), "exec")
    path.write_text(text)


def patch_xpu_op() -> None:
    path = package_file("vllm._xpu_ops")
    text = path.read_text()
    if MARKER in text:
        return

    import_anchor = """from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
"""
    import_replacement = """from collections.abc import Callable
from typing import TYPE_CHECKING

# B70_LAYER4_GDN_INOP_PERSISTENT_V3
import json
import os
import threading
import time
from pathlib import Path

import torch
"""
    if text.count(import_anchor) != 1:
        raise RuntimeError("unexpected vllm._xpu_ops import anchor")
    text = text.replace(import_anchor, import_replacement, 1)

    logger_anchor = """logger = init_logger(__name__)

if TYPE_CHECKING:
"""
    logger_replacement = r'''logger = init_logger(__name__)

_B70_GDN_INOP_DIR = Path(
    os.getenv("B70_GDN_INOP_DIR", "/b70-gdn-inop-captures")
)
_B70_GDN_INOP_RECORD = None
_B70_GDN_INOP_WATCHER = None


def _b70_gdn_inop_start_watcher() -> None:
    global _B70_GDN_INOP_WATCHER
    if _B70_GDN_INOP_WATCHER is not None:
        return

    def watch() -> None:
        arm_path = _B70_GDN_INOP_DIR / "arm.json"
        dump_path = _B70_GDN_INOP_DIR / "dump.json"
        while True:
            try:
                if arm_path.exists():
                    request = json.loads(arm_path.read_text())
                    arm_path.unlink()
                    record = _B70_GDN_INOP_RECORD
                    if record is not None:
                        record._b70_gdn_inop_label = request.get(
                            "label", "layer4-n5-inop"
                        )
                        record._b70_gdn_inop_armed = True
                if dump_path.exists():
                    request = json.loads(dump_path.read_text())
                    dump_path.unlink()
                    torch.xpu.synchronize()
                    record = _B70_GDN_INOP_RECORD
                    if record is not None:
                        label = request.get(
                            "label",
                            getattr(
                                record,
                                "_b70_gdn_inop_label",
                                "layer4-n5-inop",
                            ),
                        )
                        buffers = record._b70_gdn_inop_debug
                        payload = {
                            "schema_version": 1,
                            "label": label,
                            "host_metadata": record._b70_gdn_inop_host_metadata,
                            "buffer_metadata": {
                                name: {
                                    "shape": list(value.shape),
                                    "stride": list(value.stride()),
                                    "dtype": str(value.dtype),
                                    "storage_offset": value.storage_offset(),
                                }
                                for name, value in buffers.items()
                            },
                            "tensors": {
                                name: value.detach().cpu()
                                for name, value in buffers.items()
                            },
                        }
                        torch.save(payload, _B70_GDN_INOP_DIR / f"{label}.pt")
                time.sleep(0.02)
            except Exception as error:
                (_B70_GDN_INOP_DIR / "watcher-error.txt").write_text(
                    repr(error)
                )
                time.sleep(0.1)

    _B70_GDN_INOP_DIR.mkdir(parents=True, exist_ok=True)
    _B70_GDN_INOP_WATCHER = threading.Thread(
        target=watch,
        name="b70-gdn-inop-capture",
        daemon=True,
    )
    _B70_GDN_INOP_WATCHER.start()


if TYPE_CHECKING:
'''
    if text.count(logger_anchor) != 1:
        raise RuntimeError("unexpected vllm._xpu_ops logger anchor")
    text = text.replace(logger_anchor, logger_replacement, 1)

    call_anchor = """    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
"""
    call_replacement = r'''    global _B70_GDN_INOP_RECORD
    _b70_debug = getattr(self, "_b70_gdn_inop_debug", None)
    if _b70_debug is not None:
        _B70_GDN_INOP_RECORD = self
        _b70_gdn_inop_start_watcher()

    _b70_capture = (
        _b70_debug is not None
        and num_actual_tokens == 5
        and num_prefills == 1
        and num_decodes == 0
        and num_spec_decodes == 0
        and self.prefix.endswith(".layers.4.linear_attn")
    )
    if _b70_capture:
        self._b70_gdn_inop_dispatch_id += 1
        _b70_debug["input_qkvz"].narrow(
            0, 0, projected_states_qkvz.numel()
        ).copy_(projected_states_qkvz.flatten())
        _b70_debug["input_ba"].narrow(
            0, 0, projected_states_ba.numel()
        ).copy_(projected_states_ba.flatten())
        _b70_debug["metadata"].narrow(
            0, 0, non_spec_query_start_loc.numel()
        ).copy_(non_spec_query_start_loc.flatten())
        _b70_debug["metadata"].select(0, 4).copy_(
            non_spec_state_indices_tensor.reshape(-1)[0]
        )
        _b70_debug["metadata"].select(0, 5).copy_(
            has_initial_state.reshape(-1)[0]
        )
        _b70_debug["metadata"].select(0, 6).fill_(num_actual_tokens)
        _b70_debug["metadata"].select(0, 7).fill_(num_prefills)
        _b70_debug["metadata"].select(0, 8).fill_(num_decodes)
        _b70_debug["metadata"].select(0, 9).fill_(num_spec_decodes)
        _b70_debug["metadata"].select(0, 10).fill_(
            -1 if non_spec_token_indx is None else non_spec_token_indx.numel()
        )
        _b70_debug["marker"].select(0, 12).fill_(
            self._b70_gdn_inop_dispatch_id
        )
        _b70_debug["marker"].select(0, 13).fill_(1)

        def _b70_desc(value):
            return {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "storage_offset": value.storage_offset(),
                "data_ptr": value.data_ptr(),
                "storage_ptr": value.untyped_storage().data_ptr(),
                "storage_nbytes": value.untyped_storage().nbytes(),
            }

        self._b70_gdn_inop_host_metadata = {
            "prefix": self.prefix,
            "dispatch_id": self._b70_gdn_inop_dispatch_id,
            "runtime_mode": forward_context.cudagraph_runtime_mode.name,
            "num_actual_tokens": num_actual_tokens,
            "num_prefills": num_prefills,
            "num_decodes": num_decodes,
            "num_spec_decodes": num_spec_decodes,
            "non_spec_token_indx_none": non_spec_token_indx is None,
            "spec_query_start_loc_none": spec_query_start_loc is None,
            "spec_token_indx_none": spec_token_indx is None,
            "spec_state_indices_none": spec_state_indices_tensor is None,
            "num_accepted_tokens_none": num_accepted_tokens is None,
            "projected_qkvz": _b70_desc(projected_states_qkvz),
            "projected_ba": _b70_desc(projected_states_ba),
            "core": _b70_desc(core_attn_out),
            "z": _b70_desc(z),
            "conv_state": _b70_desc(self.kv_cache[0]),
            "ssm_state": _b70_desc(self.kv_cache[1]),
            "debug": {
                name: _b70_desc(value)
                for name, value in _b70_debug.items()
            },
        }

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
'''
    if text.count(call_anchor) != 1:
        raise RuntimeError("unexpected fused GDN call anchor")
    text = text.replace(call_anchor, call_replacement, 1)

    end_anchor = """        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )
"""
    end_replacement = """        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
        debug_q=None if not _b70_capture else _b70_debug[\"q\"],
        debug_k=None if not _b70_capture else _b70_debug[\"k\"],
        debug_v=None if not _b70_capture else _b70_debug[\"v\"],
        debug_b=None if not _b70_capture else _b70_debug[\"b\"],
        debug_a=None if not _b70_capture else _b70_debug[\"a\"],
        debug_marker=None if not _b70_capture else _b70_debug[\"marker\"],
    )
    if _b70_capture:
        _b70_debug[\"core\"].narrow(0, 0, core_attn_out.numel()).copy_(
            core_attn_out.flatten()
        )
        _b70_debug[\"z\"].narrow(0, 0, z.numel()).copy_(z.flatten())
        _b70_debug[\"marker\"].select(0, 14).fill_(3)
        # Retain qkvz/ba and the C++ q/k/v/b/a workspaces until all diagnostic
        # copies complete. Prior post-GDN sync controls did not cure n5.
        torch.xpu.synchronize()
"""
    if text.count(end_anchor) != 1:
        raise RuntimeError("unexpected fused GDN call-end anchor")
    text = text.replace(end_anchor, end_replacement, 1)
    compile(text, str(path), "exec")
    path.write_text(text)


def main() -> None:
    patch_model()
    patch_xpu_op()
    print("[B70] persistent layer-4 n5 in-op GDN capture installed")


if __name__ == "__main__":
    main()
