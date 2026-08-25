#!/usr/bin/env python3
"""Install opt-in instrumentation at the real XPU GDN call boundary.

This is an investigation aid, not an inference workaround.  It does not
change scheduling, token counts, model values, or the GDN dispatch.  The
instrumentation is disabled unless ``B70_GDN_CAPTURE_MODE`` is set to one of:

``metadata``
    Record host-visible tensor layout and dispatch metadata without reading
    device tensor contents.

``survey``
    Enqueue finite reductions around every GDN layer, synchronize once after
    the final GDN layer, and record the first layer whose inputs, outputs, or
    selected recurrent state are non-finite.

``capture``
    For the selected layer, copy the exact pre/post tensors and metadata to a
    torch checkpoint suitable for offline replay.  This mode intentionally
    synchronizes and must be checked against the unchanged API oracle for
    observer effects.

``deferred``
    Retain graph-captured tensor/flag buffers and dump them only after an API
    request when ``trigger.json`` appears.  This preserves the real XPU-graph
    replay path used by the deterministic five-token failure.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_EXACT_GDN_CAPTURE_V1"

IMPORT_ANCHOR = """from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
"""

IMPORT_REPLACEMENT = """from collections.abc import Callable
from typing import TYPE_CHECKING

# B70_EXACT_GDN_CAPTURE_V1: opt-in diagnostics for the real fused XPU GDN call.
import json
import os
import re
import threading
import time
from pathlib import Path

import torch

_B70_GDN_MODE = os.getenv("B70_GDN_CAPTURE_MODE", "").strip().lower()
_B70_GDN_DIR = Path(os.getenv("B70_GDN_CAPTURE_DIR", "/b70-gdn-captures"))
_B70_GDN_LENGTHS = {
    int(value)
    for value in os.getenv("B70_GDN_CAPTURE_LENGTHS", "4,5,6").split(",")
    if value.strip()
}
_B70_GDN_LAYER = int(os.getenv("B70_GDN_CAPTURE_LAYER", "-1"))
_B70_GDN_LAST_LAYER = int(os.getenv("B70_GDN_LAST_LAYER", "62"))
_B70_GDN_PASS_ID = 0
_B70_GDN_CAPTURE_ID = 0
_B70_GDN_SURVEY = []
_B70_GDN_DEFERRED = {}
_B70_GDN_WATCHER = None


def _b70_layer_index(value: str) -> int:
    match = re.search(r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)", value)
    return int(match.group(1)) if match else -1


def _b70_tensor_desc(tensor):
    if tensor is None:
        return None
    storage = tensor.untyped_storage()
    pointer = tensor.data_ptr()
    storage_pointer = storage.data_ptr()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_offset": tensor.storage_offset(),
        "storage_nbytes": storage.nbytes(),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "data_ptr": pointer,
        "storage_data_ptr": storage_pointer,
        "storage_data_ptr_mod_4096": storage_pointer % 4096,
        "data_ptr_from_storage_bytes": pointer - storage_pointer,
        "data_ptr_mod_64": pointer % 64,
        "data_ptr_mod_4096": pointer % 4096,
        "is_contiguous": tensor.is_contiguous(),
    }


def _b70_metadata_tensor_desc(items):
    return {name: _b70_tensor_desc(value) for name, value in items.items()}


def _b70_append_jsonl(name: str, payload) -> None:
    _B70_GDN_DIR.mkdir(parents=True, exist_ok=True)
    with (_B70_GDN_DIR / name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\\n")


def _b70_cpu(tensor):
    return None if tensor is None else tensor.detach().to("cpu")


def _b70_state_indices(*tensors):
    values = []
    for tensor in tensors:
        if tensor is not None and tensor.numel():
            values.extend(int(value) for value in tensor.detach().to("cpu").reshape(-1))
    return sorted(set(values))


def _b70_state_rows(cache, indices):
    if not indices:
        return None
    index = torch.tensor(indices, dtype=torch.int64, device=cache.device)
    return cache.index_select(0, index).detach().to("cpu")


def _b70_neighbor_indices(indices, size):
    result = set(indices)
    for value in indices:
        if value > 0:
            result.add(value - 1)
        if value + 1 < size:
            result.add(value + 1)
    return sorted(result)


def _b70_deferred_watcher() -> None:
    trigger_path = _B70_GDN_DIR / "trigger.json"
    while True:
        try:
            if not trigger_path.exists():
                time.sleep(0.05)
                continue
            trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
            trigger_path.unlink()
            length = int(trigger["length"])
            kind = str(trigger.get("kind", "survey"))
            label = str(trigger.get("label", f"n{length}"))
            layer = int(trigger.get("layer", _B70_GDN_LAYER))
            records = _B70_GDN_DEFERRED.get(length, {})
            torch.xpu.synchronize()
            if kind == "survey":
                for record in records.values():
                    output = {
                        name: value
                        for name, value in record.items()
                        if name not in {"finite_flags", "refs"}
                    }
                    output["trigger_label"] = label
                    output["finite_flags"] = {
                        name: (None if value is None else bool(value.item()))
                        for name, value in record["finite_flags"].items()
                    }
                    _b70_append_jsonl("deferred-survey.jsonl", output)
            elif kind == "capture":
                record = records[layer]
                refs = record["refs"]
                state_indices = _b70_state_indices(
                    refs["metadata"]["non_spec_state_indices_tensor"],
                    refs["metadata"]["spec_state_indices_tensor"],
                )
                conv_indices = _b70_neighbor_indices(
                    state_indices, refs["conv_state"].shape[0]
                )
                ssm_indices = _b70_neighbor_indices(
                    state_indices, refs["ssm_state"].shape[0]
                )
                payload = {
                    "metadata": {
                        name: value
                        for name, value in record.items()
                        if name not in {"finite_flags", "refs"}
                    },
                    "trigger_label": label,
                    "state_row_indices": {
                        "selected": state_indices,
                        "conv": conv_indices,
                        "ssm": ssm_indices,
                    },
                    "finite_flags": {
                        name: (None if value is None else bool(value.item()))
                        for name, value in record["finite_flags"].items()
                    },
                    "pre": {
                        "projected_states_qkvz": _b70_cpu(refs["qkvz_pre"]),
                        "projected_states_ba": _b70_cpu(refs["ba_pre"]),
                        "core_attn_out": _b70_cpu(refs["core_pre"]),
                        "z": _b70_cpu(refs["z_pre"]),
                        "conv_weights": _b70_cpu(refs["conv_weights"]),
                        "conv_bias": _b70_cpu(refs["conv_bias"]),
                        "A_log": _b70_cpu(refs["A_log"]),
                        "dt_bias": _b70_cpu(refs["dt_bias"]),
                        "conv_state_rows": _b70_cpu(refs["conv_state_pre"]),
                        "ssm_state_rows": _b70_cpu(refs["ssm_state_pre"]),
                        "metadata": {
                            name: _b70_cpu(value)
                            for name, value in refs["metadata"].items()
                        },
                    },
                    "post": {
                        "core_attn_out": _b70_cpu(refs["core_attn_out"]),
                        "z": _b70_cpu(refs["z"]),
                        "conv_state_rows": _b70_state_rows(
                            refs["conv_state"], conv_indices
                        ),
                        "ssm_state_rows": _b70_state_rows(
                            refs["ssm_state"], ssm_indices
                        ),
                    },
                }
                global _B70_GDN_CAPTURE_ID
                name = (
                    f"deferred-{label}-n{length}-l{layer}"
                    f"-c{_B70_GDN_CAPTURE_ID:04d}.pt"
                )
                torch.save(payload, _B70_GDN_DIR / name)
                _b70_append_jsonl(
                    "captures.jsonl",
                    {
                        **payload["metadata"],
                        "capture_file": name,
                        "trigger_label": label,
                        "state_row_indices": payload["state_row_indices"],
                    },
                )
                _B70_GDN_CAPTURE_ID += 1
            else:
                raise ValueError(f"unknown deferred capture kind: {kind}")
        except Exception as error:
            _b70_append_jsonl(
                "watcher-errors.jsonl",
                {"time_ns": time.time_ns(), "error": repr(error)},
            )


def _b70_start_deferred_watcher() -> None:
    global _B70_GDN_WATCHER
    if _B70_GDN_WATCHER is not None:
        return
    _B70_GDN_DIR.mkdir(parents=True, exist_ok=True)
    _B70_GDN_WATCHER = threading.Thread(
        target=_b70_deferred_watcher,
        name="b70-gdn-capture",
        daemon=True,
    )
    _B70_GDN_WATCHER.start()
"""

CALL_ANCHOR = """    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
"""

CALL_REPLACEMENT = """    # B70_EXACT_GDN_CAPTURE_V1
    global _B70_GDN_PASS_ID, _B70_GDN_CAPTURE_ID, _B70_GDN_SURVEY
    _b70_layer = _b70_layer_index(self.prefix)
    _b70_active = (
        _B70_GDN_MODE in {"metadata", "survey", "capture", "deferred"}
        and num_prefills > 0
        and num_actual_tokens in _B70_GDN_LENGTHS
    )
    _b70_meta_tensors = {
        "has_initial_state": has_initial_state,
        "non_spec_query_start_loc": non_spec_query_start_loc,
        "non_spec_token_indx": non_spec_token_indx,
        "non_spec_state_indices_tensor": non_spec_state_indices_tensor,
        "spec_query_start_loc": spec_query_start_loc,
        "spec_token_indx": spec_token_indx,
        "spec_state_indices_tensor": spec_state_indices_tensor,
        "num_accepted_tokens": num_accepted_tokens,
    }
    _b70_tensors = {
        "core_attn_out": core_attn_out,
        "z": z,
        "projected_states_qkvz": projected_states_qkvz,
        "projected_states_ba": projected_states_ba,
        "conv_state": self.kv_cache[0],
        "ssm_state": self.kv_cache[1],
        "conv_weights": conv_weights,
        "conv_bias": self.conv1d.bias,
        "A_log": self.A_log,
        "dt_bias": self.dt_bias,
    }
    _b70_base = None
    _b70_pre_capture = None
    _b70_capture_indices = None
    if _b70_active:
        _b70_base = {
            "schema_version": 1,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
            "pass_id": _B70_GDN_PASS_ID,
            "layer_name": layer_name,
            "prefix": self.prefix,
            "layer_index": _b70_layer,
            "num_actual_tokens": num_actual_tokens,
            "num_prefills": num_prefills,
            "num_decodes": num_decodes,
            "num_spec_decodes": num_spec_decodes,
            "num_k_heads": self.num_k_heads,
            "num_v_heads": self.num_v_heads,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "tp_size": self.tp_size,
            "activation": self.activation,
            "reorder_input": not self.gqa_interleaved_layout,
            "qkvz_padding_rows": projected_states_qkvz.shape[0] - num_actual_tokens,
            "ba_padding_rows": projected_states_ba.shape[0] - num_actual_tokens,
            "core_padding_rows": core_attn_out.shape[0] - num_actual_tokens,
            "quantization_at_gdn_boundary": "none; projections are materialized model-dtype tensors",
            "tensors": _b70_metadata_tensor_desc(_b70_tensors),
            "metadata_tensors": _b70_metadata_tensor_desc(_b70_meta_tensors),
            "xpu_stream": str(torch.xpu.current_stream()),
        }
        if _B70_GDN_MODE == "metadata":
            _b70_append_jsonl("metadata.jsonl", _b70_base)
        elif _B70_GDN_MODE == "survey":
            _b70_indices = []
            for _b70_value in (
                non_spec_state_indices_tensor,
                spec_state_indices_tensor,
            ):
                if _b70_value is not None and _b70_value.numel():
                    _b70_indices.append(_b70_value.reshape(-1).to(torch.int64))
            _b70_indices_tensor = (
                torch.cat(_b70_indices) if _b70_indices else None
            )
            _b70_record = dict(_b70_base)
            _b70_record["finite_flags"] = {
                "qkvz_pre": torch.isfinite(projected_states_qkvz).all(),
                "ba_pre": torch.isfinite(projected_states_ba).all(),
                "core_pre": torch.isfinite(core_attn_out).all(),
                "z_pre": torch.isfinite(z).all(),
                "conv_state_pre": (
                    torch.isfinite(self.kv_cache[0].index_select(0, _b70_indices_tensor)).all()
                    if _b70_indices_tensor is not None
                    else None
                ),
                "ssm_state_pre": (
                    torch.isfinite(self.kv_cache[1].index_select(0, _b70_indices_tensor)).all()
                    if _b70_indices_tensor is not None
                    else None
                ),
            }
            _B70_GDN_SURVEY.append((_b70_record, _b70_indices_tensor))
        elif _B70_GDN_MODE == "deferred":
            _b70_indices = []
            for _b70_value in (
                non_spec_state_indices_tensor,
                spec_state_indices_tensor,
            ):
                if _b70_value is not None and _b70_value.numel():
                    _b70_indices.append(_b70_value.reshape(-1).to(torch.int64))
            _b70_indices_tensor = (
                torch.cat(_b70_indices) if _b70_indices else None
            )
            _b70_refs = {
                "qkvz_pre": projected_states_qkvz.clone(),
                "ba_pre": projected_states_ba.clone(),
                "core_pre": core_attn_out.clone(),
                "z_pre": z.clone(),
                "core_attn_out": core_attn_out,
                "z": z,
                "conv_state": self.kv_cache[0],
                "ssm_state": self.kv_cache[1],
                "conv_weights": conv_weights,
                "conv_bias": self.conv1d.bias,
                "A_log": self.A_log,
                "dt_bias": self.dt_bias,
                "conv_state_pre": (
                    self.kv_cache[0].index_select(0, _b70_indices_tensor)
                    if _b70_indices_tensor is not None
                    else None
                ),
                "ssm_state_pre": (
                    self.kv_cache[1].index_select(0, _b70_indices_tensor)
                    if _b70_indices_tensor is not None
                    else None
                ),
                "metadata": _b70_meta_tensors,
            }
            _b70_record = dict(_b70_base)
            _b70_record["refs"] = _b70_refs
            _b70_record["finite_flags"] = {
                "qkvz_pre": torch.isfinite(_b70_refs["qkvz_pre"]).all(),
                "ba_pre": torch.isfinite(_b70_refs["ba_pre"]).all(),
                "core_pre": torch.isfinite(_b70_refs["core_pre"]).all(),
                "z_pre": torch.isfinite(_b70_refs["z_pre"]).all(),
                "conv_state_pre": (
                    torch.isfinite(_b70_refs["conv_state_pre"]).all()
                    if _b70_refs["conv_state_pre"] is not None
                    else None
                ),
                "ssm_state_pre": (
                    torch.isfinite(_b70_refs["ssm_state_pre"]).all()
                    if _b70_refs["ssm_state_pre"] is not None
                    else None
                ),
            }
            _B70_GDN_DEFERRED.setdefault(num_actual_tokens, {})[_b70_layer] = (
                _b70_record
            )
            _b70_start_deferred_watcher()
        elif _B70_GDN_MODE == "capture" and _b70_layer == _B70_GDN_LAYER:
            _b70_selected = _b70_state_indices(
                non_spec_state_indices_tensor, spec_state_indices_tensor
            )
            _b70_capture_indices = {
                "selected": _b70_selected,
                "conv": _b70_neighbor_indices(_b70_selected, self.kv_cache[0].shape[0]),
                "ssm": _b70_neighbor_indices(_b70_selected, self.kv_cache[1].shape[0]),
            }
            _b70_pre_capture = {
                "projected_states_qkvz": _b70_cpu(projected_states_qkvz),
                "projected_states_ba": _b70_cpu(projected_states_ba),
                "core_attn_out": _b70_cpu(core_attn_out),
                "z": _b70_cpu(z),
                "conv_weights": _b70_cpu(conv_weights),
                "conv_bias": _b70_cpu(self.conv1d.bias),
                "A_log": _b70_cpu(self.A_log),
                "dt_bias": _b70_cpu(self.dt_bias),
                "conv_state_rows": _b70_state_rows(
                    self.kv_cache[0], _b70_capture_indices["conv"]
                ),
                "ssm_state_rows": _b70_state_rows(
                    self.kv_cache[1], _b70_capture_indices["ssm"]
                ),
                "metadata": {
                    name: _b70_cpu(value) for name, value in _b70_meta_tensors.items()
                },
            }

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
"""

CALL_END_ANCHOR = """        reorder_input=not self.gqa_interleaved_layout,
    )


def _gdn_attention_core_xpu_fake(
"""

CALL_END_REPLACEMENT = """        reorder_input=not self.gqa_interleaved_layout,
    )

    if _b70_active and _B70_GDN_MODE == "survey":
        _b70_record, _b70_indices_tensor = _B70_GDN_SURVEY[-1]
        _b70_record["finite_flags"].update(
            {
                "core_post": torch.isfinite(core_attn_out).all(),
                "z_post": torch.isfinite(z).all(),
                "conv_state_post": (
                    torch.isfinite(self.kv_cache[0].index_select(0, _b70_indices_tensor)).all()
                    if _b70_indices_tensor is not None
                    else None
                ),
                "ssm_state_post": (
                    torch.isfinite(self.kv_cache[1].index_select(0, _b70_indices_tensor)).all()
                    if _b70_indices_tensor is not None
                    else None
                ),
            }
        )
        if _b70_layer == _B70_GDN_LAST_LAYER:
            torch.xpu.synchronize()
            for _b70_item, _ in _B70_GDN_SURVEY:
                _b70_item["finite_flags"] = {
                    name: (None if value is None else bool(value.item()))
                    for name, value in _b70_item["finite_flags"].items()
                }
                _b70_append_jsonl("survey.jsonl", _b70_item)
            _B70_GDN_SURVEY = []
    elif _b70_active and _B70_GDN_MODE == "deferred":
        _b70_record = _B70_GDN_DEFERRED[num_actual_tokens][_b70_layer]
        _b70_refs = _b70_record["refs"]
        _b70_indices = []
        for _b70_value in (
            non_spec_state_indices_tensor,
            spec_state_indices_tensor,
        ):
            if _b70_value is not None and _b70_value.numel():
                _b70_indices.append(_b70_value.reshape(-1).to(torch.int64))
        _b70_indices_tensor = (
            torch.cat(_b70_indices) if _b70_indices else None
        )
        _b70_record["finite_flags"].update(
            {
                "core_post": torch.isfinite(core_attn_out).all(),
                "z_post": torch.isfinite(z).all(),
                "conv_state_post": (
                    torch.isfinite(
                        self.kv_cache[0].index_select(0, _b70_indices_tensor)
                    ).all()
                    if _b70_indices_tensor is not None
                    else None
                ),
                "ssm_state_post": (
                    torch.isfinite(
                        self.kv_cache[1].index_select(0, _b70_indices_tensor)
                    ).all()
                    if _b70_indices_tensor is not None
                    else None
                ),
            }
        )
    elif (
        _b70_active
        and _B70_GDN_MODE == "capture"
        and _b70_layer == _B70_GDN_LAYER
        and _b70_pre_capture is not None
        and _b70_capture_indices is not None
        and _b70_base is not None
    ):
        _b70_post_capture = {
            "core_attn_out": _b70_cpu(core_attn_out),
            "z": _b70_cpu(z),
            "conv_state_rows": _b70_state_rows(
                self.kv_cache[0], _b70_capture_indices["conv"]
            ),
            "ssm_state_rows": _b70_state_rows(
                self.kv_cache[1], _b70_capture_indices["ssm"]
            ),
        }
        _B70_GDN_DIR.mkdir(parents=True, exist_ok=True)
        _b70_name = (
            f"capture-p{_B70_GDN_PASS_ID:04d}-n{num_actual_tokens}"
            f"-l{_b70_layer}-c{_B70_GDN_CAPTURE_ID:04d}.pt"
        )
        torch.save(
            {
                "metadata": _b70_base,
                "state_row_indices": _b70_capture_indices,
                "pre": _b70_pre_capture,
                "post": _b70_post_capture,
            },
            _B70_GDN_DIR / _b70_name,
        )
        _b70_append_jsonl(
            "captures.jsonl",
            {
                **_b70_base,
                "capture_file": _b70_name,
                "state_row_indices": _b70_capture_indices,
            },
        )
        _B70_GDN_CAPTURE_ID += 1

    if _b70_active and _b70_layer == _B70_GDN_LAST_LAYER:
        _B70_GDN_PASS_ID += 1


def _gdn_attention_core_xpu_fake(
"""


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    if text.count(IMPORT_ANCHOR) != 1:
        raise RuntimeError("XPU op import anchor changed; refusing to patch")
    if text.count(CALL_ANCHOR) != 1:
        raise RuntimeError("GDN call anchor changed; refusing to patch")
    if text.count(CALL_END_ANCHOR) != 1:
        raise RuntimeError("GDN call end anchor changed; refusing to patch")
    text = text.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    text = text.replace(CALL_ANCHOR, CALL_REPLACEMENT, 1)
    return text.replace(CALL_END_ANCHOR, CALL_END_REPLACEMENT, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    path = Path(next(iter(spec.submodule_search_locations))) / "_xpu_ops.py"
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
