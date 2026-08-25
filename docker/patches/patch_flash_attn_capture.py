#!/usr/bin/env python3
"""Install opt-in deferred capture at the real full-attention boundary.

The capture is diagnostic only.  It preserves the live FlashAttention dispatch
and records the active Q/K/V tensors, the exact referenced paged-cache block,
the metadata/scales, and the active output.  ``deferred`` mode uses device
clones and a trigger file for a low-observer-effect survey.  ``sync`` mode
captures one selected layer synchronously; its raw API oracle must be rerun to
prove that the synchronization did not hide the failure.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_EXACT_FLASH_ATTN_CAPTURE_V1"

IMPORT_ANCHOR = """from typing import TYPE_CHECKING, Any, cast

import torch
"""

IMPORT_REPLACEMENT = """from typing import TYPE_CHECKING, Any, cast

# B70_EXACT_FLASH_ATTN_CAPTURE_V1: exact real-graph full-attention capture.
import json
import os
import re
import threading
import time
from pathlib import Path

import torch

_B70_ATTN_MODE = os.getenv("B70_ATTN_CAPTURE_MODE", "").strip().lower()
_B70_ATTN_DIR = Path(os.getenv("B70_ATTN_CAPTURE_DIR", "/b70-attn-captures"))
_B70_ATTN_LENGTHS = {
    int(value)
    for value in os.getenv("B70_ATTN_CAPTURE_LENGTHS", "4,5,6").split(",")
    if value.strip()
}
_B70_ATTN_LAYER = int(os.getenv("B70_ATTN_CAPTURE_LAYER", "3"))
_B70_ATTN_DEFERRED = {}
_B70_ATTN_WATCHER = None
_B70_ATTN_CAPTURE_ID = 0


def _b70_attn_layer_index(value: str) -> int:
    match = re.search(r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)", value)
    return int(match.group(1)) if match else -1


def _b70_attn_tensor_desc(tensor):
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
        "data_ptr_from_storage_bytes": pointer - storage_pointer,
        "data_ptr_mod_64": pointer % 64,
        "data_ptr_mod_4096": pointer % 4096,
        "is_contiguous": tensor.is_contiguous(),
    }


def _b70_attn_cpu(tensor):
    return None if tensor is None else tensor.detach().to("cpu")


def _b70_attn_append_jsonl(name: str, payload) -> None:
    _B70_ATTN_DIR.mkdir(parents=True, exist_ok=True)
    with (_B70_ATTN_DIR / name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\\n")


def _b70_attn_dump_record(record, label: str) -> str:
    global _B70_ATTN_CAPTURE_ID
    refs = record["refs"]
    finite_flags = refs.get("finite_flags")
    if finite_flags:
        finite_flags = {
            name: bool(value.item()) for name, value in finite_flags.items()
        }
    else:
        finite_flags = {
            name: bool(torch.isfinite(refs[name]).all().item())
            for name in (
                "query_active",
                "key_active",
                "value_active",
                "kv_cache_blocks",
                "output_active",
            )
        }
    payload = {
        "metadata": {
            name: value for name, value in record.items() if name != "refs"
        },
        "trigger_label": label,
        "finite_flags": finite_flags,
        "inputs": {
            "query_active": _b70_attn_cpu(refs["query_active"]),
            "key_active": _b70_attn_cpu(refs["key_active"]),
            "value_active": _b70_attn_cpu(refs["value_active"]),
            "query_padding": _b70_attn_cpu(refs["query_padding"]),
            "key_padding": _b70_attn_cpu(refs["key_padding"]),
            "value_padding": _b70_attn_cpu(refs["value_padding"]),
            "kv_cache_blocks": _b70_attn_cpu(refs["kv_cache_blocks"]),
            "q_scale": _b70_attn_cpu(refs["q_scale"]),
            "k_scale": _b70_attn_cpu(refs["k_scale"]),
            "v_scale": _b70_attn_cpu(refs["v_scale"]),
        },
        "attention_metadata": {
            name: _b70_attn_cpu(value)
            for name, value in refs["attention_metadata"].items()
        },
        "output": _b70_attn_cpu(refs["output_active"]),
    }
    name = (
        f"flash-{label}-n{record['num_actual_tokens']}"
        f"-l{record['layer_index']}-c{_B70_ATTN_CAPTURE_ID:04d}.pt"
    )
    torch.save(payload, _B70_ATTN_DIR / name)
    _b70_attn_append_jsonl(
        "flash-captures.jsonl",
        {
            **payload["metadata"],
            "capture_file": name,
            "trigger_label": label,
            "finite_flags": payload["finite_flags"],
        },
    )
    _B70_ATTN_CAPTURE_ID += 1
    return name


def _b70_attn_watcher() -> None:
    trigger_path = _B70_ATTN_DIR / "flash-trigger.json"
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
            layer = int(trigger.get("layer", _B70_ATTN_LAYER))
            records = _B70_ATTN_DEFERRED.get(length, {})
            torch.xpu.synchronize()
            if kind == "survey":
                for record in records.values():
                    refs = record["refs"]
                    output = {
                        name: value for name, value in record.items() if name != "refs"
                    }
                    output["trigger_label"] = label
                    output["finite_flags"] = {
                        name: bool(value.item())
                        for name, value in refs["finite_flags"].items()
                    }
                    _b70_attn_append_jsonl("flash-survey.jsonl", output)
            elif kind == "capture":
                _b70_attn_dump_record(records[layer], label)
            else:
                raise ValueError(f"unknown full-attention capture kind: {kind}")
        except Exception as error:
            _b70_attn_append_jsonl(
                "flash-watcher-errors.jsonl",
                {"time_ns": time.time_ns(), "error": repr(error)},
            )


def _b70_attn_start_watcher() -> None:
    global _B70_ATTN_WATCHER
    if _B70_ATTN_WATCHER is not None:
        return
    _B70_ATTN_DIR.mkdir(parents=True, exist_ok=True)
    _B70_ATTN_WATCHER = threading.Thread(
        target=_b70_attn_watcher,
        name="b70-flash-attn-capture",
        daemon=True,
    )
    _B70_ATTN_WATCHER.start()
"""

CALL_ANCHOR = """    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )
"""

CALL_REPLACEMENT = """    # B70_EXACT_FLASH_ATTN_CAPTURE_V1
    global _B70_ATTN_CAPTURE_ID
    _b70_attn_layer = _b70_attn_layer_index(str(layer_name))
    _b70_attn_n = (
        int(attn_metadata.num_actual_tokens)
        if attn_metadata is not None
        else -1
    )
    _b70_attn_active = (
        _B70_ATTN_MODE == "deferred"
        and _b70_attn_n in _B70_ATTN_LENGTHS
        and _b70_attn_layer >= 0
    )
    _b70_attn_sync = (
        _B70_ATTN_MODE == "sync"
        and _b70_attn_n in _B70_ATTN_LENGTHS
        and _b70_attn_layer == _B70_ATTN_LAYER
        and not torch.xpu.is_current_stream_capturing()
    )
    _b70_attn_persistent = (
        _B70_ATTN_MODE == "persistent"
        and _b70_attn_n in _B70_ATTN_LENGTHS
        and _b70_attn_layer == _B70_ATTN_LAYER
    )
    _b70_attn_refs = None
    _b70_attn_sync_pre = None
    _b70_attn_persistent_refs = None
    if _b70_attn_persistent:
        _b70_attn_record = _B70_ATTN_DEFERRED.get(_b70_attn_n, {}).get(
            _b70_attn_layer
        )
        if (
            _b70_attn_record is None
            and not torch.xpu.is_current_stream_capturing()
        ):
            _b70_attn_block_ids = (
                attn_metadata.block_table[:1, :1].reshape(-1).to(torch.int64)
            )
            _b70_attn_meta_tensors = {
                "query_start_loc": attn_metadata.query_start_loc,
                "seq_lens": attn_metadata.seq_lens,
                "block_table": attn_metadata.block_table,
                "slot_mapping": attn_metadata.slot_mapping,
                "scheduler_metadata": attn_metadata.scheduler_metadata,
            }
            if isinstance(attn_metadata.causal, torch.Tensor):
                _b70_attn_meta_tensors["causal_tensor"] = attn_metadata.causal
            _b70_attn_persistent_refs = {
                "query_active": torch.empty_like(query[:_b70_attn_n]),
                "key_active": torch.empty_like(key[:_b70_attn_n]),
                "value_active": torch.empty_like(value[:_b70_attn_n]),
                "query_padding": torch.empty_like(query[_b70_attn_n:]),
                "key_padding": torch.empty_like(key[_b70_attn_n:]),
                "value_padding": torch.empty_like(value[_b70_attn_n:]),
                "kv_cache_blocks": torch.empty_like(
                    kv_cache.index_select(0, _b70_attn_block_ids)
                ),
                "q_scale": (
                    None if self._q_scale is None else torch.empty_like(self._q_scale)
                ),
                "k_scale": (
                    None if self._k_scale is None else torch.empty_like(self._k_scale)
                ),
                "v_scale": (
                    None if self._v_scale is None else torch.empty_like(self._v_scale)
                ),
                "attention_metadata": {
                    name: (None if value is None else torch.empty_like(value))
                    for name, value in _b70_attn_meta_tensors.items()
                },
                "output_active": torch.empty_like(output[:_b70_attn_n]),
                "finite_flags": {},
            }
            _b70_attn_record = {
                "schema_version": 2,
                "capture_strategy": "persistent-buffer-copy",
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
                "layer_name": str(layer_name),
                "layer_index": _b70_attn_layer,
                "num_actual_tokens": _b70_attn_n,
                "max_query_len": attn_metadata.max_query_len,
                "max_seq_len": attn_metadata.max_seq_len,
                "causal": (
                    bool(attn_metadata.causal)
                    if not isinstance(attn_metadata.causal, torch.Tensor)
                    else "tensor"
                ),
                "use_cascade": bool(attn_metadata.use_cascade),
                "common_prefix_len": attn_metadata.common_prefix_len,
                "max_num_splits": attn_metadata.max_num_splits,
                "num_decode_reqs": attn_metadata.num_decode_reqs,
                "num_prefill_reqs": attn_metadata.num_prefill_reqs,
                "num_decode_tokens": attn_metadata.num_decode_tokens,
                "num_prefill_tokens": attn_metadata.num_prefill_tokens,
                "impl_class": type(self.impl).__name__,
                "num_heads": self.impl.num_heads,
                "num_kv_heads": self.impl.num_kv_heads,
                "head_size": self.impl.head_size,
                "softmax_scale": self.impl.scale,
                "sliding_window": list(self.impl.sliding_window),
                "kv_cache_dtype": self.impl.kv_cache_dtype,
                "flash_attn_version": self.impl.vllm_flash_attn_version,
                "tensors": {
                    "query": _b70_attn_tensor_desc(query),
                    "key": _b70_attn_tensor_desc(key),
                    "value": _b70_attn_tensor_desc(value),
                    "output": _b70_attn_tensor_desc(output),
                    "kv_cache": _b70_attn_tensor_desc(kv_cache),
                    "q_scale": _b70_attn_tensor_desc(self._q_scale),
                    "k_scale": _b70_attn_tensor_desc(self._k_scale),
                    "v_scale": _b70_attn_tensor_desc(self._v_scale),
                },
                "metadata_tensors": {
                    name: _b70_attn_tensor_desc(value)
                    for name, value in _b70_attn_meta_tensors.items()
                },
                "refs": _b70_attn_persistent_refs,
            }
            _B70_ATTN_DEFERRED.setdefault(_b70_attn_n, {})[
                _b70_attn_layer
            ] = _b70_attn_record
            _b70_attn_start_watcher()
        elif _b70_attn_record is not None:
            _b70_attn_persistent_refs = _b70_attn_record["refs"]

        if _b70_attn_persistent_refs is not None:
            _b70_attn_block_ids = (
                attn_metadata.block_table[:1, :1].reshape(-1).to(torch.int64)
            )
            _b70_attn_persistent_refs["query_active"].copy_(
                query[:_b70_attn_n]
            )
            _b70_attn_persistent_refs["key_active"].copy_(key[:_b70_attn_n])
            _b70_attn_persistent_refs["value_active"].copy_(
                value[:_b70_attn_n]
            )
            _b70_attn_persistent_refs["query_padding"].copy_(
                query[_b70_attn_n:]
            )
            _b70_attn_persistent_refs["key_padding"].copy_(
                key[_b70_attn_n:]
            )
            _b70_attn_persistent_refs["value_padding"].copy_(
                value[_b70_attn_n:]
            )
            _b70_attn_persistent_refs["kv_cache_blocks"].copy_(
                kv_cache.index_select(0, _b70_attn_block_ids)
            )
            for _b70_attn_name, _b70_attn_source in (
                ("q_scale", self._q_scale),
                ("k_scale", self._k_scale),
                ("v_scale", self._v_scale),
            ):
                if _b70_attn_source is not None:
                    _b70_attn_persistent_refs[_b70_attn_name].copy_(
                        _b70_attn_source
                    )
            for _b70_attn_name, _b70_attn_source in (
                ("query_start_loc", attn_metadata.query_start_loc),
                ("seq_lens", attn_metadata.seq_lens),
                ("block_table", attn_metadata.block_table),
                ("slot_mapping", attn_metadata.slot_mapping),
                ("scheduler_metadata", attn_metadata.scheduler_metadata),
            ):
                if _b70_attn_source is not None:
                    _b70_attn_persistent_refs["attention_metadata"][
                        _b70_attn_name
                    ].copy_(_b70_attn_source)
            if isinstance(attn_metadata.causal, torch.Tensor):
                _b70_attn_persistent_refs["attention_metadata"][
                    "causal_tensor"
                ].copy_(attn_metadata.causal)
    elif _b70_attn_active:
        # The failing oracle uses one request shorter than the 64-token XPU
        # cache page.  Capture exactly its referenced page without copying the
        # unrelated multi-gigabyte KV cache.
        _b70_attn_block_ids = attn_metadata.block_table[:1, :1].reshape(-1).to(
            torch.int64
        )
        _b70_attn_meta_tensors = {
            "query_start_loc": attn_metadata.query_start_loc,
            "seq_lens": attn_metadata.seq_lens,
            "block_table": attn_metadata.block_table,
            "slot_mapping": attn_metadata.slot_mapping,
            "scheduler_metadata": attn_metadata.scheduler_metadata,
        }
        if isinstance(attn_metadata.causal, torch.Tensor):
            _b70_attn_meta_tensors["causal_tensor"] = attn_metadata.causal
        _b70_attn_refs = {
            "query_active": query[:_b70_attn_n].clone(),
            "key_active": key[:_b70_attn_n].clone(),
            "value_active": value[:_b70_attn_n].clone(),
            "query_padding": query[_b70_attn_n:].clone(),
            "key_padding": key[_b70_attn_n:].clone(),
            "value_padding": value[_b70_attn_n:].clone(),
            "kv_cache_blocks": kv_cache.index_select(0, _b70_attn_block_ids).clone(),
            "q_scale": (
                None if self._q_scale is None else self._q_scale.clone()
            ),
            "k_scale": (
                None if self._k_scale is None else self._k_scale.clone()
            ),
            "v_scale": (
                None if self._v_scale is None else self._v_scale.clone()
            ),
            "attention_metadata": {
                name: (None if value is None else value.clone())
                for name, value in _b70_attn_meta_tensors.items()
            },
        }
    elif _b70_attn_sync:
        # Synchronize only at the selected operation so that the checkpoint is
        # a coherent snapshot rather than a graph-pool buffer whose lifetime
        # extends past its compiler-visible use.  The API oracle validates the
        # observer effect after every synchronized capture run.
        torch.xpu.synchronize()
        _b70_attn_block_ids = attn_metadata.block_table[:1, :1].reshape(-1).to(
            torch.int64
        )
        _b70_attn_sync_pre = {
            "query_active": _b70_attn_cpu(query[:_b70_attn_n]),
            "key_active": _b70_attn_cpu(key[:_b70_attn_n]),
            "value_active": _b70_attn_cpu(value[:_b70_attn_n]),
            "query_padding": _b70_attn_cpu(query[_b70_attn_n:]),
            "key_padding": _b70_attn_cpu(key[_b70_attn_n:]),
            "value_padding": _b70_attn_cpu(value[_b70_attn_n:]),
            "kv_cache_blocks": _b70_attn_cpu(
                kv_cache.index_select(0, _b70_attn_block_ids)
            ),
            "q_scale": _b70_attn_cpu(self._q_scale),
            "k_scale": _b70_attn_cpu(self._k_scale),
            "v_scale": _b70_attn_cpu(self._v_scale),
            "attention_metadata": {
                "query_start_loc": _b70_attn_cpu(attn_metadata.query_start_loc),
                "seq_lens": _b70_attn_cpu(attn_metadata.seq_lens),
                "block_table": _b70_attn_cpu(attn_metadata.block_table),
                "slot_mapping": _b70_attn_cpu(attn_metadata.slot_mapping),
                "scheduler_metadata": _b70_attn_cpu(
                    attn_metadata.scheduler_metadata
                ),
                "causal_tensor": (
                    _b70_attn_cpu(attn_metadata.causal)
                    if isinstance(attn_metadata.causal, torch.Tensor)
                    else None
                ),
            },
        }

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )

    if _b70_attn_persistent_refs is not None:
        _b70_attn_persistent_refs["output_active"].copy_(
            output[:_b70_attn_n]
        )

    if _b70_attn_refs is not None:
        _b70_attn_refs["output_active"] = output[:_b70_attn_n].clone()
        _b70_attn_refs["finite_flags"] = {
            "query_active": torch.isfinite(_b70_attn_refs["query_active"]).all(),
            "key_active": torch.isfinite(_b70_attn_refs["key_active"]).all(),
            "value_active": torch.isfinite(_b70_attn_refs["value_active"]).all(),
            "kv_cache_blocks": torch.isfinite(
                _b70_attn_refs["kv_cache_blocks"]
            ).all(),
            "output_active": torch.isfinite(_b70_attn_refs["output_active"]).all(),
        }
        _B70_ATTN_DEFERRED.setdefault(_b70_attn_n, {})[_b70_attn_layer] = {
            "schema_version": 1,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
            "layer_name": str(layer_name),
            "layer_index": _b70_attn_layer,
            "num_actual_tokens": _b70_attn_n,
            "max_query_len": attn_metadata.max_query_len,
            "max_seq_len": attn_metadata.max_seq_len,
            "causal": (
                bool(attn_metadata.causal)
                if not isinstance(attn_metadata.causal, torch.Tensor)
                else "tensor"
            ),
            "use_cascade": bool(attn_metadata.use_cascade),
            "common_prefix_len": attn_metadata.common_prefix_len,
            "max_num_splits": attn_metadata.max_num_splits,
            "num_decode_reqs": attn_metadata.num_decode_reqs,
            "num_prefill_reqs": attn_metadata.num_prefill_reqs,
            "num_decode_tokens": attn_metadata.num_decode_tokens,
            "num_prefill_tokens": attn_metadata.num_prefill_tokens,
            "impl_class": type(self.impl).__name__,
            "num_heads": self.impl.num_heads,
            "num_kv_heads": self.impl.num_kv_heads,
            "head_size": self.impl.head_size,
            "softmax_scale": self.impl.scale,
            "sliding_window": list(self.impl.sliding_window),
            "kv_cache_dtype": self.impl.kv_cache_dtype,
            "flash_attn_version": self.impl.vllm_flash_attn_version,
            "tensors": {
                "query": _b70_attn_tensor_desc(query),
                "key": _b70_attn_tensor_desc(key),
                "value": _b70_attn_tensor_desc(value),
                "output": _b70_attn_tensor_desc(output),
                "kv_cache": _b70_attn_tensor_desc(kv_cache),
                "q_scale": _b70_attn_tensor_desc(self._q_scale),
                "k_scale": _b70_attn_tensor_desc(self._k_scale),
                "v_scale": _b70_attn_tensor_desc(self._v_scale),
            },
            "metadata_tensors": {
                name: _b70_attn_tensor_desc(value)
                for name, value in _b70_attn_meta_tensors.items()
            },
            "refs": _b70_attn_refs,
        }
        _b70_attn_start_watcher()
    elif _b70_attn_sync_pre is not None:
        torch.xpu.synchronize()
        _B70_ATTN_DIR.mkdir(parents=True, exist_ok=True)
        _b70_attn_sync_output = _b70_attn_cpu(output[:_b70_attn_n])
        _b70_attn_sync_flags = {
            name: bool(torch.isfinite(value).all().item())
            for name, value in _b70_attn_sync_pre.items()
            if isinstance(value, torch.Tensor)
        }
        _b70_attn_sync_flags["output_active"] = bool(
            torch.isfinite(_b70_attn_sync_output).all().item()
        )
        _b70_attn_name = (
            f"flash-sync-n{_b70_attn_n}-l{_b70_attn_layer}"
            f"-c{_B70_ATTN_CAPTURE_ID:04d}.pt"
        )
        torch.save(
            {
                "metadata": {
                    "schema_version": 1,
                    "pid": os.getpid(),
                    "time_ns": time.time_ns(),
                    "layer_name": str(layer_name),
                    "layer_index": _b70_attn_layer,
                    "num_actual_tokens": _b70_attn_n,
                    "max_query_len": attn_metadata.max_query_len,
                    "max_seq_len": attn_metadata.max_seq_len,
                    "causal": (
                        bool(attn_metadata.causal)
                        if not isinstance(attn_metadata.causal, torch.Tensor)
                        else "tensor"
                    ),
                    "use_cascade": bool(attn_metadata.use_cascade),
                    "common_prefix_len": attn_metadata.common_prefix_len,
                    "max_num_splits": attn_metadata.max_num_splits,
                    "num_decode_reqs": attn_metadata.num_decode_reqs,
                    "num_prefill_reqs": attn_metadata.num_prefill_reqs,
                    "num_decode_tokens": attn_metadata.num_decode_tokens,
                    "num_prefill_tokens": attn_metadata.num_prefill_tokens,
                    "impl_class": type(self.impl).__name__,
                    "num_heads": self.impl.num_heads,
                    "num_kv_heads": self.impl.num_kv_heads,
                    "head_size": self.impl.head_size,
                    "softmax_scale": self.impl.scale,
                    "sliding_window": list(self.impl.sliding_window),
                    "kv_cache_dtype": self.impl.kv_cache_dtype,
                    "flash_attn_version": self.impl.vllm_flash_attn_version,
                    "tensors": {
                        "query": _b70_attn_tensor_desc(query),
                        "key": _b70_attn_tensor_desc(key),
                        "value": _b70_attn_tensor_desc(value),
                        "output": _b70_attn_tensor_desc(output),
                        "kv_cache": _b70_attn_tensor_desc(kv_cache),
                    },
                },
                "finite_flags": _b70_attn_sync_flags,
                "inputs": {
                    name: value
                    for name, value in _b70_attn_sync_pre.items()
                    if name != "attention_metadata"
                },
                "attention_metadata": _b70_attn_sync_pre[
                    "attention_metadata"
                ],
                "output": _b70_attn_sync_output,
            },
            _B70_ATTN_DIR / _b70_attn_name,
        )
        _b70_attn_append_jsonl(
            "flash-sync-captures.jsonl",
            {
                "capture_file": _b70_attn_name,
                "layer_index": _b70_attn_layer,
                "num_actual_tokens": _b70_attn_n,
                "finite_flags": _b70_attn_sync_flags,
            },
        )
        _B70_ATTN_CAPTURE_ID += 1
"""


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    if text.count(IMPORT_ANCHOR) != 1:
        raise RuntimeError("attention import anchor changed; refusing to patch")
    if text.count(CALL_ANCHOR) != 1:
        raise RuntimeError("attention call anchor changed; refusing to patch")
    text = text.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    return text.replace(CALL_ANCHOR, CALL_REPLACEMENT, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "model_executor/layers/attention/attention.py"
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
