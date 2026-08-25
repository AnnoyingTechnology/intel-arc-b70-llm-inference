#!/usr/bin/env python3
"""Capture persistent Qwen3.5 layer-3/4 buffers across XPU graph replay.

The buffers are registered on the model before graph recording.  ``copy_``
mutations are then captured in the real graph and update the same persistent
storage on replay.  A watcher dumps only after ``stage-trigger.json`` appears.
This avoids both graph-pool clone reuse and mid-graph synchronization.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_MODEL_STAGE_CAPTURE_V1"

IMPORT_ANCHOR = """from collections.abc import Iterable
from itertools import islice

import torch
"""

IMPORT_REPLACEMENT = """from collections.abc import Iterable
from itertools import islice

# B70_MODEL_STAGE_CAPTURE_V1: persistent buffers mutated by XPU graph replay.
import json
import os
import threading
import time
from pathlib import Path

import torch

_B70_STAGE_MODE = os.getenv("B70_STAGE_CAPTURE_MODE", "").strip().lower()
_B70_STAGE_DIR = Path(os.getenv("B70_STAGE_CAPTURE_DIR", "/b70-stage-captures"))
_B70_STAGE_MAX_ROWS = int(os.getenv("B70_STAGE_CAPTURE_MAX_ROWS", "8"))
_B70_STAGE_REFS = {"attention": {}, "decoder": {}}
_B70_STAGE_WATCHER = None
_B70_STAGE_CAPTURE_ID = 0


def _b70_stage_register_buffer(module, name, shape, dtype):
    module.register_buffer(
        name,
        torch.full(shape, float("nan"), dtype=dtype),
        persistent=False,
    )
    return getattr(module, name)


def _b70_stage_register_attention(module, prefix, config, model_config):
    module._b70_stage_refs = None
    layer = extract_layer_index(prefix)
    if _B70_STAGE_MODE != "persistent" or layer != 3:
        return
    dtype = model_config.dtype
    q_size = module.num_heads * module.head_dim
    kv_size = module.num_kv_heads * module.head_dim
    qkv_size = q_size * (2 if module.attn_output_gate else 1) + 2 * kv_size
    refs = {
        "hidden_input": _b70_stage_register_buffer(
            module, "_b70_hidden_input", (_B70_STAGE_MAX_ROWS, module.hidden_size), dtype
        ),
        "qkv": _b70_stage_register_buffer(
            module, "_b70_qkv", (_B70_STAGE_MAX_ROWS, qkv_size), dtype
        ),
        "gate": _b70_stage_register_buffer(
            module, "_b70_gate", (_B70_STAGE_MAX_ROWS, q_size), dtype
        ),
        "attention_raw": _b70_stage_register_buffer(
            module, "_b70_attention_raw", (_B70_STAGE_MAX_ROWS, q_size), dtype
        ),
        "attention_gated": _b70_stage_register_buffer(
            module, "_b70_attention_gated", (_B70_STAGE_MAX_ROWS, q_size), dtype
        ),
        "output_projection": _b70_stage_register_buffer(
            module,
            "_b70_output_projection",
            (_B70_STAGE_MAX_ROWS, module.hidden_size),
            dtype,
        ),
    }
    module._b70_stage_refs = refs
    _B70_STAGE_REFS["attention"][layer] = {
        "prefix": prefix,
        "refs": refs,
    }
    _b70_stage_start_watcher()


def _b70_stage_register_decoder(module, prefix, config, model_config):
    module._b70_stage_refs = None
    layer = module.layer_idx
    if (
        _B70_STAGE_MODE != "persistent"
        or layer not in {0, 1, 2, 3, 4}
        or not prefix.startswith("language_model.")
    ):
        return
    dtype = model_config.dtype
    shape = (_B70_STAGE_MAX_ROWS, config.hidden_size)
    refs = {
        name: _b70_stage_register_buffer(module, f"_b70_{name}", shape, dtype)
        for name in (
            "entry_hidden",
            "entry_residual",
            "input_norm",
            "attention_output",
            "post_attention_norm",
            "post_attention_residual",
            "mlp_output",
        )
    }
    module._b70_stage_refs = refs
    _B70_STAGE_REFS["decoder"][layer] = {
        "prefix": prefix,
        "layer_type": module.layer_type,
        "refs": refs,
    }
    _b70_stage_start_watcher()


def _b70_stage_copy(ref, tensor):
    # Keep the mutation shape static so Dynamo can compile both the 8192-token
    # profiling graph and the small replay buckets.  Slicing caps the symbolic
    # row count; padding makes every captured copy exactly MAX_ROWS rows.
    value = tensor[:_B70_STAGE_MAX_ROWS]
    value = torch.nn.functional.pad(
        value,
        (0, 0, 0, _B70_STAGE_MAX_ROWS - value.shape[0]),
    )
    ref.copy_(value)


def _b70_stage_append_jsonl(name, payload):
    _B70_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    with (_B70_STAGE_DIR / name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\\n")


def _b70_stage_watcher():
    global _B70_STAGE_CAPTURE_ID
    trigger_path = _B70_STAGE_DIR / "stage-trigger.json"
    while True:
        try:
            if not trigger_path.exists():
                time.sleep(0.05)
                continue
            trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
            trigger_path.unlink()
            length = int(trigger["length"])
            label = str(trigger.get("label", f"n{length}"))
            torch.xpu.synchronize()
            payload = {
                "schema_version": 1,
                "capture_strategy": "registered-persistent-buffer-copy",
                "trigger_label": label,
                "length": length,
                "attention": {},
                "decoder": {},
            }
            flags = {}
            for section in ("attention", "decoder"):
                for layer, record in _B70_STAGE_REFS[section].items():
                    key = f"{section}.layer{layer}"
                    payload[section][layer] = {
                        name: tensor[:length].detach().to("cpu")
                        for name, tensor in record["refs"].items()
                    }
                    payload[section][layer]["metadata"] = {
                        name: value for name, value in record.items() if name != "refs"
                    }
                    flags[key] = {
                        name: bool(torch.isfinite(tensor[:length]).all().item())
                        for name, tensor in record["refs"].items()
                    }
            payload["finite_flags"] = flags
            name = (
                f"model-stages-{label}-n{length}"
                f"-c{_B70_STAGE_CAPTURE_ID:04d}.pt"
            )
            torch.save(payload, _B70_STAGE_DIR / name)
            _b70_stage_append_jsonl(
                "model-stage-captures.jsonl",
                {
                    "capture_file": name,
                    "trigger_label": label,
                    "length": length,
                    "finite_flags": flags,
                },
            )
            _B70_STAGE_CAPTURE_ID += 1
        except Exception as error:
            _b70_stage_append_jsonl(
                "model-stage-errors.jsonl",
                {"time_ns": time.time_ns(), "error": repr(error)},
            )


def _b70_stage_start_watcher():
    global _B70_STAGE_WATCHER
    if _B70_STAGE_WATCHER is not None:
        return
    _B70_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    _B70_STAGE_WATCHER = threading.Thread(
        target=_b70_stage_watcher,
        name="b70-model-stage-capture",
        daemon=True,
    )
    _B70_STAGE_WATCHER.start()
"""

ATTN_INIT_ANCHOR = """        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Fuse the gated split + QK-RMSNorm + (partial) NeoX RoPE + gate copy.
"""

ATTN_INIT_REPLACEMENT = """        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        _b70_stage_register_attention(self, prefix, config, model_config)

        # Fuse the gated split + QK-RMSNorm + (partial) NeoX RoPE + gate copy.
"""

ATTN_FORWARD_ANCHOR = """        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        attn_output = self.attn(q, k, v)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output
"""

ATTN_FORWARD_REPLACEMENT = """        if self._b70_stage_refs is not None:
            _b70_stage_copy(self._b70_stage_refs["hidden_input"], hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        if self._b70_stage_refs is not None:
            _b70_stage_copy(self._b70_stage_refs["qkv"], qkv)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        if self._b70_stage_refs is not None and gate is not None:
            _b70_stage_copy(self._b70_stage_refs["gate"], gate)
        attn_output = self.attn(q, k, v)
        if self._b70_stage_refs is not None:
            _b70_stage_copy(self._b70_stage_refs["attention_raw"], attn_output)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        if self._b70_stage_refs is not None:
            _b70_stage_copy(self._b70_stage_refs["attention_gated"], attn_output)
        output, _ = self.o_proj(attn_output)
        if self._b70_stage_refs is not None:
            _b70_stage_copy(self._b70_stage_refs["output_projection"], output)
        return output
"""

DECODER_INIT_ANCHOR = """        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
"""

DECODER_INIT_REPLACEMENT = """        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        _b70_stage_register_decoder(self, prefix, config, model_config)

        self.layer_scale = getattr(config, "layer_scale", False)
"""

DECODER_ENTRY_ANCHOR = """        full_num_tokens = positions.shape[-1]

        if residual is None:
"""

DECODER_ENTRY_REPLACEMENT = """        full_num_tokens = positions.shape[-1]

        if self._b70_stage_refs is not None:
            _b70_stage_copy(self._b70_stage_refs["entry_hidden"], hidden_states)
            if residual is not None:
                _b70_stage_copy(self._b70_stage_refs["entry_residual"], residual)

        if residual is None:
"""

DECODER_NORM_ANCHOR = """        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.use_attn_reduce_scatter_for_moe:
"""

DECODER_NORM_REPLACEMENT = """        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self._b70_stage_refs is not None:
            _b70_stage_copy(self._b70_stage_refs["input_norm"], hidden_states)

        if self.use_attn_reduce_scatter_for_moe:
"""

DECODER_ATTN_ANCHOR = """        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
"""

DECODER_ATTN_REPLACEMENT = """        else:
            raise ValueError("Invalid layer_type")

        if self._b70_stage_refs is not None:
            _b70_stage_copy(self._b70_stage_refs["attention_output"], hidden_states)

        if self.layer_scale:
"""

DECODER_POST_NORM_ANCHOR = """        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.use_attn_reduce_scatter_for_moe:
"""

DECODER_POST_NORM_REPLACEMENT = """        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self._b70_stage_refs is not None:
            _b70_stage_copy(
                self._b70_stage_refs["post_attention_norm"], hidden_states
            )
            _b70_stage_copy(
                self._b70_stage_refs["post_attention_residual"], residual
            )
        if self.use_attn_reduce_scatter_for_moe:
"""

DECODER_MLP_ANCHOR = """        else:
            hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
"""

DECODER_MLP_REPLACEMENT = """        else:
            hidden_states = self.mlp(hidden_states)

        if self._b70_stage_refs is not None:
            _b70_stage_copy(self._b70_stage_refs["mlp_output"], hidden_states)

        if self.layer_scale:
"""


REPLACEMENTS = (
    (IMPORT_ANCHOR, IMPORT_REPLACEMENT, "import"),
    (ATTN_INIT_ANCHOR, ATTN_INIT_REPLACEMENT, "attention init"),
    (ATTN_FORWARD_ANCHOR, ATTN_FORWARD_REPLACEMENT, "attention forward"),
    (DECODER_INIT_ANCHOR, DECODER_INIT_REPLACEMENT, "decoder init"),
    (DECODER_ENTRY_ANCHOR, DECODER_ENTRY_REPLACEMENT, "decoder entry"),
    (DECODER_NORM_ANCHOR, DECODER_NORM_REPLACEMENT, "decoder input norm"),
    (DECODER_ATTN_ANCHOR, DECODER_ATTN_REPLACEMENT, "decoder attention"),
    (DECODER_POST_NORM_ANCHOR, DECODER_POST_NORM_REPLACEMENT, "decoder post norm"),
    (DECODER_MLP_ANCHOR, DECODER_MLP_REPLACEMENT, "decoder mlp"),
)

Q35_IMPORT_ANCHOR = """    QwenNextMixtureOfExperts,
    _is_shared_expert_fse_compatible,
)"""

Q35_IMPORT_REPLACEMENT = """    QwenNextMixtureOfExperts,
    _b70_stage_register_decoder,
    _is_shared_expert_fse_compatible,
)"""

Q35_INIT_ANCHOR = """        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
"""

Q35_INIT_REPLACEMENT = """        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        _b70_stage_register_decoder(self, prefix, config, model_config)

        self.layer_scale = getattr(config, "layer_scale", False)
"""


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    for anchor, replacement, name in REPLACEMENTS:
        if text.count(anchor) != 1:
            raise RuntimeError(f"{name} anchor changed; refusing to patch")
        text = text.replace(anchor, replacement, 1)
    return text


def patch_qwen35_text(text: str) -> str:
    marker = "_b70_stage_register_decoder(self, prefix, config, model_config)"
    if marker in text:
        return text
    for anchor, replacement, name in (
        (Q35_IMPORT_ANCHOR, Q35_IMPORT_REPLACEMENT, "qwen3.5 import"),
        (Q35_INIT_ANCHOR, Q35_INIT_REPLACEMENT, "qwen3.5 decoder init"),
    ):
        if text.count(anchor) != 1:
            raise RuntimeError(f"{name} anchor changed; refusing to patch")
        text = text.replace(anchor, replacement, 1)
    return text


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    base = Path(next(iter(spec.submodule_search_locations)))
    path = (
        base
        / "model_executor/models/qwen3_next.py"
    )
    original = path.read_text()
    patched = patch_text(original)
    if patched == original:
        print(f"already patched {path}")
        return
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"patched {path}")

    qwen35_path = base / "model_executor/models/qwen3_5.py"
    qwen35_original = qwen35_path.read_text()
    qwen35_patched = patch_qwen35_text(qwen35_original)
    if qwen35_patched == qwen35_original:
        print(f"already patched {qwen35_path}")
        return
    compile(qwen35_patched, str(qwen35_path), "exec")
    qwen35_path.write_text(qwen35_patched)
    print(f"patched {qwen35_path}")


if __name__ == "__main__":
    main()
