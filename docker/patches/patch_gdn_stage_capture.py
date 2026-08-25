#!/usr/bin/env python3
"""Capture layer-4 XPU GDN stages through persistent graph mutations.

This diagnostic is inert unless ``B70_GDN_STAGE_CAPTURE_MODE=persistent``.
It records the physical eight-row graph bucket, while the trigger supplies the
active token count.  No synchronization occurs inside graph capture/replay.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_GDN_STAGE_CAPTURE_V1"

IMPORT_ANCHOR = """import os
from typing import Literal

import torch
"""

IMPORT_REPLACEMENT = """import os
from typing import Literal

# B70_GDN_STAGE_CAPTURE_V1: persistent buffers mutated by XPU graph replay.
import json
import re
import threading
import time
from pathlib import Path

import torch

_B70_GDN_STAGE_MODE = os.getenv("B70_GDN_STAGE_CAPTURE_MODE", "").strip().lower()
_B70_GDN_STAGE_DIR = Path(
    os.getenv("B70_GDN_STAGE_CAPTURE_DIR", "/b70-stage-captures")
)
_B70_GDN_STAGE_LAYERS = {
    int(value)
    for value in os.getenv("B70_GDN_STAGE_CAPTURE_LAYERS", "0,1,2,4").split(",")
    if value.strip()
}
_B70_GDN_STAGE_MAX_ROWS = int(os.getenv("B70_GDN_STAGE_CAPTURE_MAX_ROWS", "8"))
_B70_GDN_STAGE_DISPATCH_LENGTHS = {4, 5, 6}
_B70_GDN_STAGE_REFS = {}
_B70_GDN_STAGE_WATCHER = None
_B70_GDN_STAGE_CAPTURE_ID = 0


def _b70_gdn_stage_layer(prefix):
    match = re.search(r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _b70_gdn_stage_buffer(module, name, shape, dtype):
    fill_value = float("nan") if dtype.is_floating_point else 0
    module.register_buffer(
        name,
        torch.full(shape, fill_value, dtype=dtype),
        persistent=False,
    )
    return getattr(module, name)


def _b70_gdn_stage_register(module, vllm_config, prefix):
    module._b70_gdn_stage_refs = None
    module._b70_gdn_stage_dispatch_refs = {}
    layer = _b70_gdn_stage_layer(prefix)
    if (
        _B70_GDN_STAGE_MODE != "persistent"
        or layer not in _B70_GDN_STAGE_LAYERS
        or not prefix.startswith("language_model.")
    ):
        return
    dtype = vllm_config.model_config.dtype
    rows = _B70_GDN_STAGE_MAX_ROWS
    local_v_heads = module.num_v_heads // module.tp_size
    local_value_dim = local_v_heads * module.head_v_dim
    local_qkvz_dim = (
        2 * module.key_dim + 2 * module.value_dim
    ) // module.tp_size
    local_ba_dim = 2 * local_v_heads
    virtual_rows = rows + 63
    conv_depth = module.conv_kernel_size - 1 + module.num_spec
    conv_dim = (2 * module.key_dim + module.value_dim) // module.tp_size
    refs = {
        "hidden_input": _b70_gdn_stage_buffer(
            module, "_b70_gdn_hidden_input", (rows, module.hidden_size), dtype
        ),
        "projected_qkvz": _b70_gdn_stage_buffer(
            module, "_b70_gdn_projected_qkvz", (rows, local_qkvz_dim), dtype
        ),
        "projected_ba": _b70_gdn_stage_buffer(
            module, "_b70_gdn_projected_ba", (rows, local_ba_dim), dtype
        ),
        "causal_q": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_causal_q",
            (virtual_rows, module.num_k_heads // module.tp_size, module.head_k_dim),
            dtype,
        ),
        "causal_k": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_causal_k",
            (virtual_rows, module.num_k_heads // module.tp_size, module.head_k_dim),
            dtype,
        ),
        "causal_v": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_causal_v",
            (virtual_rows, local_v_heads, module.head_v_dim),
            dtype,
        ),
        "causal_b": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_causal_b",
            (local_v_heads, virtual_rows),
            torch.float32,
        ),
        "causal_a": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_causal_a",
            (local_v_heads, virtual_rows),
            torch.float32,
        ),
        "conv_state_pre": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_conv_state_pre",
            (1, conv_depth, conv_dim),
            dtype,
        ),
        "conv_state_post": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_conv_state_post",
            (1, conv_depth, conv_dim),
            dtype,
        ),
        "ssm_state_pre": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_ssm_state_pre",
            (1, local_v_heads, module.head_v_dim, module.head_k_dim),
            torch.float32,
        ),
        "ssm_state_post": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_ssm_state_post",
            (1, local_v_heads, module.head_v_dim, module.head_k_dim),
            torch.float32,
        ),
        "state_index": _b70_gdn_stage_buffer(
            module, "_b70_gdn_state_index", (1,), torch.int32
        ),
        "has_initial_state": _b70_gdn_stage_buffer(
            module, "_b70_gdn_has_initial_state", (1,), torch.bool
        ),
        "query_start": _b70_gdn_stage_buffer(
            module, "_b70_gdn_query_start", (2,), torch.int32
        ),
        "token_index": _b70_gdn_stage_buffer(
            module, "_b70_gdn_token_index", (rows,), torch.int32
        ),
        "core_output": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_core_output",
            (rows, local_v_heads, module.head_v_dim),
            dtype,
        ),
        "core_pre": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_core_pre",
            (rows, local_v_heads, module.head_v_dim),
            dtype,
        ),
        "z_output": _b70_gdn_stage_buffer(
            module,
            "_b70_gdn_z_output",
            (rows, local_v_heads, module.head_v_dim),
            dtype,
        ),
        "norm_output": _b70_gdn_stage_buffer(
            module, "_b70_gdn_norm_output", (rows, local_value_dim), dtype
        ),
        "output_projection": _b70_gdn_stage_buffer(
            module, "_b70_gdn_output_projection", (rows, module.hidden_size), dtype
        ),
    }
    refs["token_index"].copy_(torch.arange(rows, dtype=torch.int32))
    module._b70_gdn_stage_refs = refs
    dispatch_ref_names = (
        "hidden_input",
        "projected_qkvz",
        "projected_ba",
        "causal_q",
        "causal_k",
        "causal_v",
        "causal_b",
        "causal_a",
        "conv_state_pre",
        "conv_state_post",
        "ssm_state_pre",
        "ssm_state_post",
        "state_index",
        "has_initial_state",
        "query_start",
        "token_index",
        "core_pre",
        "core_output",
        "z_output",
        "norm_output",
        "output_projection",
    )
    for dispatch_length in sorted(_B70_GDN_STAGE_DISPATCH_LENGTHS):
        bank = {}
        for name in dispatch_ref_names:
            template = refs[name]
            bank[name] = _b70_gdn_stage_buffer(
                module,
                f"_b70_gdn_d{dispatch_length}_{name}",
                tuple(template.shape),
                template.dtype,
            )
        bank["write_marker"] = _b70_gdn_stage_buffer(
            module,
            f"_b70_gdn_d{dispatch_length}_write_marker",
            (1,),
            torch.int32,
        )
        bank["token_index"].copy_(torch.arange(rows, dtype=torch.int32))
        module._b70_gdn_stage_dispatch_refs[dispatch_length] = bank
    _B70_GDN_STAGE_REFS[layer] = {
        "prefix": prefix,
        "num_k_heads": module.num_k_heads,
        "num_v_heads": module.num_v_heads,
        "head_k_dim": module.head_k_dim,
        "head_v_dim": module.head_v_dim,
        "tp_size": module.tp_size,
        "gqa_interleaved_layout": module.gqa_interleaved_layout,
        "module": module,
        "refs": refs,
        "dispatch_refs": module._b70_gdn_stage_dispatch_refs,
    }
    _b70_gdn_stage_start_watcher()


def _b70_gdn_stage_copy(ref, tensor):
    value = tensor[:_B70_GDN_STAGE_MAX_ROWS]
    value = torch.nn.functional.pad(
        value,
        (0, 0, 0, _B70_GDN_STAGE_MAX_ROWS - value.shape[0]),
    )
    ref.copy_(value)


def _b70_gdn_stage_desc(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
    }


def _b70_gdn_stage_append(name, payload):
    _B70_GDN_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    with (_B70_GDN_STAGE_DIR / name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\\n")


def _b70_gdn_stage_watcher():
    global _B70_GDN_STAGE_CAPTURE_ID
    trigger_path = _B70_GDN_STAGE_DIR / "gdn-stage-trigger.json"
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
            for layer, record in _B70_GDN_STAGE_REFS.items():
                module = record["module"]
                banks = [("last", None, record["refs"])]
                banks.extend(
                    (f"dispatch-n{dispatch_length}", dispatch_length, refs)
                    for dispatch_length, refs in record["dispatch_refs"].items()
                )
                for bank_label, dispatch_length, refs in banks:
                    tensors = {
                        name: tensor.detach().to("cpu")
                        for name, tensor in refs.items()
                    }
                    static = {
                        "conv_weights": module.conv1d.weight.detach()
                        .view(
                            module.conv1d.weight.size(0),
                            module.conv1d.weight.size(2),
                        )
                        .to("cpu"),
                        "conv_bias": (
                            None
                            if module.conv1d.bias is None
                            else module.conv1d.bias.detach().to("cpu")
                        ),
                        "A_log": module.A_log.detach().to("cpu"),
                        "dt_bias": module.dt_bias.detach().to("cpu"),
                    }
                    finite = {
                        name: {
                            "active": bool(
                                torch.isfinite(tensor[:length]).all().item()
                            ),
                            "physical": bool(torch.isfinite(tensor).all().item()),
                        }
                        for name, tensor in refs.items()
                    }
                    payload = {
                        "schema_version": 2,
                        "capture_strategy": "registered-persistent-buffer-copy",
                        "trigger_label": label,
                        "active_length": length,
                        "physical_rows": _B70_GDN_STAGE_MAX_ROWS,
                        "layer": layer,
                        "capture_bank": bank_label,
                        "dispatch_length": dispatch_length,
                        "metadata": {
                            name: value
                            for name, value in record.items()
                            if name
                            not in {"refs", "dispatch_refs", "module"}
                        },
                        "tensor_metadata": {
                            name: _b70_gdn_stage_desc(tensor)
                            for name, tensor in refs.items()
                        },
                        "finite_flags": finite,
                        "static": static,
                        "state_metadata": {
                            "conv_shape": list(module.kv_cache[0].shape),
                            "conv_stride": list(module.kv_cache[0].stride()),
                            "conv_dtype": str(module.kv_cache[0].dtype),
                            "ssm_shape": list(module.kv_cache[1].shape),
                            "ssm_stride": list(module.kv_cache[1].stride()),
                            "ssm_dtype": str(module.kv_cache[1].dtype),
                        },
                        "tensors": tensors,
                    }
                    name = (
                        f"gdn-stages-{label}-n{length}-l{layer}"
                        f"-{bank_label}-c{_B70_GDN_STAGE_CAPTURE_ID:04d}.pt"
                    )
                    torch.save(payload, _B70_GDN_STAGE_DIR / name)
                    _b70_gdn_stage_append(
                        "gdn-stage-captures.jsonl",
                        {
                            "capture_file": name,
                            "trigger_label": label,
                            "active_length": length,
                            "physical_rows": _B70_GDN_STAGE_MAX_ROWS,
                            "layer": layer,
                            "capture_bank": bank_label,
                            "dispatch_length": dispatch_length,
                            "finite_flags": finite,
                        },
                    )
                    _B70_GDN_STAGE_CAPTURE_ID += 1
        except Exception as error:
            _b70_gdn_stage_append(
                "gdn-stage-errors.jsonl",
                {"time_ns": time.time_ns(), "error": repr(error)},
            )


def _b70_gdn_stage_start_watcher():
    global _B70_GDN_STAGE_WATCHER
    if _B70_GDN_STAGE_WATCHER is not None:
        return
    _B70_GDN_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    _B70_GDN_STAGE_WATCHER = threading.Thread(
        target=_b70_gdn_stage_watcher,
        name="b70-gdn-stage-capture",
        daemon=True,
    )
    _B70_GDN_STAGE_WATCHER.start()
"""

INIT_ANCHOR = """        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=reduce_results,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
"""

INIT_REPLACEMENT = """        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=reduce_results,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )
        _b70_gdn_stage_register(self, vllm_config, prefix)

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
"""

FORWARD_ANCHOR = """    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        \"\"\"
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out
"""

FORWARD_REPLACEMENT = """    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"XPU forward with opt-in, graph-persistent stage capture.\"\"\"
        num_tokens = hidden_states.size(0)
        _b70_forward_refs = self._b70_gdn_stage_refs
        if _b70_forward_refs is not None:
            _b70_gdn_stage_copy(
                _b70_forward_refs["hidden_input"], hidden_states
            )

        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        if _b70_forward_refs is not None:
            _b70_gdn_stage_copy(
                _b70_forward_refs["projected_qkvz"],
                projected_states_qkvz,
            )
            _b70_gdn_stage_copy(
                _b70_forward_refs["projected_ba"], projected_states_ba
            )

        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if _b70_forward_refs is not None:
            _b70_gdn_stage_copy(
                _b70_forward_refs["core_pre"], core_attn_out
            )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )
        if _b70_forward_refs is not None:
            _b70_gdn_stage_copy(
                _b70_forward_refs["core_output"], core_attn_out
            )
            _b70_gdn_stage_copy(_b70_forward_refs["z_output"], z)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)
        if _b70_forward_refs is not None:
            _b70_gdn_stage_copy(
                _b70_forward_refs["norm_output"], core_attn_out
            )
        out, _ = self.out_proj(core_attn_out)
        if _b70_forward_refs is not None:
            _b70_gdn_stage_copy(
                _b70_forward_refs["output_projection"], out
            )
        return out
"""


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    for anchor, replacement, name in (
        (IMPORT_ANCHOR, IMPORT_REPLACEMENT, "import"),
        (INIT_ANCHOR, INIT_REPLACEMENT, "initialization"),
        (FORWARD_ANCHOR, FORWARD_REPLACEMENT, "XPU forward"),
    ):
        if text.count(anchor) != 1:
            raise RuntimeError(f"{name} anchor changed; refusing to patch")
        text = text.replace(anchor, replacement, 1)
    return text


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
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
