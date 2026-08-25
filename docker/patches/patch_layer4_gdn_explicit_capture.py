#!/usr/bin/env python3
"""Install an explicit-argument, length-keyed layer-4 GDN capture.

Every debug tensor crosses the outer torch.compile custom-op boundary as an
explicit mutated argument.  Float banks use finite sentinels and the live
API result remains the non-perturbation oracle.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_LAYER4_GDN_EXPLICIT_CAPTURE_V1"
LENGTHS = (4, 5, 6)
FLOAT_NAMES = (
    "input_qkvz",
    "input_ba",
    "q",
    "k",
    "v",
    "b",
    "a",
    "core",
    "z",
)
ALL_NAMES = FLOAT_NAMES + ("marker",)


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


def argument_names() -> list[str]:
    return [f"n{length}_{name}" for length in LENGTHS for name in ALL_NAMES]


def patch_xpu_ops() -> None:
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

# B70_LAYER4_GDN_EXPLICIT_CAPTURE_V1
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

_B70_GDN_EXPLICIT_DIR = Path(
    os.getenv("B70_GDN_EXPLICIT_DIR", "/b70-gdn-explicit-captures")
)
_B70_GDN_EXPLICIT_RECORD = None
_B70_GDN_EXPLICIT_WATCHER = None


def _b70_gdn_explicit_start_watcher() -> None:
    global _B70_GDN_EXPLICIT_WATCHER
    if _B70_GDN_EXPLICIT_WATCHER is not None:
        return

    def watch() -> None:
        trigger = _B70_GDN_EXPLICIT_DIR / "dump.json"
        while True:
            try:
                if not trigger.exists():
                    time.sleep(0.02)
                    continue
                request = json.loads(trigger.read_text())
                trigger.unlink()
                torch.xpu.synchronize()
                record = _B70_GDN_EXPLICIT_RECORD
                if record is None:
                    continue
                label = request.get("label", "layer4-gdn-explicit")
                buffers = record._b70_gdn_explicit_buffers
                payload = {
                    "schema_version": 1,
                    "label": label,
                    "host_metadata": record._b70_gdn_explicit_host_metadata,
                    "tensors": {
                        name: value.detach().cpu()
                        for name, value in buffers.items()
                    },
                }
                torch.save(
                    payload,
                    _B70_GDN_EXPLICIT_DIR / f"{label}.pt",
                )
            except Exception as error:
                (_B70_GDN_EXPLICIT_DIR / "watcher-error.txt").write_text(
                    repr(error)
                )
                time.sleep(0.1)

    _B70_GDN_EXPLICIT_DIR.mkdir(parents=True, exist_ok=True)
    _B70_GDN_EXPLICIT_WATCHER = threading.Thread(
        target=watch,
        name="b70-gdn-explicit-capture",
        daemon=True,
    )
    _B70_GDN_EXPLICIT_WATCHER.start()


if TYPE_CHECKING:
'''
    if text.count(logger_anchor) != 1:
        raise RuntimeError("unexpected vllm._xpu_ops logger anchor")
    text = text.replace(logger_anchor, logger_replacement, 1)

    params = "\n".join(f"    {name}: torch.Tensor," for name in argument_names())
    bank_rows = []
    for length in LENGTHS:
        fields = ", ".join(
            f'"{name}": n{length}_{name}' for name in ALL_NAMES
        )
        bank_rows.append(f"        {length}: {{{fields}}},")
    banks = "\n".join(bank_rows)

    implementation = f'''

# {MARKER}
def _b70_gdn_attention_core_xpu_explicit_impl(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
{params}
    layer_name: str,
) -> None:
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    global _B70_GDN_EXPLICIT_RECORD
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    _B70_GDN_EXPLICIT_RECORD = self
    _b70_gdn_explicit_start_watcher()
    attn_metadata_raw = forward_context.attn_metadata
    if attn_metadata_raw is None:
        return
    assert isinstance(attn_metadata_raw, dict)
    attn_metadata = attn_metadata_raw[self.prefix]
    assert isinstance(attn_metadata, GDNAttentionMetadata)

    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens
    num_prefills = attn_metadata.num_prefills
    num_decodes = attn_metadata.num_decodes
    num_spec_decodes = attn_metadata.num_spec_decodes
    has_initial_state = attn_metadata.has_initial_state
    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
    non_spec_state_indices_tensor = (
        non_spec_state_indices_tensor.contiguous()
        if non_spec_state_indices_tensor is not None
        else None
    )
    spec_query_start_loc = attn_metadata.spec_query_start_loc
    spec_token_indx = attn_metadata.spec_token_indx
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
    spec_sequence_masks = attn_metadata.spec_sequence_masks
    if spec_sequence_masks is not None:
        if non_spec_token_indx is not None:
            non_spec_token_indx = non_spec_token_indx.to(torch.int32)
        if spec_token_indx is not None:
            spec_token_indx = spec_token_indx.to(torch.int32)

    _b70_banks = {{
{banks}
    }}
    _b70_bank = _b70_banks.get(int(num_actual_tokens))
    _b70_capture = (
        _b70_bank is not None
        and num_prefills == 1
        and num_decodes == 0
        and num_spec_decodes == 0
        and self.prefix.endswith(".layers.4.linear_attn")
    )

    if _b70_capture:
        _b70_bank["input_qkvz"].narrow(
            0, 0, projected_states_qkvz.numel()
        ).copy_(projected_states_qkvz.flatten())
        _b70_bank["input_ba"].narrow(
            0, 0, projected_states_ba.numel()
        ).copy_(projected_states_ba.flatten())
        _b70_bank["marker"].select(0, 12).fill_(num_actual_tokens)
        _b70_bank["marker"].select(0, 13).fill_(1)

        def _b70_desc(value):
            return {{
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "data_ptr": value.data_ptr(),
                "storage_ptr": value.untyped_storage().data_ptr(),
                "storage_nbytes": value.untyped_storage().nbytes(),
                "storage_offset": value.storage_offset(),
            }}

        self._b70_gdn_explicit_host_metadata[int(num_actual_tokens)] = {{
            "runtime_mode": forward_context.cudagraph_runtime_mode.name,
            "num_actual_tokens": num_actual_tokens,
            "num_prefills": num_prefills,
            "num_decodes": num_decodes,
            "num_spec_decodes": num_spec_decodes,
            "non_spec_token_indx_none": non_spec_token_indx is None,
            "spec_token_indx_none": spec_token_indx is None,
            "projected_qkvz": _b70_desc(projected_states_qkvz),
            "projected_ba": _b70_desc(projected_states_ba),
            "core": _b70_desc(core_attn_out),
            "z": _b70_desc(z),
            "conv_state": _b70_desc(self.kv_cache[0]),
            "ssm_state": _b70_desc(self.kv_cache[1]),
            "debug": {{
                name: _b70_desc(value) for name, value in _b70_bank.items()
            }},
        }}

    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )
    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        self.num_k_heads,
        self.num_v_heads,
        self.head_k_dim,
        self.head_v_dim,
        conv_state=self.kv_cache[0],
        ssm_state=self.kv_cache[1],
        conv_weights=conv_weights,
        conv_bias=self.conv1d.bias,
        activation=self.activation,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_spec_decodes=num_spec_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_token_indx=non_spec_token_indx,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_query_start_loc=spec_query_start_loc,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        num_actual_tokens=num_actual_tokens,
        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
        debug_q=None if not _b70_capture else _b70_bank["q"],
        debug_k=None if not _b70_capture else _b70_bank["k"],
        debug_v=None if not _b70_capture else _b70_bank["v"],
        debug_b=None if not _b70_capture else _b70_bank["b"],
        debug_a=None if not _b70_capture else _b70_bank["a"],
        debug_marker=None if not _b70_capture else _b70_bank["marker"],
    )
    if _b70_capture:
        _b70_bank["core"].narrow(0, 0, core_attn_out.numel()).copy_(
            core_attn_out.flatten()
        )
        _b70_bank["z"].narrow(0, 0, z.numel()).copy_(z.flatten())
        _b70_bank["marker"].select(0, 14).fill_(3)


def _b70_gdn_attention_core_xpu_explicit_fake(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
{params}
    layer_name: str,
) -> None:
    return
'''
    function_anchor = """def _xpu_ops_deepseek_scaling_rope_impl(
"""
    if text.count(function_anchor) != 1:
        raise RuntimeError("unexpected diagnostic-function anchor")
    text = text.replace(function_anchor, implementation + "\n" + function_anchor, 1)

    registration_anchor = '''            direct_register_custom_op(
                op_name="gdn_attention_core_xpu",
                op_func=eager_break_during_capture(_gdn_attention_core_xpu_impl),
                mutates_args=["core_attn_out", "z"],
                fake_impl=_gdn_attention_core_xpu_fake,
            )
'''
    mutated = ["core_attn_out", "z", *argument_names()]
    registration_replacement = registration_anchor + f'''
            direct_register_custom_op(
                op_name="b70_gdn_attention_core_xpu_explicit",
                op_func=eager_break_during_capture(
                    _b70_gdn_attention_core_xpu_explicit_impl
                ),
                mutates_args={mutated!r},
                fake_impl=_b70_gdn_attention_core_xpu_explicit_fake,
            )
'''
    if text.count(registration_anchor) != 1:
        raise RuntimeError("unexpected diagnostic-registration anchor")
    text = text.replace(registration_anchor, registration_replacement, 1)
    compile(text, str(path), "exec")
    path.write_text(text)


def patch_model() -> None:
    path = package_file(
        "vllm",
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    )
    text = path.read_text()
    if MARKER in text:
        return

    init_anchor = """        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)
"""
    init_replacement = init_anchor + r'''

        # B70_LAYER4_GDN_EXPLICIT_CAPTURE_V1
        self._b70_gdn_explicit_buffers = None
        self._b70_gdn_explicit_host_metadata = {}
        if prefix.endswith(".layers.4.linear_attn"):
            dtype = vllm_config.model_config.dtype
            sizes = {
                "input_qkvz": 8 * 16384,
                "input_ba": 8 * 96,
                "q": 8 * 16 * 128,
                "k": 8 * 16 * 128,
                "v": 8 * 48 * 128,
                "b": 8 * 48,
                "a": 8 * 48,
                "core": 8 * 48 * 128,
                "z": 8 * 48 * 128,
            }
            buffers = {}
            sentinel = 101.0
            for length in (4, 5, 6):
                for name, size in sizes.items():
                    buffer_name = f"_b70_explicit_n{length}_{name}"
                    buffer_dtype = (
                        torch.float32 if name in ("b", "a") else dtype
                    )
                    self.register_buffer(
                        buffer_name,
                        torch.full((size,), sentinel, dtype=buffer_dtype),
                        persistent=False,
                    )
                    buffers[f"n{length}_{name}"] = getattr(self, buffer_name)
                    sentinel += 1.0
                marker_name = f"_b70_explicit_n{length}_marker"
                self.register_buffer(
                    marker_name,
                    torch.zeros((32,), dtype=torch.int32),
                    persistent=False,
                )
                buffers[f"n{length}_marker"] = getattr(self, marker_name)
            self._b70_gdn_explicit_buffers = buffers
'''
    if text.count(init_anchor) != 1:
        raise RuntimeError("unexpected model-init anchor")
    text = text.replace(init_anchor, init_replacement, 1)

    call_anchor = '''        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )
'''
    call_args = "\n".join(
        f'                _b70_buffers["{name}"],'
        for name in argument_names()
    )
    call_replacement = f'''        _b70_buffers = self._b70_gdn_explicit_buffers
        if _b70_buffers is None:
            torch.ops.vllm.gdn_attention_core_xpu(
                core_attn_out,
                z,
                projected_states_qkvz,
                projected_states_ba,
                self.prefix,
            )
        else:
            torch.ops.vllm.b70_gdn_attention_core_xpu_explicit(
                core_attn_out,
                z,
                projected_states_qkvz,
                projected_states_ba,
{call_args}
                self.prefix,
            )
'''
    if text.count(call_anchor) != 1:
        raise RuntimeError("unexpected model-call anchor")
    text = text.replace(call_anchor, call_replacement, 1)
    compile(text, str(path), "exec")
    path.write_text(text)


def main() -> None:
    patch_xpu_ops()
    patch_model()
    print("[B70] explicit length-keyed layer-4 GDN capture installed")


if __name__ == "__main__":
    main()
