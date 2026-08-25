#!/usr/bin/env python3
"""Select the split XPU GDN entry points for a controlled lifetime probe.

This is diagnostic instrumentation.  It leaves scheduling and token geometry
unchanged and replaces only the legacy C++ ``gdn_attention`` wrapper with its
two constituent public ops.  ``split-ephemeral`` preserves ordinary local
lifetime; ``split-retained`` keeps the five intermediate tensors alive on the
per-layer backend so XPU-graph pool reuse cannot reclaim them.
"""
from __future__ import annotations

import importlib.util
import os
import textwrap
from pathlib import Path


MODE = os.getenv("B70_GDN_LOW_LEVEL_IMPL", "fused").strip().lower()
VALID_MODES = {
    "fused",
    "split-ephemeral",
    "split-retained",
    "split-retained-inputs",
}

OLD_CALL = """    torch.ops._xpu_C.gdn_attention(
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
        num_prefills=num_prefills,  # type: ignore[attr-defined]
        num_decodes=num_decodes,  # type: ignore[attr-defined]
        num_spec_decodes=num_spec_decodes,  # type: ignore[attr-defined]
        has_initial_state=has_initial_state,  # type: ignore[attr-defined]
        non_spec_query_start_loc=non_spec_query_start_loc,  # type: ignore[attr-defined]
        non_spec_token_indx=non_spec_token_indx,  # type: ignore[attr-defined]
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,  # type: ignore[attr-defined]
        spec_query_start_loc=spec_query_start_loc,  # type: ignore[attr-defined]
        spec_token_indx=spec_token_indx,  # type: ignore[attr-defined]
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,  # type: ignore[attr-defined]
        num_actual_tokens=num_actual_tokens,  # type: ignore[attr-defined]
        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )
"""

SPLIT_CALL = """    # B70_GDN_SPLIT_LIFETIME_PROBE_V1
    _b70_stage_refs = getattr(self, "_b70_gdn_stage_refs", None)
    _b70_dispatch_refs = getattr(self, "_b70_gdn_stage_dispatch_refs", {})
    _b70_stage_refs = _b70_dispatch_refs.get(
        int(num_actual_tokens), _b70_stage_refs
    )
    _b70_state_indices = None
    if _b70_stage_refs is not None and num_prefills > 0:
        if "write_marker" in _b70_stage_refs:
            _b70_stage_refs["write_marker"].fill_(int(num_actual_tokens))
        for _b70_name, _b70_value in (
            ("projected_qkvz", projected_states_qkvz),
            ("projected_ba", projected_states_ba),
        ):
            _b70_target = _b70_stage_refs[_b70_name]
            _b70_target.copy_(
                torch.nn.functional.pad(
                    _b70_value[: _b70_target.shape[0]],
                    (
                        0,
                        0,
                        0,
                        _b70_target.shape[0]
                        - _b70_value[: _b70_target.shape[0]].shape[0],
                    ),
                )
            )
        _b70_state_indices = non_spec_state_indices_tensor.reshape(-1).to(torch.int64)
        if _b70_state_indices.numel() != 1:
            raise RuntimeError(
                "state capture requires one pure-prefill state row, got "
                f"{_b70_state_indices.numel()}"
            )
        _b70_stage_refs["state_index"].copy_(
            non_spec_state_indices_tensor.reshape(-1)
        )
        _b70_stage_refs["has_initial_state"].copy_(
            has_initial_state.reshape(-1)
        )
        _b70_stage_refs["query_start"].copy_(
            non_spec_query_start_loc.reshape(-1)
        )
        if non_spec_token_indx is not None:
            _b70_token_index = non_spec_token_indx.reshape(-1)
            if _b70_token_index.numel() > _b70_stage_refs["token_index"].numel():
                raise RuntimeError(
                    "token-index capture exceeds persistent buffer: "
                    f"{_b70_token_index.numel()} > "
                    f"{_b70_stage_refs['token_index'].numel()}"
                )
            _b70_stage_refs["token_index"].copy_(
                torch.nn.functional.pad(
                    _b70_token_index,
                    (
                        0,
                        _b70_stage_refs["token_index"].numel()
                        - _b70_token_index.numel(),
                    ),
                    value=-1,
                )
            )
        _b70_stage_refs["conv_state_pre"].copy_(
            self.kv_cache[0].index_select(0, _b70_state_indices)
        )
        _b70_stage_refs["ssm_state_pre"].copy_(
            self.kv_cache[1].index_select(0, _b70_state_indices)
        )
    _b70_gdn_intermediates = torch.ops._xpu_C.causal_conv1d(
        z,
        projected_states_qkvz,
        projected_states_ba,
        self.num_k_heads,
        self.num_v_heads,
        self.head_k_dim,
        self.head_v_dim,
        conv_state=self.kv_cache[0],
        conv_weights=conv_weights,
        conv_bias=self.conv1d.bias,
        activation=self.activation,
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
    )
    if len(_b70_gdn_intermediates) != 5:
        raise RuntimeError(
            "split causal_conv1d returned "
            f"{len(_b70_gdn_intermediates)} tensors instead of 5"
        )
    _b70_probe_mode = __B70_PROBE_MODE__
    if _b70_probe_mode in {"split-retained", "split-retained-inputs"}:
        _b70_retained = getattr(self, "_b70_gdn_retained_allocations", None)
        if _b70_retained is None:
            _b70_retained = {}
            self._b70_gdn_retained_allocations = _b70_retained
        _b70_values = tuple(_b70_gdn_intermediates)
        if _b70_probe_mode == "split-retained-inputs":
            _b70_values = (
                projected_states_qkvz,
                projected_states_ba,
                *_b70_values,
            )
        _b70_key = tuple(value.data_ptr() for value in _b70_values)
        _b70_retained.setdefault(_b70_key, _b70_values)
    _b70_q, _b70_k, _b70_v, _b70_b, _b70_a = _b70_gdn_intermediates
    _b70_decisive_capture = (
        num_actual_tokens == 5
        and num_prefills == 1
        and num_decodes == 0
        and num_spec_decodes == 0
        and self.prefix.endswith(".layers.4.linear_attn")
        and not getattr(self, "_b70_decisive_split_captured", False)
    )
    _b70_decisive_payload = None
    if _b70_decisive_capture:
        torch.xpu.synchronize()
        _b70_decisive_state_indices = (
            non_spec_state_indices_tensor.reshape(-1).to(torch.int64)
        )
        if _b70_decisive_state_indices.numel() != 1:
            raise RuntimeError(
                "decisive split capture requires one pure-prefill state row, got "
                f"{_b70_decisive_state_indices.numel()}"
            )
        _b70_decisive_payload = {
            "schema_version": 1,
            "stage_marker": 1,
            "metadata": {
                "prefix": self.prefix,
                "runtime_mode": forward_context.cudagraph_runtime_mode.name,
                "num_actual_tokens": num_actual_tokens,
                "num_prefills": num_prefills,
                "num_decodes": num_decodes,
                "num_spec_decodes": num_spec_decodes,
                "num_k_heads": self.num_k_heads,
                "num_v_heads": self.num_v_heads,
                "head_k_dim": self.head_k_dim,
                "head_v_dim": self.head_v_dim,
                "tp_size": self.tp_size,
                "reorder_input": not self.gqa_interleaved_layout,
                "non_spec_token_indx_none": non_spec_token_indx is None,
            },
            "tensor_metadata": {
                name: {
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                    "storage_offset": value.storage_offset(),
                }
                for name, value in {
                    "projected_qkvz": projected_states_qkvz,
                    "projected_ba": projected_states_ba,
                    "q": _b70_q,
                    "k": _b70_k,
                    "v": _b70_v,
                    "b": _b70_b,
                    "a": _b70_a,
                    "conv_state": self.kv_cache[0],
                    "ssm_state": self.kv_cache[1],
                }.items()
            },
            "tensors": {
                "projected_qkvz": projected_states_qkvz.detach().cpu(),
                "projected_ba": projected_states_ba.detach().cpu(),
                "q": _b70_q.detach().cpu(),
                "k": _b70_k.detach().cpu(),
                "v": _b70_v.detach().cpu(),
                "b": _b70_b.detach().cpu(),
                "a": _b70_a.detach().cpu(),
                "conv_state_post_causal": self.kv_cache[0]
                .index_select(0, _b70_decisive_state_indices)
                .detach()
                .cpu(),
                "ssm_state_pre_delta": self.kv_cache[1]
                .index_select(0, _b70_decisive_state_indices)
                .detach()
                .cpu(),
                "query_start": non_spec_query_start_loc.detach().cpu(),
                "state_indices": non_spec_state_indices_tensor.detach().cpu(),
                "has_initial_state": has_initial_state.detach().cpu(),
            },
        }
    if _b70_stage_refs is not None and num_prefills > 0:
        _b70_stage_refs["conv_state_post"].copy_(
            self.kv_cache[0].index_select(0, _b70_state_indices)
        )
        for _b70_name, _b70_value in (
            ("causal_q", _b70_q),
            ("causal_k", _b70_k),
            ("causal_v", _b70_v),
        ):
            _b70_target = _b70_stage_refs[_b70_name]
            _b70_target.copy_(
                torch.nn.functional.pad(
                    _b70_value,
                    (0, 0, 0, 0, 0, _b70_target.shape[0] - _b70_value.shape[0]),
                )
            )
        _b70_z_target = _b70_stage_refs["z_output"]
        _b70_z_target.copy_(
            torch.nn.functional.pad(
                z[: _b70_z_target.shape[0]],
                (
                    0,
                    0,
                    0,
                    0,
                    0,
                    _b70_z_target.shape[0]
                    - z[: _b70_z_target.shape[0]].shape[0],
                ),
            )
        )
        for _b70_name, _b70_value in (
            ("causal_b", _b70_b),
            ("causal_a", _b70_a),
        ):
            _b70_target = _b70_stage_refs[_b70_name]
            _b70_target.copy_(
                torch.nn.functional.pad(
                    _b70_value,
                    (0, _b70_target.shape[1] - _b70_value.shape[1]),
                )
            )
    torch.ops._xpu_C.gated_delta_rule(
        core_attn_out,
        _b70_q,
        _b70_k,
        _b70_v,
        _b70_b,
        _b70_a,
        self.num_v_heads,
        self.head_v_dim,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        ssm_state=self.kv_cache[1],
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
    )
    if _b70_decisive_capture:
        torch.xpu.synchronize()
        _b70_decisive_payload["stage_marker"] = 2
        _b70_decisive_payload["tensors"]["core_post_delta"] = (
            core_attn_out.detach().cpu()
        )
        _b70_decisive_payload["tensors"]["z_post_causal"] = z.detach().cpu()
        _b70_decisive_payload["tensors"]["ssm_state_post_delta"] = (
            self.kv_cache[1]
            .index_select(0, _b70_decisive_state_indices)
            .detach()
            .cpu()
        )
        torch.save(
            _b70_decisive_payload,
            "/b70-split-capture/layer4-n5-causal-delta.pt",
        )
        self._b70_decisive_split_captured = True
    if _b70_stage_refs is not None and num_prefills > 0:
        _b70_stage_refs["ssm_state_post"].copy_(
            self.kv_cache[1].index_select(0, _b70_state_indices)
        )
        _b70_core_target = _b70_stage_refs["core_output"]
        _b70_core_target.copy_(
            torch.nn.functional.pad(
                core_attn_out[: _b70_core_target.shape[0]],
                (
                    0,
                    0,
                    0,
                    0,
                    0,
                    _b70_core_target.shape[0]
                    - core_attn_out[: _b70_core_target.shape[0]].shape[0],
                ),
            )
        )
"""


def main() -> None:
    if MODE not in VALID_MODES:
        raise RuntimeError(f"invalid B70_GDN_LOW_LEVEL_IMPL={MODE!r}")
    if MODE == "fused":
        print("[B70] GDN low-level implementation: fused (unchanged)")
        return

    spec = importlib.util.find_spec("vllm._xpu_ops")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not locate vllm._xpu_ops")
    path = Path(spec.origin)
    source = path.read_text(encoding="utf-8")
    marker = "B70_GDN_SPLIT_LIFETIME_PROBE_V1"
    if marker in source:
        print(f"[B70] GDN split lifetime probe already installed: {path}")
        return
    count = source.count(OLD_CALL)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one legacy fused GDN call, found {count} in {path}"
        )
    split_call = SPLIT_CALL.replace("__B70_PROBE_MODE__", repr(MODE))
    replacement = (
        "    if not self.prefix.endswith(\".layers.4.linear_attn\"):\n"
        + textwrap.indent(OLD_CALL, "    ")
        + "    else:\n"
        + textwrap.indent(split_call, "    ")
    )
    path.write_text(source.replace(OLD_CALL, replacement), encoding="utf-8")
    print(f"[B70] GDN low-level implementation: {MODE} ({path})")


if __name__ == "__main__":
    main()
