#!/usr/bin/env python3
"""Print allocation/dispatch metadata for the first layer-4 n5 GDN call."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_LAYER4_GDN_POINTER_PROBE_V1"


def main() -> None:
    spec = importlib.util.find_spec("vllm._xpu_ops")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate vllm._xpu_ops")
    path = Path(spec.origin)
    text = path.read_text()
    if MARKER in text:
        return
    anchor = """    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
"""
    replacement = """    # B70_LAYER4_GDN_POINTER_PROBE_V1
    if (
        num_actual_tokens == 5
        and num_prefills == 1
        and num_decodes == 0
        and num_spec_decodes == 0
        and self.prefix.endswith(".layers.4.linear_attn")
        and not getattr(self, "_b70_pointer_probe_printed", False)
    ):
        def _b70_desc(value):
            storage = value.untyped_storage()
            return {
                "shape": tuple(value.shape),
                "stride": tuple(value.stride()),
                "dtype": str(value.dtype),
                "data_ptr": value.data_ptr(),
                "storage_ptr": storage.data_ptr(),
                "storage_nbytes": storage.nbytes(),
                "storage_offset": value.storage_offset(),
            }
        print(
            "[B70_GDN_POINTER_PROBE]",
            {
                "mode": forward_context.cudagraph_runtime_mode.name,
                "prefix": self.prefix,
                "core": _b70_desc(core_attn_out),
                "z": _b70_desc(z),
                "qkvz": _b70_desc(projected_states_qkvz),
                "ba": _b70_desc(projected_states_ba),
                "conv_state": _b70_desc(self.kv_cache[0]),
                "ssm_state": _b70_desc(self.kv_cache[1]),
                "non_spec_token_indx_none": non_spec_token_indx is None,
                "spec_token_indx_none": spec_token_indx is None,
                "has_initial_state_none": has_initial_state is None,
                "query_start_shape": tuple(non_spec_query_start_loc.shape),
                "state_indices_shape": tuple(
                    non_spec_state_indices_tensor.shape
                ),
            },
            flush=True,
        )
        self._b70_pointer_probe_printed = True

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
"""
    if text.count(anchor) != 1:
        raise RuntimeError("unexpected fused GDN anchor")
    text = text.replace(anchor, replacement, 1)
    post_anchor = """        reorder_input=not self.gqa_interleaved_layout,
    )


def _gdn_attention_core_xpu_fake(
"""
    post_replacement = """        reorder_input=not self.gqa_interleaved_layout,
    )
    if forward_context.cudagraph_runtime_mode.name != "FULL":
        torch.xpu.synchronize()


def _gdn_attention_core_xpu_fake(
"""
    if text.count(post_anchor) != 1:
        raise RuntimeError("unexpected fused GDN call-end anchor")
    path.write_text(text.replace(post_anchor, post_replacement, 1))
    print(f"patched {path}")


if __name__ == "__main__":
    main()
