#!/usr/bin/env python3
"""Pass the layer input into the existing length-keyed diagnostic op.

The copy runs inside ``gdn_attention_core_xpu``, where the real dispatch's
``num_actual_tokens`` is available without specializing a symbolic model
dimension.  It records only the projection input; the existing split probe
records both projection outputs into the same write-marked bank.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_LAYER4_PROJECTION_INPUT_CAPTURE_V1"


def package_file(module: str, relative: str | None = None) -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise RuntimeError(f"cannot locate {module}")
    if relative is None:
        if spec.origin is None:
            raise RuntimeError(f"cannot locate file for {module}")
        return Path(spec.origin)
    if not spec.submodule_search_locations:
        raise RuntimeError(f"cannot locate package for {module}")
    return Path(next(iter(spec.submodule_search_locations))) / relative


def patch_xpu_ops(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        return
    signature = """    projected_states_ba: torch.Tensor,
    layer_name: str,
) -> None:"""
    replacement = """    projected_states_ba: torch.Tensor,
    projection_input: torch.Tensor,
    layer_name: str,
) -> None:"""
    if text.count(signature) != 2:
        raise RuntimeError("unexpected GDN impl/fake signatures")
    text = text.replace(signature, replacement)
    anchor = """    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens
"""
    capture = """    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens

    # B70_LAYER4_PROJECTION_INPUT_CAPTURE_V1
    if ".layers.4." in self.prefix:
        _b70_banks = getattr(self, "_b70_gdn_stage_dispatch_refs", {})
        _b70_refs = _b70_banks.get(int(num_actual_tokens))
        if _b70_refs is not None:
            _b70_target = _b70_refs["hidden_input"]
            _b70_value = projection_input[: _b70_target.shape[0]]
            _b70_target.copy_(
                torch.nn.functional.pad(
                    _b70_value,
                    (0, 0, 0, _b70_target.shape[0] - _b70_value.shape[0]),
                )
            )
"""
    if text.count(anchor) != 1:
        raise RuntimeError("unexpected GDN metadata anchor")
    path.write_text(text.replace(anchor, capture, 1))


def patch_model(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        return
    anchor = """            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )"""
    replacement = """            projected_states_qkvz,
            projected_states_ba,
            hidden_states,  # B70_LAYER4_PROJECTION_INPUT_CAPTURE_V1
            self.prefix,
        )"""
    if text.count(anchor) != 1:
        raise RuntimeError("unexpected GDN model call")
    path.write_text(text.replace(anchor, replacement, 1))


def main() -> None:
    patch_xpu_ops(package_file("vllm._xpu_ops"))
    patch_model(
        package_file(
            "vllm",
            "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        )
    )
    print("[B70] layer-4 projection input capture installed")


if __name__ == "__main__":
    main()
