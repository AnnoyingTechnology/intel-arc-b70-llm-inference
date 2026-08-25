#!/usr/bin/env python3
"""Capture the three exact layer-4 projection boundaries for n=4/5/6."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "B70_LAYER4_PROJECTION_DECISIVE_V1"


def package_file(module: str, relative: str) -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"cannot locate {module}")
    return Path(next(iter(spec.submodule_search_locations))) / relative


def main() -> None:
    path = package_file(
        "vllm",
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    )
    text = path.read_text()
    if MARKER in text:
        return

    import_anchor = """import os
from typing import Literal
"""
    import_replacement = """import os
from typing import Literal

# B70_LAYER4_PROJECTION_DECISIVE_V1
import json
import threading
import time
from pathlib import Path
"""
    if text.count(import_anchor) != 1:
        raise RuntimeError("unexpected import anchor")
    text = text.replace(import_anchor, import_replacement, 1)

    logger_anchor = """logger = init_logger(__name__)

MAX_FUSED_GDN_MTP_TOKENS = 8
"""
    logger_replacement = r'''logger = init_logger(__name__)

# B70_LAYER4_PROJECTION_DECISIVE_V1: every copy has an independent marker.
_B70_PROJECTION_CAPTURE_DIR = Path("/b70-projection-captures")
_B70_PROJECTION_RECORD = None
_B70_PROJECTION_WATCHER = None


def _b70_projection_capture_impl(
    value: torch.Tensor,
    target4: torch.Tensor,
    target5: torch.Tensor,
    target6: torch.Tensor,
    marker4: torch.Tensor,
    marker5: torch.Tensor,
    marker6: torch.Tensor,
    layer_name: str,
    stage: int,
) -> None:
    context = get_forward_context()
    metadata = context.attn_metadata
    if metadata is None:
        return
    attention = metadata[layer_name]
    length = int(attention.num_actual_tokens)
    if length == 4:
        target, marker = target4, marker4
    elif length == 5:
        target, marker = target5, marker5
    elif length == 6:
        target, marker = target6, marker6
    else:
        return
    rows = value[: target.shape[0]]
    target.copy_(
        torch.nn.functional.pad(
            rows,
            (0, 0, 0, target.shape[0] - rows.shape[0]),
        )
    )
    marker.fill_(stage)


def _b70_projection_capture_fake(
    value: torch.Tensor,
    target4: torch.Tensor,
    target5: torch.Tensor,
    target6: torch.Tensor,
    marker4: torch.Tensor,
    marker5: torch.Tensor,
    marker6: torch.Tensor,
    layer_name: str,
    stage: int,
) -> None:
    return


direct_register_custom_op(
    op_name="b70_capture_layer4_projection",
    op_func=_b70_projection_capture_impl,
    mutates_args=[
        "target4", "target5", "target6", "marker4", "marker5", "marker6"
    ],
    fake_impl=_b70_projection_capture_fake,
)


def _b70_projection_start_watcher() -> None:
    global _B70_PROJECTION_WATCHER
    if _B70_PROJECTION_WATCHER is not None:
        return

    def watch() -> None:
        trigger = _B70_PROJECTION_CAPTURE_DIR / "trigger.json"
        while True:
            try:
                if not trigger.exists():
                    time.sleep(0.05)
                    continue
                request = json.loads(trigger.read_text())
                trigger.unlink()
                torch.xpu.synchronize()
                record = _B70_PROJECTION_RECORD
                if record is None:
                    continue
                payload = {
                    "label": request.get("label", "capture"),
                    "prefix": record.prefix,
                    "tensors": {
                        name: tensor.detach().cpu()
                        for name, tensor in record._b70_projection_buffers.items()
                    },
                }
                output = _B70_PROJECTION_CAPTURE_DIR / (
                    str(request.get("label", "capture")) + ".pt"
                )
                torch.save(payload, output)
            except Exception as error:
                (_B70_PROJECTION_CAPTURE_DIR / "watcher-error.txt").write_text(
                    repr(error)
                )
                time.sleep(0.1)

    _B70_PROJECTION_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    _B70_PROJECTION_WATCHER = threading.Thread(target=watch, daemon=True)
    _B70_PROJECTION_WATCHER.start()


MAX_FUSED_GDN_MTP_TOKENS = 8
'''
    if text.count(logger_anchor) != 1:
        raise RuntimeError("unexpected logger anchor")
    text = text.replace(logger_anchor, logger_replacement, 1)

    init_anchor = """        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )
        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)
"""
    init_replacement = init_anchor + r'''

        self._b70_projection_buffers = None
        if prefix.endswith(".layers.4.linear_attn"):
            global _B70_PROJECTION_RECORD
            rows = 8
            dtype = vllm_config.model_config.dtype
            dimensions = {
                "input": self.hidden_size,
                "qkvz": (2 * self.key_dim + 2 * self.value_dim) // self.tp_size,
                "ba": 2 * self.num_v_heads // self.tp_size,
                "z": self.value_dim // self.tp_size,
                "core": self.value_dim // self.tp_size,
            }
            buffers = {}
            for stage_name, width in dimensions.items():
                for length in (4, 5, 6):
                    tensor_name = f"_b70_{stage_name}_n{length}"
                    marker_name = f"_b70_{stage_name}_marker_n{length}"
                    self.register_buffer(
                        tensor_name,
                        torch.full((rows, width), float("nan"), dtype=dtype),
                        persistent=False,
                    )
                    self.register_buffer(
                        marker_name,
                        torch.zeros((1,), dtype=torch.int32),
                        persistent=False,
                    )
                    buffers[f"{stage_name}_n{length}"] = getattr(self, tensor_name)
                    buffers[f"{stage_name}_marker_n{length}"] = getattr(
                        self, marker_name
                    )
            self._b70_projection_buffers = buffers
            _B70_PROJECTION_RECORD = self
            _b70_projection_start_watcher()
'''
    if text.count(init_anchor) != 1:
        raise RuntimeError("unexpected projection init anchor")
    text = text.replace(init_anchor, init_replacement, 1)

    forward_anchor = """        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
"""
    forward_replacement = r'''        _b70_buffers = self._b70_projection_buffers
        if _b70_buffers is not None:
            torch.ops.vllm.b70_capture_layer4_projection(
                hidden_states,
                _b70_buffers["input_n4"],
                _b70_buffers["input_n5"],
                _b70_buffers["input_n6"],
                _b70_buffers["input_marker_n4"],
                _b70_buffers["input_marker_n5"],
                _b70_buffers["input_marker_n6"],
                self.prefix,
                1,
            )
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        if _b70_buffers is not None:
            torch.ops.vllm.b70_capture_layer4_projection(
                projected_states_qkvz,
                _b70_buffers["qkvz_n4"],
                _b70_buffers["qkvz_n5"],
                _b70_buffers["qkvz_n6"],
                _b70_buffers["qkvz_marker_n4"],
                _b70_buffers["qkvz_marker_n5"],
                _b70_buffers["qkvz_marker_n6"],
                self.prefix,
                2,
            )
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        if _b70_buffers is not None:
            torch.ops.vllm.b70_capture_layer4_projection(
                projected_states_ba,
                _b70_buffers["ba_n4"],
                _b70_buffers["ba_n5"],
                _b70_buffers["ba_n6"],
                _b70_buffers["ba_marker_n4"],
                _b70_buffers["ba_marker_n5"],
                _b70_buffers["ba_marker_n6"],
                self.prefix,
                3,
            )
'''
    if text.count(forward_anchor) != 1:
        raise RuntimeError("unexpected XPU forward anchor")
    text = text.replace(forward_anchor, forward_replacement, 1)

    gdn_anchor = """        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )
"""
    gdn_replacement = gdn_anchor + r'''
        if _b70_buffers is not None:
            torch.ops.vllm.b70_capture_layer4_projection(
                z.flatten(-2),
                _b70_buffers["z_n4"],
                _b70_buffers["z_n5"],
                _b70_buffers["z_n6"],
                _b70_buffers["z_marker_n4"],
                _b70_buffers["z_marker_n5"],
                _b70_buffers["z_marker_n6"],
                self.prefix,
                4,
            )
            torch.ops.vllm.b70_capture_layer4_projection(
                core_attn_out.flatten(-2),
                _b70_buffers["core_n4"],
                _b70_buffers["core_n5"],
                _b70_buffers["core_n6"],
                _b70_buffers["core_marker_n4"],
                _b70_buffers["core_marker_n5"],
                _b70_buffers["core_marker_n6"],
                self.prefix,
                5,
            )
'''
    if text.count(gdn_anchor) != 1:
        raise RuntimeError("unexpected XPU GDN call anchor")
    text = text.replace(gdn_anchor, gdn_replacement, 1)
    path.write_text(text)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
