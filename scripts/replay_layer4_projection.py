#!/usr/bin/env python3
"""Replay the exact layer-4 projections for dispatch lengths 4/5/6."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


PREFIX = "model.language_model.layers.4.linear_attn"


def load_checkpoint_weights(model_dir: Path) -> tuple[torch.Tensor, ...]:
    shard = model_dir / "model-00001-of-00007.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        get = handle.get_tensor
        qweight = torch.cat(
            (
                get(f"{PREFIX}.in_proj_qkv.qweight"),
                get(f"{PREFIX}.in_proj_z.qweight"),
            ),
            dim=1,
        ).t().contiguous()
        scales = torch.cat(
            (
                get(f"{PREFIX}.in_proj_qkv.scales"),
                get(f"{PREFIX}.in_proj_z.scales"),
            ),
            dim=1,
        ).contiguous()
        qzeros = torch.cat(
            (
                get(f"{PREFIX}.in_proj_qkv.qzeros"),
                get(f"{PREFIX}.in_proj_z.qzeros"),
            ),
            dim=1,
        ).t().contiguous()
        ba_weight = torch.cat(
            (
                get(f"{PREFIX}.in_proj_b.weight"),
                get(f"{PREFIX}.in_proj_a.weight"),
            ),
            dim=0,
        ).to(torch.float16).contiguous()
    return qweight, scales, qzeros, ba_weight


def load_input(capture_dir: Path, length: int) -> torch.Tensor:
    pattern = str(
        capture_dir
        / (
            "gdn-stages-decisive-layer4-projection-v2-20260825-"
            f"n6-l4-dispatch-n{length}-*.pt"
        )
    )
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise RuntimeError(f"expected one capture for n={length}, got {matches}")
    capture = torch.load(matches[0], map_location="cpu", weights_only=False)
    return capture["tensors"]["hidden_input"][:length].contiguous()


def finite(label: str, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu()
    count = torch.isfinite(values).sum().item()
    print(f"{label}: finite={count}/{values.numel()} shape={tuple(values.shape)}")


def projections(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    qkvz = torch.ops._xpu_C.int4_gemm_w4a16(
        x, qweight.t(), None, scales, qzeros, 128, None
    )
    ba = F.linear(x, ba_weight)
    return qkvz, ba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    args = parser.parse_args()

    import vllm_xpu_kernels._xpu_C  # noqa: F401

    weights = tuple(
        value.to("xpu")
        for value in load_checkpoint_weights(args.model_dir)
    )
    torch.xpu.synchronize()

    for length in (4, 5, 6):
        x = load_input(args.capture_dir, length).to("xpu")
        finite(f"n{length} input", x)

        eager_qkvz, eager_ba = projections(x, *weights)
        torch.xpu.synchronize()
        finite(f"n{length} eager qkvz", eager_qkvz)
        finite(f"n{length} eager ba", eager_ba)

        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            graph_qkvz, graph_ba = projections(x, *weights)
        graph.replay()
        torch.xpu.synchronize()
        finite(f"n{length} graph qkvz", graph_qkvz)
        finite(f"n{length} graph ba", graph_ba)


if __name__ == "__main__":
    main()
