#!/usr/bin/env python3
"""Replay an exact graph-persistent GDN capture through fused and split ops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


def count_nonfinite(tensor: torch.Tensor) -> int:
    return int((~torch.isfinite(tensor)).sum().item())


def build_case(payload: dict, conv_depth: int | None = None) -> dict:
    n = int(payload["active_length"])
    metadata = payload["metadata"]
    tensors = payload["tensors"]
    static = payload["static"]
    state_metadata = payload["state_metadata"]

    num_k_heads = int(metadata["num_k_heads"])
    num_v_heads = int(metadata["num_v_heads"])
    head_k_dim = int(metadata["head_k_dim"])
    head_v_dim = int(metadata["head_v_dim"])
    tp_size = int(metadata["tp_size"])
    qkv_size = (
        2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    ) // tp_size
    depth = (
        int(state_metadata["conv_shape"][1])
        if conv_depth is None
        else conv_depth
    )
    dtype = tensors["projected_qkvz"].dtype
    device = "xpu"
    return {
        "n": n,
        "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "tp_size": tp_size,
        "reorder_input": not bool(metadata["gqa_interleaved_layout"]),
        "conv_depth": depth,
        "qkvz": tensors["projected_qkvz"][:n].to(device),
        "ba": tensors["projected_ba"][:n].to(device),
        "conv_weights": static["conv_weights"].to(device),
        "conv_bias": (
            None if static["conv_bias"] is None else static["conv_bias"].to(device)
        ),
        "A_log": static["A_log"].to(device),
        "dt_bias": static["dt_bias"].to(device),
        "conv_state": torch.zeros(1, depth, qkv_size, dtype=dtype, device=device),
        "ssm_state": torch.zeros(
            1,
            num_v_heads // tp_size,
            head_v_dim,
            head_k_dim,
            dtype=torch.float32,
            device=device,
        ),
        "query_start": torch.tensor([0, n], dtype=torch.int32, device=device),
        "has_initial_state": torch.zeros(1, dtype=torch.bool, device=device),
        "state_indices": torch.zeros(1, dtype=torch.int32, device=device),
    }


def fused_call(case: dict, core: torch.Tensor, z: torch.Tensor) -> None:
    torch.ops._xpu_C.gdn_attention(
        core,
        z,
        case["qkvz"],
        case["ba"],
        case["num_k_heads"],
        case["num_v_heads"],
        case["head_k_dim"],
        case["head_v_dim"],
        conv_state=case["conv_state"],
        ssm_state=case["ssm_state"],
        conv_weights=case["conv_weights"],
        conv_bias=case["conv_bias"],
        activation="silu",
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        has_initial_state=case["has_initial_state"],
        non_spec_query_start_loc=case["query_start"],
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=case["state_indices"],
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=case["n"],
        tp_size=case["tp_size"],
        reorder_input=case["reorder_input"],
    )


def summarize(case: dict, path: str, core: torch.Tensor, z: torch.Tensor) -> dict:
    return {
        "path": path,
        "length": case["n"],
        "conv_depth": case["conv_depth"],
        "core_nonfinite": count_nonfinite(core),
        "z_nonfinite": count_nonfinite(z),
        "conv_state_nonfinite": count_nonfinite(case["conv_state"]),
        "ssm_state_nonfinite": count_nonfinite(case["ssm_state"]),
    }


def run_fused(payload: dict, conv_depth: int | None, graph: bool) -> dict:
    case = build_case(payload, conv_depth)
    core = torch.zeros(
        case["n"],
        case["num_v_heads"] // case["tp_size"],
        case["head_v_dim"],
        dtype=case["qkvz"].dtype,
        device="xpu",
    )
    z = torch.empty_like(core)
    if not graph:
        fused_call(case, core, z)
        torch.xpu.synchronize()
        return summarize(case, "fused-eager", core, z)

    warm = build_case(payload, conv_depth)
    warm_core = torch.zeros_like(core)
    warm_z = torch.empty_like(z)
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        fused_call(warm, warm_core, warm_z)
    stream.synchronize()

    graph_obj = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph_obj):
        fused_call(case, core, z)
    torch.xpu.synchronize()
    capture = summarize(case, "fused-graph-capture", core, z)
    core.zero_()
    z.zero_()
    case["conv_state"].zero_()
    case["ssm_state"].zero_()
    torch.xpu.synchronize()
    graph_obj.replay()
    torch.xpu.synchronize()
    replay = summarize(case, "fused-graph-replay", core, z)
    return {"capture": capture, "replay": replay}


def run_split(payload: dict, conv_depth: int | None) -> dict:
    case = build_case(payload, conv_depth)
    n = case["n"]
    core = torch.zeros(
        n,
        case["num_v_heads"] // case["tp_size"],
        case["head_v_dim"],
        dtype=case["qkvz"].dtype,
        device="xpu",
    )
    z = torch.empty_like(core)
    q, k, v, b, a = torch.ops._xpu_C.causal_conv1d(
        z,
        case["qkvz"],
        case["ba"],
        case["num_k_heads"],
        case["num_v_heads"],
        case["head_k_dim"],
        case["head_v_dim"],
        conv_state=case["conv_state"],
        conv_weights=case["conv_weights"],
        conv_bias=case["conv_bias"],
        activation="silu",
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        has_initial_state=case["has_initial_state"],
        non_spec_query_start_loc=case["query_start"],
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=case["state_indices"],
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=n,
        tp_size=case["tp_size"],
        reorder_input=case["reorder_input"],
    )
    torch.xpu.synchronize()
    conv = {
        name: count_nonfinite(tensor)
        for name, tensor in (("q", q), ("k", k), ("v", v), ("b", b), ("a", a), ("z", z))
    }
    torch.ops._xpu_C.gated_delta_rule(
        core,
        q,
        k,
        v,
        b,
        a,
        case["num_v_heads"],
        case["head_v_dim"],
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        ssm_state=case["ssm_state"],
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        has_initial_state=case["has_initial_state"],
        non_spec_query_start_loc=case["query_start"],
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=case["state_indices"],
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=n,
        tp_size=case["tp_size"],
    )
    torch.xpu.synchronize()
    result = summarize(case, "split-eager", core, z)
    result["causal_conv_nonfinite"] = conv
    result["q_shape"] = list(q.shape)
    result["b_shape"] = list(b.shape)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--conv-depths",
        default="exact,3,7",
        help="comma-separated exact or integer convolution-state depths",
    )
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()
    payload = torch.load(args.capture, map_location="cpu", weights_only=True)
    for value in args.conv_depths.split(","):
        conv_depth = None if value == "exact" else int(value)
        print(json.dumps(run_fused(payload, conv_depth, False), sort_keys=True))
        print(json.dumps(run_split(payload, conv_depth), sort_keys=True))
        if not args.skip_graph:
            print(json.dumps(run_fused(payload, conv_depth, True), sort_keys=True))


if __name__ == "__main__":
    main()
