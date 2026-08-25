#!/usr/bin/env python3
"""Replay exact captured GDN layers sequentially inside one XPU graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


def nonfinite(tensor: torch.Tensor) -> int:
    return int((~torch.isfinite(tensor)).sum().item())


def build_case(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    tensors = payload["tensors"]
    static = payload["static"]
    n = int(payload["active_length"])
    num_k_heads = int(metadata["num_k_heads"])
    num_v_heads = int(metadata["num_v_heads"])
    head_k_dim = int(metadata["head_k_dim"])
    head_v_dim = int(metadata["head_v_dim"])
    tp_size = int(metadata["tp_size"])
    physical_rows = int(payload["physical_rows"])
    dtype = tensors["projected_qkvz"].dtype
    qkvz = tensors["projected_qkvz"].to("xpu")
    ba = tensors["projected_ba"].to("xpu")
    return {
        "path": str(path),
        "layer": int(payload["layer"]),
        "n": n,
        "physical_rows": physical_rows,
        "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "tp_size": tp_size,
        "qkvz": qkvz,
        "ba": ba,
        "conv_weights": static["conv_weights"].to("xpu"),
        "conv_bias": (
            None if static["conv_bias"] is None else static["conv_bias"].to("xpu")
        ),
        "A_log": static["A_log"].to("xpu"),
        "dt_bias": static["dt_bias"].to("xpu"),
        "conv_state": tensors["conv_state_pre"].to("xpu"),
        "ssm_state": tensors["ssm_state_pre"].to("xpu"),
        "query_start": tensors["query_start"].to("xpu"),
        "has_initial_state": tensors["has_initial_state"].to("xpu"),
        "state_index": tensors["state_index"].to("xpu"),
        "core": torch.zeros(
            physical_rows,
            num_v_heads // tp_size,
            head_v_dim,
            dtype=dtype,
            device="xpu",
        ),
        "z": torch.zeros(
            physical_rows,
            num_v_heads // tp_size,
            head_v_dim,
            dtype=dtype,
            device="xpu",
        ),
        "intermediates": None,
    }


def causal(case: dict) -> None:
    case["intermediates"] = torch.ops._xpu_C.causal_conv1d(
        case["z"],
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
        non_spec_state_indices_tensor=case["state_index"],
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=case["n"],
        tp_size=case["tp_size"],
        reorder_input=not bool(False),
    )


def delta(case: dict) -> None:
    q, k, v, b, a = case["intermediates"]
    torch.ops._xpu_C.gated_delta_rule(
        case["core"],
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
        non_spec_state_indices_tensor=case["state_index"],
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=case["n"],
        tp_size=case["tp_size"],
    )


def execute(cases: list[dict], causal_only: bool) -> None:
    for case in cases:
        causal(case)
        if not causal_only:
            delta(case)


def reset(cases: list[dict]) -> None:
    for case in cases:
        case["conv_state"].zero_()
        case["ssm_state"].zero_()
        case["core"].zero_()
        case["z"].zero_()


def summarize(cases: list[dict], phase: str) -> dict:
    layers = []
    for call_index, case in enumerate(cases):
        item = {
            "call_index": call_index,
            "layer": case["layer"],
            "qkvz_nonfinite": nonfinite(case["qkvz"]),
            "ba_nonfinite": nonfinite(case["ba"]),
            "z_nonfinite": nonfinite(case["z"][: case["n"]]),
            "conv_state_nonfinite": nonfinite(case["conv_state"]),
            "core_nonfinite": nonfinite(case["core"][: case["n"]]),
            "ssm_state_nonfinite": nonfinite(case["ssm_state"]),
        }
        if case["intermediates"] is not None:
            item["intermediate_nonfinite"] = {
                name: nonfinite(value)
                for name, value in zip(
                    ("q", "k", "v", "b", "a"), case["intermediates"]
                )
            }
            item["intermediate_shapes"] = [
                list(value.shape) for value in case["intermediates"]
            ]
        layers.append(item)
    return {"phase": phase, "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--causal-only", action="store_true")
    parser.add_argument(
        "--repeat-first",
        type=int,
        default=0,
        help="repeat the first capture this many times instead of using all paths",
    )
    args = parser.parse_args()
    paths = args.captures
    if args.repeat_first:
        paths = [paths[0]] * args.repeat_first

    warm = [build_case(path) for path in paths]
    execute(warm, args.causal_only)
    torch.xpu.synchronize()
    print(json.dumps(summarize(warm, "warm-eager"), sort_keys=True))

    cases = [build_case(path) for path in paths]
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        execute(cases, args.causal_only)
    torch.xpu.synchronize()
    print(json.dumps(summarize(cases, "graph-capture"), sort_keys=True))

    reset(cases)
    torch.xpu.synchronize()
    graph.replay()
    torch.xpu.synchronize()
    print(json.dumps(summarize(cases, "graph-replay"), sort_keys=True))


if __name__ == "__main__":
    main()
