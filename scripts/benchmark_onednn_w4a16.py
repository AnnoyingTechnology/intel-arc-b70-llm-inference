#!/usr/bin/env python3
"""Benchmark Qwen3.8-27B verifier W4A16 shapes on XPU.

Use random packed weights and rotate enough copies to exceed the B70's LLC.
Repeating one compressible or cache-resident weight materially overstates
effective memory bandwidth for decode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401 - registers torch operators


# Calls per target-model verifier pass, derived from the Qwen3.8-27B trace.
SHAPES = (
    (65, 5120, 34816),
    (65, 17408, 5120),
    (48, 5120, 16384),
    (65, 6144, 5120),
    (17, 5120, 14336),
    (1, 10240, 5120),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--working-set-mb", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--hash-output",
        action="store_true",
        help="copy the final output to the host and report its SHA-256 digest",
    )
    parser.add_argument(
        "--shape-index",
        type=int,
        choices=range(len(SHAPES)),
        help="benchmark one zero-based SHAPES entry instead of the full mix",
    )
    return parser.parse_args()


def payload_bytes(m: int, k: int, n: int) -> int:
    packed_weight = k * n // 2
    fp16_group_scales = (k // 128) * n * 2
    activations = (m * k + m * n) * 2
    return packed_weight + fp16_group_scales + activations


def benchmark_shape(
    calls: int,
    m: int,
    k: int,
    n: int,
    iterations: int,
    working_set_bytes: int,
    hash_output: bool,
) -> dict[str, object]:
    resident_bytes = payload_bytes(0, k, n)
    copies = max(2, min(16, math.ceil(working_set_bytes / resident_bytes)))
    weights = [
        torch.randint(
            -(2**31),
            2**31 - 1,
            (n, k // 8),
            dtype=torch.int32,
            device="xpu",
        ).t()
        for _ in range(copies)
    ]
    scales = [
        torch.rand((k // 128, n), dtype=torch.float16, device="xpu")
        for _ in range(copies)
    ]
    zero_point = torch.tensor([8], dtype=torch.int8, device="xpu")
    activation = torch.randn((m, k), dtype=torch.float16, device="xpu")

    for index in range(copies * 3):
        output = torch.ops._xpu_C.int4_gemm_w4a16(
            activation,
            weights[index % copies],
            None,
            scales[index % copies],
            zero_point,
            128,
            None,
        )
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for index in range(iterations):
        output = torch.ops._xpu_C.int4_gemm_w4a16(
            activation,
            weights[index % copies],
            None,
            scales[index % copies],
            zero_point,
            128,
            None,
        )
    end.record()
    end.synchronize()

    milliseconds = start.elapsed_time(end) / iterations
    transferred = payload_bytes(m, k, n)
    finite = bool(torch.isfinite(output).all().cpu())
    checksum = float(output.float().sum().cpu())
    result = {
        "calls_per_cycle": calls,
        "m": m,
        "k": k,
        "n": n,
        "copies": copies,
        "working_set_mb": resident_bytes * copies / 1e6,
        "milliseconds": milliseconds,
        "payload_mb": transferred / 1e6,
        "payload_gbps": transferred / 1e6 / milliseconds,
        "finite": finite,
        "checksum": checksum,
    }
    if hash_output:
        output_bytes = output.detach().cpu().contiguous().numpy().tobytes()
        result["output_sha256"] = hashlib.sha256(output_bytes).hexdigest()
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    shapes = (
        SHAPES
        if args.shape_index is None
        else (SHAPES[args.shape_index],)
    )
    results = [
        benchmark_shape(
            calls,
            args.m,
            k,
            n,
            args.iterations,
            args.working_set_mb * 1_000_000,
            args.hash_output,
        )
        for calls, k, n in shapes
    ]
    cycle_ms = sum(
        row["calls_per_cycle"] * row["milliseconds"] for row in results
    )
    cycle_bytes = sum(
        row["calls_per_cycle"] * row["payload_mb"] * 1e6 for row in results
    )
    print(
        json.dumps(
            {
                "strategy": os.environ.get("GEMM_KERNEL", "catalog-default"),
                "iterations": args.iterations,
                "working_set_target_mb": args.working_set_mb,
                "shape_index": args.shape_index,
                "cycle_milliseconds": cycle_ms,
                "cycle_payload_gb": cycle_bytes / 1e9,
                "cycle_payload_gbps": cycle_bytes / 1e6 / cycle_ms,
                "all_finite": all(row["finite"] for row in results),
                "shapes": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
