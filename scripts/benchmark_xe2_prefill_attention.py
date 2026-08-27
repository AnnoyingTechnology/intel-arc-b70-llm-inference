#!/usr/bin/env python3
"""Benchmark the exact Qwen3.8-27B Xe2 prefill-attention shapes.

Run inside the pinned vLLM image while the serving container is stopped. This
isolates the existing FlashAttention kernel; it does not load or alter model
weights.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func


PAGE_SIZE = 1664
PHYSICAL_BLOCKS = 146
BLOCK_TABLE_WIDTH = 119
QUERY_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
ATTENTION_LAYERS = 17  # 16 target layers plus the MTP layer during prefill.
BF16_PEAK_TFLOPS = 182.0  # Upstream B70 benchmark preset; reference only.


@dataclass(frozen=True)
class Chunk:
    query_tokens: int
    kv_tokens: int


WORKLOADS = {
    # Trusted profiler shapes.
    "8k": (Chunk(4992, 4992), Chunk(3136, 8128), Chunk(62, 8190)),
    "32k": (
        Chunk(6656, 6656),
        Chunk(6656, 13312),
        Chunk(6656, 19968),
        Chunk(6656, 26624),
        Chunk(3328, 29952),
        Chunk(2560, 32512),
        Chunk(49, 32561),
    ),
    # The measured 99,889-token prompt is exactly 15 * 6,656 + 49.
    "100k": tuple(
        [Chunk(6656, 6656 * index) for index in range(1, 16)]
        + [Chunk(49, 99889)]
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload", choices=(*WORKLOADS, "all"), default="all"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def causal_flops(chunk: Chunk) -> int:
    q = chunk.query_tokens
    kv = chunk.kv_tokens
    attended_pairs = q * kv - q * (q - 1) // 2
    return 4 * QUERY_HEADS * HEAD_DIM * attended_pairs


def timed_call(
    chunk: Chunk,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    descale: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    q = torch.randn(
        (chunk.query_tokens, QUERY_HEADS, HEAD_DIM),
        device="xpu",
        dtype=torch.float16,
    )
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor(
        [0, chunk.query_tokens], device="xpu", dtype=torch.int32
    )
    seqused_k = torch.tensor([chunk.kv_tokens], device="xpu", dtype=torch.int32)

    def invoke() -> None:
        flash_attn_varlen_func(
            q,
            key_cache,
            value_cache,
            chunk.query_tokens,
            cu_seqlens_q,
            chunk.kv_tokens,
            seqused_k=seqused_k,
            softmax_scale=HEAD_DIM**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            out=out,
            k_descale=descale,
            v_descale=descale,
        )

    for _ in range(warmup):
        invoke()
    torch.xpu.synchronize()

    samples_ms: list[float] = []
    for _ in range(iterations):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))

    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError(f"non-finite output for {chunk}")

    median_ms = statistics.median(samples_ms)
    tflops = causal_flops(chunk) / (median_ms / 1000.0) / 1e12
    return {
        "query_tokens": chunk.query_tokens,
        "kv_tokens": chunk.kv_tokens,
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "tflops": tflops,
        "bf16_peak_percent": 100.0 * tflops / BF16_PEAK_TFLOPS,
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("warmup must be >= 0 and iterations must be >= 1")

    torch.manual_seed(4242)
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")

    # These shapes and strides match the live hybrid-cache capture.
    cache_shape = (PHYSICAL_BLOCKS, PAGE_SIZE, KV_HEADS, HEAD_DIM)
    key_cache = torch.ones(cache_shape, dtype=torch.float8_e4m3fn)
    value_cache = torch.ones_like(key_cache)
    block_table = torch.arange(
        BLOCK_TABLE_WIDTH, dtype=torch.int32
    ).reshape(1, BLOCK_TABLE_WIDTH)
    descale = torch.ones((), dtype=torch.float32).expand(1, KV_HEADS)

    names = WORKLOADS if args.workload == "all" else (args.workload,)
    report: dict[str, object] = {
        "device": torch.xpu.get_device_name(0),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "cache_shape": cache_shape,
        "workloads": {},
    }

    for name in names:
        rows = [
            timed_call(
                chunk,
                key_cache,
                value_cache,
                block_table,
                descale,
                args.warmup,
                args.iterations,
            )
            for chunk in WORKLOADS[name]
        ]
        one_layer_ms = sum(float(row["median_ms"]) for row in rows)
        modeled_model_ms = one_layer_ms * ATTENTION_LAYERS
        report["workloads"][name] = {
            "chunks": rows,
            "one_layer_ms": one_layer_ms,
            "modeled_17_layer_ms": modeled_model_ms,
        }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"device: {report['device']}")
    for name, workload in report["workloads"].items():
        print(f"\n{name}")
        for row in workload["chunks"]:
            print(
                f"  q={row['query_tokens']:5d} kv={row['kv_tokens']:6d} "
                f"{row['median_ms']:9.3f} ms  {row['tflops']:7.2f} TF/s"
            )
        print(
            "  modeled attention total: "
            f"{workload['modeled_17_layer_ms'] / 1000.0:.3f} s"
        )


if __name__ == "__main__":
    main()
