#!/usr/bin/env python3
"""Summarize exact full-attention capture checkpoints without printing values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def tensor_summary(tensor: torch.Tensor | None) -> dict | None:
    if tensor is None:
        return None
    if tensor.is_floating_point():
        finite = torch.isfinite(tensor)
        nonfinite = int((~finite).sum())
        per_row = (
            [int((~torch.isfinite(row)).sum()) for row in tensor]
            if tensor.ndim > 0
            else [nonfinite]
        )
        nan = int(torch.isnan(tensor).sum())
        posinf = int(torch.isposinf(tensor).sum())
        neginf = int(torch.isneginf(tensor).sum())
    else:
        nonfinite = nan = posinf = neginf = 0
        per_row = [0] * (tensor.shape[0] if tensor.ndim else 1)
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "nonfinite": nonfinite,
        "nan": nan,
        "posinf": posinf,
        "neginf": neginf,
        "nonfinite_per_dim0": per_row,
    }


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def summarize(path: Path) -> dict:
    capture = load(path)
    metadata = capture["metadata"]
    return {
        "file": path.name,
        "metadata": {
            name: metadata.get(name)
            for name in (
                "layer_name",
                "layer_index",
                "num_actual_tokens",
                "max_query_len",
                "max_seq_len",
                "causal",
                "num_heads",
                "num_kv_heads",
                "head_size",
                "softmax_scale",
                "kv_cache_dtype",
                "flash_attn_version",
            )
        },
        "finite_flags": capture.get("finite_flags"),
        "inputs": {
            name: tensor_summary(tensor)
            for name, tensor in capture["inputs"].items()
        },
        "attention_metadata": {
            name: (
                None
                if tensor is None
                else {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "values": tensor.reshape(-1).tolist(),
                }
            )
            for name, tensor in capture["attention_metadata"].items()
        },
        "output": tensor_summary(capture["output"]),
    }


def compare(left_path: Path, right_path: Path) -> dict:
    left = load(left_path)
    right = load(right_path)
    result = {"left": left_path.name, "right": right_path.name, "tensors": {}}
    tensors = {
        "query_active": (left["inputs"]["query_active"], right["inputs"]["query_active"]),
        "key_active": (left["inputs"]["key_active"], right["inputs"]["key_active"]),
        "value_active": (left["inputs"]["value_active"], right["inputs"]["value_active"]),
        "output": (left["output"], right["output"]),
    }
    for name, (a, b) in tensors.items():
        rows = min(a.shape[0], b.shape[0])
        a = a[:rows]
        b = b[:rows]
        finite_pair = torch.isfinite(a) & torch.isfinite(b)
        max_abs = (
            float((a[finite_pair].float() - b[finite_pair].float()).abs().max())
            if bool(finite_pair.any())
            else None
        )
        result["tensors"][name] = {
            "common_rows": rows,
            "bitwise_equal": bool(torch.equal(a, b)),
            "mismatched_elements": int((a != b).sum()),
            "max_abs_finite_diff": max_abs,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.captures:
        print(json.dumps(summarize(path), sort_keys=True))
    for left, right in zip(args.captures, args.captures[1:]):
        print(json.dumps(compare(left, right), sort_keys=True))


if __name__ == "__main__":
    main()
