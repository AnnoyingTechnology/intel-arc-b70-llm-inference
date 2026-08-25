#!/usr/bin/env python3
"""Summarize exact deferred GDN capture checkpoints without printing values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def first_nonfinite(tensor: torch.Tensor) -> dict[str, Any] | None:
    mask = ~torch.isfinite(tensor)
    if not bool(mask.any()):
        return None
    coordinate = torch.nonzero(mask, as_tuple=False)[0]
    index = tuple(int(value) for value in coordinate)
    value = tensor[index]
    if tensor.dtype in (torch.float16, torch.bfloat16):
        bits = int(value.reshape(1).view(torch.int16)[0].item()) & 0xFFFF
        bit_pattern = f"0x{bits:04x}"
    elif tensor.dtype == torch.float32:
        bits = int(value.reshape(1).view(torch.int32)[0].item()) & 0xFFFFFFFF
        bit_pattern = f"0x{bits:08x}"
    else:
        bit_pattern = None
    return {
        "coordinate": list(index),
        "kind": (
            "nan"
            if bool(torch.isnan(value))
            else "positive_inf"
            if bool(value > 0)
            else "negative_inf"
        ),
        "bits": bit_pattern,
    }


def tensor_summary(tensor: torch.Tensor | None, active_rows: int | None = None):
    if tensor is None:
        return None
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "nonfinite": int((~torch.isfinite(tensor)).sum().item()),
        "first_nonfinite": first_nonfinite(tensor),
    }
    if tensor.dim() and active_rows is not None and tensor.shape[0] >= active_rows:
        active = tensor[:active_rows]
        result["active_nonfinite"] = int((~torch.isfinite(active)).sum().item())
        result["active_first_nonfinite"] = first_nonfinite(active)
        result["nonfinite_per_row"] = [
            int((~torch.isfinite(row)).sum().item()) for row in tensor
        ]
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel():
        result["finite_min"] = float(finite.min().item())
        result["finite_max"] = float(finite.max().item())
        result["finite_abs_max"] = float(finite.abs().max().item())
    return result


def summarize(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    capture = torch.load(path, map_location="cpu", weights_only=False)
    metadata = capture["metadata"]
    active_rows = int(metadata["num_actual_tokens"])
    output: dict[str, Any] = {
        "file": path.name,
        "trigger_label": capture.get("trigger_label"),
        "layer_index": metadata["layer_index"],
        "num_actual_tokens": active_rows,
        "finite_flags": capture.get("finite_flags"),
        "state_row_indices": capture["state_row_indices"],
        "dispatch": {
            key: metadata[key]
            for key in (
                "num_prefills",
                "num_decodes",
                "num_spec_decodes",
                "num_k_heads",
                "num_v_heads",
                "head_k_dim",
                "head_v_dim",
                "tp_size",
                "activation",
                "reorder_input",
                "qkvz_padding_rows",
                "ba_padding_rows",
                "core_padding_rows",
            )
        },
        "layout": metadata["tensors"],
        "metadata_values": {
            name: (None if value is None else value.tolist())
            for name, value in capture["pre"]["metadata"].items()
        },
        "pre": {},
        "post": {},
    }
    row_tensors = {
        "projected_states_qkvz",
        "projected_states_ba",
        "core_attn_out",
        "z",
    }
    for phase in ("pre", "post"):
        for name, tensor in capture[phase].items():
            if name == "metadata":
                continue
            output[phase][name] = tensor_summary(
                tensor, active_rows if name in row_tensors else None
            )
    return output, capture


def comparison(captures: list[tuple[Path, dict[str, Any]]]):
    result = []
    for left_index, (left_path, left) in enumerate(captures):
        for right_path, right in captures[left_index + 1 :]:
            common_rows = min(
                int(left["metadata"]["num_actual_tokens"]),
                int(right["metadata"]["num_actual_tokens"]),
            )
            tensors = {}
            for name in ("projected_states_qkvz", "projected_states_ba"):
                a = left["pre"][name][:common_rows]
                b = right["pre"][name][:common_rows]
                finite_pair = torch.isfinite(a) & torch.isfinite(b)
                tensors[name] = {
                    "common_rows": common_rows,
                    "left_nonfinite": int((~torch.isfinite(a)).sum().item()),
                    "right_nonfinite": int((~torch.isfinite(b)).sum().item()),
                    "finite_max_abs_diff": (
                        float((a[finite_pair] - b[finite_pair]).abs().max().item())
                        if bool(finite_pair.any())
                        else None
                    ),
                }
            result.append(
                {"left": left_path.name, "right": right_path.name, "tensors": tensors}
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    args = parser.parse_args()
    summaries = []
    loaded = []
    for path in args.captures:
        summary, capture = summarize(path)
        summaries.append(summary)
        loaded.append((path, capture))
    print(
        json.dumps(
            {"captures": summaries, "comparisons": comparison(loaded)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
