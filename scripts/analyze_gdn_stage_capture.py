#!/usr/bin/env python3
"""Summarize graph-persistent GDN stage captures without tensor values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ORDER = (
    "hidden_input",
    "projected_qkvz",
    "projected_ba",
    "causal_q",
    "causal_k",
    "causal_v",
    "causal_b",
    "causal_a",
    "state_index",
    "has_initial_state",
    "query_start",
    "conv_state_pre",
    "conv_state_post",
    "ssm_state_pre",
    "ssm_state_post",
    "core_pre",
    "core_output",
    "z_output",
    "norm_output",
    "output_projection",
)


def summarize_tensor(
    name: str, tensor: torch.Tensor, active_length: int
) -> dict[str, object]:
    value = tensor.detach().to("cpu")
    if name in {"causal_q", "causal_k", "causal_v"}:
        active = value[: active_length + 63]
    elif name in {"causal_b", "causal_a"}:
        active = value[:, : active_length + 63]
    elif name in {
        "state_index",
        "has_initial_state",
        "query_start",
        "conv_state_pre",
        "conv_state_post",
        "ssm_state_pre",
        "ssm_state_post",
    }:
        active = value
    else:
        active = value[:active_length]
    result: dict[str, object] = {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "active_finite": bool(torch.isfinite(active).all().item()),
        "physical_finite": bool(torch.isfinite(value).all().item()),
        "active_nonfinite": int((~torch.isfinite(active)).sum().item()),
        "physical_nonfinite": int((~torch.isfinite(value)).sum().item()),
        "active_sha256": hashlib.sha256(
            active.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "physical_sha256": hashlib.sha256(
            value.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }
    finite = active[torch.isfinite(active)].float()
    result["active_max_abs_finite"] = (
        float(finite.abs().max().item()) if finite.numel() else None
    )
    result["rows"] = [
        {
            "row": row_id,
            "finite": bool(torch.isfinite(row).all().item()),
            "nonfinite": int((~torch.isfinite(row)).sum().item()),
            "max_abs_finite": (
                float(row[torch.isfinite(row)].float().abs().max().item())
                if torch.isfinite(row).any().item()
                else None
            ),
        }
        for row_id, row in enumerate(value)
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    for path in args.captures:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        active_length = int(payload["active_length"])
        result = {
            "path": str(path),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema_version": payload["schema_version"],
            "capture_strategy": payload["capture_strategy"],
            "active_length": active_length,
            "physical_rows": payload["physical_rows"],
            "layer": payload["layer"],
            "metadata": payload["metadata"],
            "tensor_metadata": payload["tensor_metadata"],
            "stages": {
                name: summarize_tensor(name, payload["tensors"][name], active_length)
                for name in ORDER
            },
        }
        if args.compact:
            result["stages"] = {
                name: {
                    key: summary[key]
                    for key in (
                        "shape",
                        "dtype",
                        "active_finite",
                        "active_nonfinite",
                        "active_max_abs_finite",
                        "active_sha256",
                    )
                }
                for name, summary in result["stages"].items()
            }
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
