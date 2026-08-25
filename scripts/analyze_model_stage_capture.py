#!/usr/bin/env python3
"""Summarize persistent model-stage captures without printing tensor values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


STAGE_ORDER = {
    "attention": (
        "hidden_input",
        "qkv",
        "gate",
        "attention_raw",
        "attention_gated",
        "output_projection",
    ),
    "decoder": (
        "entry_hidden",
        "entry_residual",
        "input_norm",
        "attention_output",
        "post_attention_norm",
        "post_attention_residual",
        "mlp_output",
    ),
}


def tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().to("cpu")
    finite = torch.isfinite(value)
    rows = []
    for row_id, row in enumerate(value):
        row_finite = torch.isfinite(row)
        finite_values = row[row_finite].float()
        rows.append(
            {
                "row": row_id,
                "finite": bool(row_finite.all().item()),
                "nonfinite": int((~row_finite).sum().item()),
                "max_abs_finite": (
                    float(finite_values.abs().max().item())
                    if finite_values.numel()
                    else None
                ),
            }
        )
    raw = value.contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "finite": bool(finite.all().item()),
        "nonfinite": int((~finite).sum().item()),
        "max_abs_finite": (
            float(value[finite].float().abs().max().item())
            if finite.any().item()
            else None
        ),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": rows,
    }


def summarize(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    result: dict[str, object] = {
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema_version": payload.get("schema_version"),
        "capture_strategy": payload.get("capture_strategy"),
        "trigger_label": payload.get("trigger_label"),
        "length": payload.get("length"),
        "sections": {},
    }
    first_nonfinite = None
    for section in ("decoder", "attention"):
        records = payload.get(section, {})
        section_result = {}
        for layer in sorted(records, key=int):
            record = records[layer]
            stage_result = {"metadata": record.get("metadata", {})}
            for name in STAGE_ORDER[section]:
                tensor = record.get(name)
                if tensor is None:
                    continue
                summary = tensor_summary(tensor)
                stage_result[name] = summary
                if not summary["finite"] and first_nonfinite is None:
                    first_nonfinite = {
                        "section": section,
                        "layer": int(layer),
                        "stage": name,
                    }
            section_result[str(layer)] = stage_result
        result["sections"][section] = section_result
    result["first_nonfinite_in_report_order"] = first_nonfinite
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    for capture in args.captures:
        result = summarize(capture)
        if args.compact:
            compact = {
                "path": result["path"],
                "file_sha256": result["file_sha256"],
                "length": result["length"],
                "stages": {},
            }
            for section, layers in result["sections"].items():
                for layer, record in layers.items():
                    for name, summary in record.items():
                        if name == "metadata":
                            continue
                        compact["stages"][f"{section}.layer{layer}.{name}"] = {
                            key: summary[key]
                            for key in (
                                "shape",
                                "dtype",
                                "finite",
                                "nonfinite",
                                "max_abs_finite",
                                "sha256",
                            )
                        }
            print(json.dumps(compact, sort_keys=True))
        else:
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
