#!/usr/bin/env python3
"""Report finite-value boundaries in the narrow layer-4 GDN captures."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+")
    parser.add_argument("--active-tokens", type=int)
    args = parser.parse_args()

    for path in args.captures:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        print(f"FILE {path}")
        print(f"metadata={payload.get('metadata')}")
        for name, value in payload.get("tensors", {}).items():
            if not isinstance(value, torch.Tensor):
                continue
            finite = torch.isfinite(value)
            bad = torch.nonzero(~finite, as_tuple=False)
            first_bad = None if bad.numel() == 0 else bad[0].tolist()
            print(
                f"{name}: shape={tuple(value.shape)} dtype={value.dtype} "
                f"finite={int(finite.sum())}/{finite.numel()} "
                f"first_bad={first_bad}"
            )
            if args.active_tokens and value.ndim:
                if name in {"causal_b", "causal_a", "b", "a"}:
                    active = value[:, : args.active_tokens]
                else:
                    active = value[: args.active_tokens]
                active_finite = torch.isfinite(active)
                print(
                    f"  active: finite={int(active_finite.sum())}/"
                    f"{active_finite.numel()}"
                )
            if "marker" in name:
                print(f"  values={value.tolist()}")


if __name__ == "__main__":
    main()
