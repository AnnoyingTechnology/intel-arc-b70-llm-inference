#!/usr/bin/env python3
"""Synthetic direct-operator probe for the XPU GDN partial-tail failure.

Run this only in a disposable XPU container while the inference service is
stopped. It uses no model weights or captured text. The split path identifies
whether causal_conv1d or gated_delta_rule first becomes non-finite and verifies
the virtual-padding contract between the two calls. The fused path exercises
the operator shape dispatched by vLLM.

A finite synthetic run does not overrule the end-to-end API reproducer, whose
real model projections remain the acceptance oracle.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128
WIDTH = 4
TP_SIZE = 1


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def parse_lengths(value: str) -> list[int]:
    lengths: set[int] = set()
    for item in parse_csv(value):
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 1 or end < start:
                raise argparse.ArgumentTypeError(f"invalid range: {item}")
            lengths.update(range(start, end + 1))
        else:
            length = int(item)
            if length < 1:
                raise argparse.ArgumentTypeError("lengths must be positive")
            lengths.add(length)
    return sorted(lengths)


def parse_bools(value: str) -> list[bool]:
    mapping = {"false": False, "true": True}
    try:
        return [mapping[item.lower()] for item in parse_csv(value)]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            "boolean values must be true or false"
        ) from error


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Case:
    length: int
    dtype_name: str
    state_dtype_name: str
    reorder_input: bool
    seed: int

    @property
    def dtype(self) -> torch.dtype:
        return DTYPES[self.dtype_name]

    @property
    def state_dtype(self) -> torch.dtype:
        if self.state_dtype_name == "model":
            return self.dtype
        return torch.float32


def nonfinite(tensor: torch.Tensor) -> int:
    return int((~torch.isfinite(tensor)).sum().item())


def nonzero(tensor: torch.Tensor) -> int:
    return int(torch.count_nonzero(tensor).item())


def make_inputs(case: Case) -> dict[str, torch.Tensor]:
    torch.manual_seed(case.seed)
    torch.xpu.manual_seed_all(case.seed)
    device = "xpu"
    dtype = case.dtype
    n = case.length

    qkvz_size = NUM_K_HEADS // TP_SIZE * (
        2 * HEAD_K_DIM
        + 2 * HEAD_V_DIM * NUM_V_HEADS // NUM_K_HEADS
    )
    ba_size = NUM_K_HEADS // TP_SIZE * (
        2 * NUM_V_HEADS // NUM_K_HEADS
    )
    qkv_size = NUM_K_HEADS // TP_SIZE * (
        2 * HEAD_K_DIM
        + HEAD_V_DIM * NUM_V_HEADS // NUM_K_HEADS
    )

    return {
        "qkvz": torch.randn(n, qkvz_size, dtype=dtype, device=device),
        "ba": torch.randn(n, ba_size, dtype=dtype, device=device),
        "conv_state": torch.zeros(
            1, WIDTH - 1, qkv_size, dtype=dtype, device=device
        ),
        "ssm_state": torch.zeros(
            1,
            NUM_V_HEADS // TP_SIZE,
            HEAD_V_DIM,
            HEAD_K_DIM,
            dtype=case.state_dtype,
            device=device,
        ),
        "conv_weights": torch.randn(
            qkv_size, WIDTH, dtype=dtype, device=device
        ),
        "A_log": torch.randn(
            NUM_V_HEADS // TP_SIZE, dtype=torch.float32, device=device
        ),
        "dt_bias": torch.randn(
            NUM_V_HEADS // TP_SIZE, dtype=dtype, device=device
        ),
        "query_start": torch.tensor(
            [0, n], dtype=torch.int32, device=device
        ),
        "has_initial_state": torch.zeros(1, dtype=torch.bool, device=device),
        "state_indices": torch.zeros(1, dtype=torch.int32, device=device),
    }


def metadata(case: Case, path: str) -> dict[str, object]:
    return {
        "path": path,
        "length": case.length,
        "tail": case.length % 64,
        "dtype": case.dtype_name,
        "state_dtype": case.state_dtype_name,
        "reorder_input": case.reorder_input,
    }


def run_split(case: Case) -> dict[str, object]:
    data = make_inputs(case)
    n = case.length
    core = torch.zeros(
        n,
        NUM_V_HEADS // TP_SIZE,
        HEAD_V_DIM,
        dtype=case.dtype,
        device="xpu",
    )
    z = torch.empty_like(core)

    q, k, v, b, a = torch.ops._xpu_C.causal_conv1d(
        z,
        data["qkvz"],
        data["ba"],
        NUM_K_HEADS,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
        conv_state=data["conv_state"],
        conv_weights=data["conv_weights"],
        conv_bias=None,
        activation="silu",
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        has_initial_state=data["has_initial_state"],
        non_spec_query_start_loc=data["query_start"],
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=data["state_indices"],
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=n,
        tp_size=TP_SIZE,
        reorder_input=case.reorder_input,
    )
    torch.xpu.synchronize()

    conv_nonfinite = {
        name: nonfinite(tensor)
        for name, tensor in (("q", q), ("k", k), ("v", v), ("b", b), ("a", a), ("z", z))
    }
    padding_nonzero_before_delta = {
        "q": nonzero(q[n:]),
        "k": nonzero(k[n:]),
        "v": nonzero(v[n:]),
        "b": nonzero(b[:, n:]),
        "a": nonzero(a[:, n:]),
    }

    torch.ops._xpu_C.gated_delta_rule(
        core,
        q,
        k,
        v,
        b,
        a,
        NUM_V_HEADS,
        HEAD_V_DIM,
        A_log=data["A_log"],
        dt_bias=data["dt_bias"],
        ssm_state=data["ssm_state"],
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        has_initial_state=data["has_initial_state"],
        non_spec_query_start_loc=data["query_start"],
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=data["state_indices"],
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=n,
        tp_size=TP_SIZE,
    )
    torch.xpu.synchronize()

    result = metadata(case, "split")
    result.update(
        {
            "conv_nonfinite": conv_nonfinite,
            "padding_nonzero_before_delta": padding_nonzero_before_delta,
            "delta_nonfinite": {
                "core": nonfinite(core),
                "last_core": nonfinite(core[-1]),
                "ssm_state": nonfinite(data["ssm_state"]),
            },
        }
    )
    result["passed"] = (
        not any(conv_nonfinite.values())
        and not any(padding_nonzero_before_delta.values())
        and not any(result["delta_nonfinite"].values())
    )
    return result


def run_fused(case: Case) -> dict[str, object]:
    data = make_inputs(case)
    n = case.length
    core = torch.zeros(
        n,
        NUM_V_HEADS // TP_SIZE,
        HEAD_V_DIM,
        dtype=case.dtype,
        device="xpu",
    )
    z = torch.empty_like(core)

    torch.ops._xpu_C.gdn_attention(
        core,
        z,
        data["qkvz"],
        data["ba"],
        NUM_K_HEADS,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
        conv_state=data["conv_state"],
        ssm_state=data["ssm_state"],
        conv_weights=data["conv_weights"],
        conv_bias=None,
        activation="silu",
        A_log=data["A_log"],
        dt_bias=data["dt_bias"],
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        has_initial_state=data["has_initial_state"],
        non_spec_query_start_loc=data["query_start"],
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=data["state_indices"],
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=n,
        tp_size=TP_SIZE,
        reorder_input=case.reorder_input,
    )
    torch.xpu.synchronize()

    output_nonfinite = {
        "core": nonfinite(core),
        "last_core": nonfinite(core[-1]),
        "z": nonfinite(z),
        "conv_state": nonfinite(data["conv_state"]),
        "ssm_state": nonfinite(data["ssm_state"]),
    }
    result = metadata(case, "fused")
    result["output_nonfinite"] = output_nonfinite
    result["passed"] = not any(output_nonfinite.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("4-6,68-70,132-134"),
    )
    parser.add_argument(
        "--dtypes",
        type=parse_csv,
        default=parse_csv("float16,bfloat16"),
    )
    parser.add_argument(
        "--state-dtypes",
        type=parse_csv,
        default=parse_csv("float32,model"),
    )
    parser.add_argument(
        "--reorder-inputs",
        type=parse_bools,
        default=parse_bools("false,true"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-fused", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    args = parser.parse_args()

    unknown_dtypes = sorted(set(args.dtypes) - set(DTYPES))
    unknown_state_dtypes = sorted(
        set(args.state_dtypes) - {"float32", "model"}
    )
    if unknown_dtypes or unknown_state_dtypes:
        parser.error(
            f"unknown dtypes: model={unknown_dtypes}, state={unknown_state_dtypes}"
        )
    if args.skip_fused and args.skip_split:
        parser.error("both probe paths cannot be skipped")
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        parser.error("an available Intel XPU is required")

    paths = []
    if not args.skip_split:
        paths.append(run_split)
    if not args.skip_fused:
        paths.append(run_fused)

    failed: list[dict[str, object]] = []
    cases = itertools.product(
        args.lengths,
        args.dtypes,
        args.state_dtypes,
        args.reorder_inputs,
    )
    requests = 0
    for length, dtype_name, state_dtype_name, reorder_input in cases:
        case = Case(
            length=length,
            dtype_name=dtype_name,
            state_dtype_name=state_dtype_name,
            reorder_input=reorder_input,
            seed=args.seed,
        )
        for runner in paths:
            requests += 1
            try:
                result = runner(case)
            except Exception as error:  # report only type and bounded message
                result = metadata(case, runner.__name__.removeprefix("run_"))
                result.update(
                    {
                        "passed": False,
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    }
                )
            print(json.dumps(result, sort_keys=True))
            if not result["passed"]:
                failed.append(result)

    print(
        json.dumps(
            {
                "cases": requests,
                "failed": len(failed),
                "shape": "NKH16/NVH48/HKD128/HVD128/W4/TP1",
            },
            sort_keys=True,
        )
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
