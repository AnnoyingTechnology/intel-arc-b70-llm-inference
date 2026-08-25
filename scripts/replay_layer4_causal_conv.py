#!/usr/bin/env python3
"""Replay the first GDN kernel with the exact live n5 projection outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def report(label: str, tensors: list[torch.Tensor]) -> None:
    names = ("q", "k", "v", "b", "a")
    fields = []
    for name, tensor in zip(names, tensors, strict=True):
        host = tensor.detach().cpu()
        fields.append(
            f"{name}={int(torch.isfinite(host).sum())}/{host.numel()}"
        )
    print(f"{label}: " + " ".join(fields))


def run(
    projections: dict,
    stage: dict,
    graph: bool,
    fused: bool,
    exact_physical: bool,
    state_layout: str,
    poison_active_state: bool,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    tensors = stage["tensors"]
    static = stage["static"]
    metadata = stage["metadata"]

    rows = 5 if exact_physical else projections["qkvz_n5"].shape[0]
    qkvz = projections["qkvz_n5"][:rows].contiguous().to("xpu")
    ba = projections["ba_n5"][:rows].contiguous().to("xpu")
    z = torch.full(
        (qkvz.shape[0], metadata["num_v_heads"], metadata["head_v_dim"]),
        float("nan"),
        dtype=qkvz.dtype,
        device="xpu",
    )
    if state_layout == "live":
        # Reproduce the exact hybrid-cache view from the failing execution.
        # conv_state and ssm_state share one byte pool; each block has the
        # captured 3,407,872-byte pitch and ssm_state starts after conv_state.
        slots = 146
        block_pitch_bytes = 3_407_872
        raw_state = torch.full(
            (slots * block_pitch_bytes,),
            255 if poison_active_state else 0,
            dtype=torch.uint8,
            device="xpu",
        )
        conv_state = raw_state.view(torch.float16).as_strided(
            (146, 7, 10240),
            (1703936, 10240, 1),
            0,
        )
        ssm_state = raw_state.view(torch.float32).as_strided(
            (146, 48, 128, 128),
            (851968, 16384, 128, 1),
            35840,
        )
        if not poison_active_state:
            conv_state[0].copy_(tensors["conv_state_pre"][0].to("xpu"))
            ssm_state[0].copy_(tensors["ssm_state_pre"][0].to("xpu"))
    else:
        conv_state = tensors["conv_state_pre"].to("xpu")
        ssm_state = tensors["ssm_state_pre"].to("xpu")
        if poison_active_state:
            conv_state.fill_(float("nan"))
            ssm_state.fill_(float("nan"))
    conv_weights = static["conv_weights"].to("xpu")
    A_log = static["A_log"].to("xpu")
    dt_bias = static["dt_bias"].to("xpu")
    has_initial_state = tensors["has_initial_state"].to("xpu")
    query_start = tensors["query_start"].to("xpu")
    state_index = tensors["state_index"].to("xpu")
    core = torch.zeros(
        (qkvz.shape[0], metadata["num_v_heads"], metadata["head_v_dim"]),
        dtype=qkvz.dtype,
        device="xpu",
    )
    post_z = torch.full_like(z, float("nan"))
    post_core = torch.full_like(core, float("nan"))
    retained_pressure: list[torch.Tensor] = []

    def overwrite_freed_gdn_workspaces() -> None:
        # Match A, w and u exactly. If their local lifetime ends too early,
        # the caching allocator can hand these blocks straight back here.
        virtual_rows = 5 + 63
        retained_pressure.extend(
            (
                torch.full(
                    (metadata["num_v_heads"], virtual_rows, 64),
                    float("nan"),
                    dtype=qkvz.dtype,
                    device="xpu",
                ),
                torch.full(
                    (
                        metadata["num_v_heads"],
                        virtual_rows,
                        metadata["head_k_dim"],
                    ),
                    float("nan"),
                    dtype=qkvz.dtype,
                    device="xpu",
                ),
                torch.full(
                    (
                        metadata["num_v_heads"],
                        virtual_rows,
                        metadata["head_v_dim"],
                    ),
                    float("nan"),
                    dtype=qkvz.dtype,
                    device="xpu",
                ),
            )
        )

    def invoke() -> list[torch.Tensor]:
        if fused:
            torch.ops._xpu_C.gdn_attention(
                core,
                z,
                qkvz,
                ba,
                metadata["num_k_heads"],
                metadata["num_v_heads"],
                metadata["head_k_dim"],
                metadata["head_v_dim"],
                conv_state,
                ssm_state,
                conv_weights,
                None,
                "silu",
                A_log,
                dt_bias,
                1,
                0,
                0,
                has_initial_state,
                query_start,
                None,
                state_index,
                None,
                None,
                None,
                None,
                5,
                metadata["tp_size"],
                not metadata["gqa_interleaved_layout"],
            )
            overwrite_freed_gdn_workspaces()
            post_z.copy_(z)
            post_core.copy_(core)
            return []
        outputs = list(
            torch.ops._xpu_C.causal_conv1d(
                z,
                qkvz,
                ba,
                metadata["num_k_heads"],
                metadata["num_v_heads"],
                metadata["head_k_dim"],
                metadata["head_v_dim"],
                conv_state,
                conv_weights,
                None,
                "silu",
                1,
                0,
                0,
                has_initial_state,
                query_start,
                None,
                state_index,
                None,
                None,
                None,
                None,
                5,
                metadata["tp_size"],
                not metadata["gqa_interleaved_layout"],
            )
        )
        torch.ops._xpu_C.gated_delta_rule(
            core,
            *outputs,
            metadata["num_v_heads"],
            metadata["head_v_dim"],
            A_log,
            dt_bias,
            ssm_state,
            1,
            0,
            0,
            has_initial_state,
            query_start,
            None,
            state_index,
            None,
            None,
            None,
            None,
            5,
            metadata["tp_size"],
        )
        overwrite_freed_gdn_workspaces()
        post_z.copy_(z)
        post_core.copy_(core)
        return outputs

    if not graph:
        outputs = invoke()
    else:
        captured = torch.xpu.XPUGraph()
        with torch.xpu.graph(captured):
            outputs = invoke()
        captured.replay()
    torch.xpu.synchronize()
    return z, outputs, core, ssm_state, post_z, post_core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-capture", type=Path, required=True)
    parser.add_argument("--stage-capture", type=Path, required=True)
    parser.add_argument(
        "--implementation",
        choices=("both", "fused", "split"),
        default="both",
    )
    parser.add_argument(
        "--execution",
        choices=("both", "eager", "graph"),
        default="both",
    )
    parser.add_argument(
        "--physical",
        choices=("both", "physical5", "padded8"),
        default="both",
    )
    parser.add_argument(
        "--state-layout",
        choices=("compact", "live"),
        default="compact",
    )
    parser.add_argument("--poison-active-state", action="store_true")
    args = parser.parse_args()

    import vllm_xpu_kernels._xpu_C  # noqa: F401

    projections = torch.load(
        args.projection_capture, map_location="cpu", weights_only=False
    )["tensors"]
    stage = torch.load(args.stage_capture, map_location="cpu", weights_only=False)

    physical_modes = {
        "both": (True, False),
        "physical5": (True,),
        "padded8": (False,),
    }[args.physical]
    execution_modes = {
        "both": (False, True),
        "eager": (False,),
        "graph": (True,),
    }[args.execution]
    for exact_physical in physical_modes:
        physical = "physical5" if exact_physical else "padded8"
        implementations = {
            "both": (False, True),
            "fused": (True,),
            "split": (False,),
        }[args.implementation]
        for fused in implementations:
            implementation = "fused" if fused else "split-retained"
            for mode in execution_modes:
                label = (
                    f"{physical}-{implementation}-"
                    f"{'graph' if mode else 'eager'}"
                )
                z, outputs, core, ssm_state, post_z, post_core = run(
                    projections,
                    stage,
                    mode,
                    fused,
                    exact_physical,
                    args.state_layout,
                    args.poison_active_state,
                )
                if outputs:
                    report(label, outputs)
                z_host = z[:5].detach().cpu()
                print(
                    f"{label}: z={int(torch.isfinite(z_host).sum())}/"
                    f"{z_host.numel()}"
                )
                core_host = core[:5].detach().cpu()
                state_host = ssm_state[0].detach().cpu()
                print(
                    f"{label}: core={int(torch.isfinite(core_host).sum())}/"
                    f"{core_host.numel()} state="
                    f"{int(torch.isfinite(state_host).sum())}/"
                    f"{state_host.numel()}"
                )
                post_z_host = post_z[:5].detach().cpu()
                post_core_host = post_core[:5].detach().cpu()
                print(
                    f"{label}: copied-z="
                    f"{int(torch.isfinite(post_z_host).sum())}/"
                    f"{post_z_host.numel()} copied-core="
                    f"{int(torch.isfinite(post_core_host).sum())}/"
                    f"{post_core_host.numel()}"
                )


if __name__ == "__main__":
    main()
