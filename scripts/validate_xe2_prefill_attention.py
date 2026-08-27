#!/usr/bin/env python3
"""Validate the exact Qwen3.8 Xe2 prefill-attention specialization."""

from __future__ import annotations

import math

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func


PAGE_SIZE = 1664
QUERY_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
CASES = ((5, 65), (49, 1665), (257, 4097))


def reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_tokens: int,
    kv_tokens: int,
) -> torch.Tensor:
    key = key_cache.flatten(0, 1)[:kv_tokens].to(torch.float16)
    value = value_cache.flatten(0, 1)[:kv_tokens].to(torch.float16)
    key = torch.repeat_interleave(key, QUERY_HEADS // KV_HEADS, dim=1)
    value = torch.repeat_interleave(value, QUERY_HEADS // KV_HEADS, dim=1)

    scores = torch.einsum(
        "qhd,khd->hqk", query * (HEAD_DIM**-0.5), key
    ).float()
    mask = torch.ones(
        (query_tokens, kv_tokens), device="xpu", dtype=torch.bool
    ).triu(diagonal=kv_tokens - query_tokens + 1)
    scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
    probability = torch.softmax(scores, dim=-1).to(torch.float16)
    return torch.einsum("hqk,khd->qhd", probability, value)


def run_case(query_tokens: int, kv_tokens: int) -> float:
    blocks = math.ceil(kv_tokens / PAGE_SIZE)
    query = torch.randn(
        (query_tokens, QUERY_HEADS, HEAD_DIM), dtype=torch.float16
    )
    key = torch.randn(
        (blocks, PAGE_SIZE, KV_HEADS, HEAD_DIM), dtype=torch.float16
    ).clamp_(-4, 4).to(torch.float8_e4m3fn)
    value = torch.randn_like(key, dtype=torch.float16).clamp_(-4, 4).to(
        torch.float8_e4m3fn
    )
    block_table = torch.arange(blocks, dtype=torch.int32).reshape(1, blocks)
    cu_seqlens_q = torch.tensor(
        [0, query_tokens], device="xpu", dtype=torch.int32
    )
    seqused_k = torch.tensor([kv_tokens], device="xpu", dtype=torch.int32)
    descale = torch.ones((), dtype=torch.float32).expand(1, KV_HEADS)

    output = flash_attn_varlen_func(
        query,
        key,
        value,
        query_tokens,
        cu_seqlens_q,
        kv_tokens,
        seqused_k=seqused_k,
        softmax_scale=HEAD_DIM**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        k_descale=descale,
        v_descale=descale,
    )
    expected = reference(
        query, key, value, query_tokens=query_tokens, kv_tokens=kv_tokens
    )
    max_abs = float((output.float() - expected.float()).abs().max().item())
    torch.testing.assert_close(output, expected, atol=1.5e-2, rtol=1.5e-2)
    return max_abs


def main() -> None:
    torch.manual_seed(4242)
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")
    for query_tokens, kv_tokens in CASES:
        max_abs = run_case(query_tokens, kv_tokens)
        print(f"q={query_tokens} kv={kv_tokens}: PASS, max_abs={max_abs:.6g}")


if __name__ == "__main__":
    main()
