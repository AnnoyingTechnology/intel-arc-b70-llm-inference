#!/usr/bin/env python3
"""Privacy-safe probe for the XPU GDN 64-token tail failure.

The prompt is synthetic and the response text is never printed.  A unique
cache namespace is used for every request.  The oracle is vLLM's strict JSON
serialization of requested logprobs: a non-finite distribution returns HTTP
400 with a ``nan`` error, while a healthy request returns 20 finite values.
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
import urllib.error
import urllib.request
from typing import Any


def parse_lengths(value: str) -> list[int]:
    lengths: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 1 or end < start:
                raise argparse.ArgumentTypeError(f"invalid range: {part}")
            lengths.update(range(start, end + 1))
        else:
            length = int(part)
            if length < 1:
                raise argparse.ArgumentTypeError("lengths must be positive")
            lengths.add(length)
    if not lengths:
        raise argparse.ArgumentTypeError("at least one length is required")
    return sorted(lengths)


def finite_top_logprobs(response: dict[str, Any]) -> tuple[int, int]:
    choices = response.get("choices") or []
    logprobs = (choices[0].get("logprobs") if choices else None) or {}
    values = [
        value
        for row in logprobs.get("top_logprobs") or []
        for value in (row or {}).values()
    ]
    finite = sum(
        isinstance(value, (int, float)) and math.isfinite(value) for value in values
    )
    return finite, len(values)


def probe(args: argparse.Namespace, length: int) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "prompt": " A" * length,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 1,
        "logprobs": 20,
        "stream": False,
        "cache_salt": secrets.token_urlsafe(32),
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local-b70",
        },
        method="POST",
    )
    expected_nan = length % args.modulus == args.remainder
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        prompt_tokens = (result.get("usage") or {}).get("prompt_tokens")
        finite, total = finite_top_logprobs(result)
        observed_nan = False
        status = "finite" if finite == total == 20 else "non_finite_or_missing"
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").lower()
        prompt_tokens = None
        finite = total = 0
        observed_nan = error.code == 400 and "nan" in detail
        status = "http_400_nan" if observed_nan else f"http_{error.code}"

    return {
        "requested_prompt_tokens": length,
        "observed_prompt_tokens": prompt_tokens,
        "expected_nan": expected_nan,
        "observed_nan": observed_nan,
        "finite_top_logprobs": finite,
        "top_logprobs": total,
        "status": status,
        "matched_expectation": (
            observed_nan == expected_nan
            and (prompt_tokens in (None, length))
            and (observed_nan or finite == total == 20)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="http://127.0.0.1:19622/v1/completions"
    )
    parser.add_argument("--model", default="qwen38")
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("4-6,68-70,132-134"),
        help="comma-separated lengths and inclusive ranges",
    )
    parser.add_argument("--modulus", type=int, default=64)
    parser.add_argument("--remainder", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()

    results = [probe(args, length) for length in args.lengths]
    for result in results:
        print(json.dumps(result, sort_keys=True))
    unexpected = [
        result["requested_prompt_tokens"]
        for result in results
        if not result["matched_expectation"]
    ]
    print(
        json.dumps(
            {
                "requests": len(results),
                "expected_failure_rule": (
                    f"prompt_tokens % {args.modulus} == {args.remainder}"
                ),
                "unexpected_lengths": unexpected,
            },
            sort_keys=True,
        )
    )
    if unexpected:
        sys.exit(1)


if __name__ == "__main__":
    main()
