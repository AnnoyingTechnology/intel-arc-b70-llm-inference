#!/usr/bin/env python3
"""Measure a long-context tool call followed by tool-result ingestion."""

from __future__ import annotations

import argparse
import copy
import json
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def request(url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    http_request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start_ns = time.monotonic_ns()
    with urllib.request.urlopen(http_request, timeout=600) as response:
        body = json.loads(response.read())
    elapsed_s = (time.monotonic_ns() - start_ns) / 1e9
    return body, elapsed_s


def compact_usage(usage: dict[str, Any]) -> dict[str, Any]:
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--content-multiplier", type=int, default=4)
    parser.add_argument(
        "--url", default="http://127.0.0.1:19622/v1/chat/completions"
    )
    parser.add_argument("--model", default="qwen38")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt_set = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    messages = copy.deepcopy(prompt_set["prompts"][0]["messages"])
    messages[0]["content"] = (
        f"TOOLCACHE-{uuid.uuid4().hex} " + messages[0]["content"]
    )
    messages[-1]["content"] = (
        messages[-1]["content"] * args.content_multiplier
        + "\nYou must call lookup_record with key B70-TOOL-42. Do not answer directly."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_record",
                "description": "Look up a local record by key.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            },
        }
    ]
    common = {
        "model": args.model,
        "temperature": 0,
        "reasoning_effort": "off",
        "tools": tools,
    }

    first_payload = common | {
        "messages": messages,
        "tool_choice": "auto",
        "max_tokens": 128,
    }
    first, first_elapsed_s = request(args.url, first_payload)
    first_message = first["choices"][0]["message"]
    tool_calls = first_message.get("tool_calls") or []
    if len(tool_calls) != 1 or tool_calls[0]["function"]["name"] != "lookup_record":
        raise RuntimeError(f"unexpected tool call response: {first_message!r}")

    followup_messages = messages + [
        {
            "role": "assistant",
            "content": first_message.get("content"),
            "tool_calls": tool_calls,
        },
        {
            "role": "tool",
            "tool_call_id": tool_calls[0]["id"],
            "content": "Record B70-TOOL-42 has value cobalt-orbit-731.",
        },
    ]
    second_payload = common | {
        "messages": followup_messages,
        "tool_choice": "none",
        "max_tokens": 64,
    }
    second, second_elapsed_s = request(args.url, second_payload)
    second_message = second["choices"][0]["message"]

    document = {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "content_multiplier": args.content_multiplier,
        "first": {
            "elapsed_s": first_elapsed_s,
            "usage": compact_usage(first.get("usage") or {}),
            "finish_reason": first["choices"][0].get("finish_reason"),
            "tool_name": tool_calls[0]["function"]["name"],
            "tool_arguments": tool_calls[0]["function"]["arguments"],
        },
        "followup": {
            "elapsed_s": second_elapsed_s,
            "usage": compact_usage(second.get("usage") or {}),
            "finish_reason": second["choices"][0].get("finish_reason"),
            "content": second_message.get("content"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
