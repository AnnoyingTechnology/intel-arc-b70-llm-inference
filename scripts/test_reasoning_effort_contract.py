#!/usr/bin/env python3
"""Validate the B70 model-specific vLLM reasoning-effort contract."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


SUPPORTED = ("none", "low", "medium", "xhigh")
UNSUPPORTED = ("off", "minimal", "high", "max")


def enum_values(schema: dict[str, Any], property_name: str) -> set[str]:
    property_schema = schema["properties"][property_name]
    candidates = [property_schema, *property_schema.get("anyOf", [])]
    for candidate in candidates:
        if "enum" in candidate:
            return set(candidate["enum"])
    raise AssertionError(f"no enum found for {property_name}: {property_schema}")


def chat_payload(effort: str) -> dict[str, Any]:
    return {
        "model": "qwen38",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": effort,
    }


def batch_payload(effort: str) -> dict[str, Any]:
    return {
        "model": "qwen38",
        "messages": [[{"role": "user", "content": "hello"}]],
        "reasoning_effort": effort,
    }


def responses_payload(effort: str) -> dict[str, Any]:
    return {
        "model": "qwen38",
        "input": "hello",
        "reasoning": {"effort": effort},
    }


def anthropic_payload(effort: str) -> dict[str, Any]:
    return {
        "model": "qwen38",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "output_config": {"effort": effort},
    }


def assert_contract(model_type, payload_factory) -> None:
    for effort in SUPPORTED:
        model_type.model_validate(payload_factory(effort))
    for effort in UNSUPPORTED:
        try:
            model_type.model_validate(payload_factory(effort))
        except ValidationError:
            continue
        raise AssertionError(f"{model_type.__name__} accepted unsupported {effort!r}")


def main() -> None:
    assert enum_values(
        ChatCompletionRequest.model_json_schema(), "reasoning_effort"
    ) == set(SUPPORTED)
    assert enum_values(
        BatchChatCompletionRequest.model_json_schema(), "reasoning_effort"
    ) == set(SUPPORTED)

    assert_contract(ChatCompletionRequest, chat_payload)
    assert_contract(BatchChatCompletionRequest, batch_payload)
    assert_contract(ResponsesRequest, responses_payload)

    for effort in ("low", "medium", "xhigh"):
        AnthropicMessagesRequest.model_validate(anthropic_payload(effort))
    for effort in ("none", "off", "minimal", "high", "max"):
        try:
            AnthropicMessagesRequest.model_validate(anthropic_payload(effort))
        except ValidationError:
            continue
        raise AssertionError(f"AnthropicMessagesRequest accepted {effort!r}")

    for effort in SUPPORTED:
        request = ChatCompletionRequest.model_validate(chat_payload(effort))
        params = request.build_chat_params(None, "auto")
        expected_thinking = effort != "none"
        assert params.chat_template_kwargs["enable_thinking"] is expected_thinking

    print("reasoning-effort contract: PASS")


if __name__ == "__main__":
    main()
