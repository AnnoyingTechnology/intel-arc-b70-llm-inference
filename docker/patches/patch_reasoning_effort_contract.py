#!/usr/bin/env python3
"""Constrain vLLM reasoning effort to the resident Qwen model contract.

This deployment serves one fixed Qwen3.8 model.  Its chat template supports
``low``, ``medium``, and ``xhigh`` reasoning, plus ``none`` through vLLM's
``enable_thinking=False`` mapping.  vLLM's generic OpenAI protocol advertises
additional values for other model families; accepting those values here only
defers the failure to chat-template rendering.

Patch the request models before FastAPI builds OpenAPI so chat completions,
batch chat completions, render routes, Responses, and the Anthropic-compatible
Messages route all validate and advertise the model-specific contract.
"""

from __future__ import annotations

import os
import sys


CHAT_MARKER = "B70_QWEN_REASONING_EFFORT_CONTRACT_CHAT"
RESPONSES_MARKER = "B70_QWEN_REASONING_EFFORT_CONTRACT_RESPONSES"
ANTHROPIC_MARKER = "B70_QWEN_REASONING_EFFORT_CONTRACT_ANTHROPIC"

CHAT_FIELD_OLD = '''    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = Field(
'''
CHAT_FIELD_NEW = f'''    # {CHAT_MARKER}
    reasoning_effort: Literal["none", "low", "medium", "xhigh"] | None = Field(
'''

CHAT_DESCRIPTION_OLD = '''            "Currently supported values are none, minimal, low, medium, "
            "high, xhigh, and max. Reducing reasoning effort can result in "
            "faster responses and fewer tokens used on reasoning in a response. "
            "Note that 'max' is specific to the DeepSeek V4 series and is not "
            "part of the standard OpenAI API specification."
'''
CHAT_DESCRIPTION_NEW = '''            "For this deployment, supported values are none, low, medium, "
            "and xhigh. The value none disables thinking; the other values "
            "select the corresponding Qwen reasoning profile."
'''

BATCH_FIELD_OLD = '''    tool_choice: Literal["none"] | None = "none"
    include_reasoning: bool = True
'''
BATCH_FIELD_NEW = '''    tool_choice: Literal["none"] | None = "none"
    reasoning_effort: Literal["none", "low", "medium", "xhigh"] | None = Field(
        default=None,
        description=(
            "For this deployment, supported values are none, low, medium, "
            "and xhigh."
        ),
    )
    include_reasoning: bool = True
'''

RESPONSES_IMPORT_OLD = "from openai.types.shared import Metadata, Reasoning\n"
RESPONSES_IMPORT_NEW = (
    "from openai.types.shared import Metadata, Reasoning as OpenAIReasoning\n"
)
RESPONSES_CLASS_ANCHOR = '''_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


'''
RESPONSES_CLASS_REPLACEMENT = f'''_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class Reasoning(OpenAIReasoning):
    """Reasoning options supported by the resident Qwen model."""

    # {RESPONSES_MARKER}
    effort: Literal["none", "low", "medium", "xhigh"] | None = None


'''

ANTHROPIC_FIELD_OLD = '''    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
'''
ANTHROPIC_FIELD_NEW = f'''    # {ANTHROPIC_MARKER}
    effort: Literal["low", "medium", "xhigh"] | None = None
'''


def replace_once(text: str, old: str, new: str, name: str) -> str:
    if text.count(old) != 1:
        sys.exit(f"expected one {name} anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    import vllm

    package_dir = os.path.dirname(vllm.__file__)
    chat_path = os.path.join(
        package_dir,
        "entrypoints",
        "openai",
        "chat_completion",
        "protocol.py",
    )
    responses_path = os.path.join(
        package_dir,
        "entrypoints",
        "openai",
        "responses",
        "protocol.py",
    )
    anthropic_path = os.path.join(
        package_dir,
        "entrypoints",
        "anthropic",
        "protocol.py",
    )

    with open(chat_path, encoding="utf-8") as source:
        chat_text = source.read()
    with open(responses_path, encoding="utf-8") as source:
        responses_text = source.read()
    with open(anthropic_path, encoding="utf-8") as source:
        anthropic_text = source.read()

    chat_patched = CHAT_MARKER in chat_text
    responses_patched = RESPONSES_MARKER in responses_text
    anthropic_patched = ANTHROPIC_MARKER in anthropic_text
    if chat_patched and responses_patched and anthropic_patched:
        print(
            "already patched "
            f"{chat_path}, {responses_path}, and {anthropic_path}"
        )
        return
    if chat_patched or responses_patched or anthropic_patched:
        sys.exit("inconsistent partial reasoning-effort contract patch")

    chat_text = replace_once(
        chat_text,
        CHAT_FIELD_OLD,
        CHAT_FIELD_NEW,
        "chat reasoning_effort field",
    )
    chat_text = replace_once(
        chat_text,
        CHAT_DESCRIPTION_OLD,
        CHAT_DESCRIPTION_NEW,
        "chat reasoning_effort description",
    )
    chat_text = replace_once(
        chat_text,
        BATCH_FIELD_OLD,
        BATCH_FIELD_NEW,
        "batch reasoning_effort field",
    )

    responses_text = replace_once(
        responses_text,
        RESPONSES_IMPORT_OLD,
        RESPONSES_IMPORT_NEW,
        "Responses Reasoning import",
    )
    responses_text = replace_once(
        responses_text,
        RESPONSES_CLASS_ANCHOR,
        RESPONSES_CLASS_REPLACEMENT,
        "Responses Reasoning class",
    )
    anthropic_text = replace_once(
        anthropic_text,
        ANTHROPIC_FIELD_OLD,
        ANTHROPIC_FIELD_NEW,
        "Anthropic output_config effort field",
    )

    compile(chat_text, chat_path, "exec")
    compile(responses_text, responses_path, "exec")
    compile(anthropic_text, anthropic_path, "exec")

    with open(chat_path, "w", encoding="utf-8") as destination:
        destination.write(chat_text)
    with open(responses_path, "w", encoding="utf-8") as destination:
        destination.write(responses_text)
    with open(anthropic_path, "w", encoding="utf-8") as destination:
        destination.write(anthropic_text)

    print(f"patched {chat_path}")
    print(f"patched {responses_path}")
    print(f"patched {anthropic_path}")


if __name__ == "__main__":
    main()
