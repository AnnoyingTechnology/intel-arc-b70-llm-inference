#!/usr/bin/env python3
"""Accept OpenCode's ``reasoning_effort=off`` alias in vLLM chat requests.

OpenCode 1.18.4 exposes the disabled reasoning variant as ``off`` and sends
that literal value to OpenAI-compatible providers. vLLM accepts ``none``.
This pinned compatibility patch admits ``off`` at validation and normalizes it
to ``none`` before rendering the chat template.
"""

from __future__ import annotations

import os
import sys


MARKER = "B70_REASONING_EFFORT_OFF_ALIAS"

TYPE_OLD = (
    'Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]'
)
TYPE_NEW = (
    'Literal["off", "none", "minimal", "low", "medium", "high", '
    f'"xhigh", "max"]  # {MARKER}'
)

KWARG_OLD = "            reasoning_effort=self.reasoning_effort,\n"
KWARG_NEW = (
    "            reasoning_effort=(\n"
    "                \"none\"\n"
    "                if self.reasoning_effort == \"off\"\n"
    "                else self.reasoning_effort\n"
    "            ),\n"
)

ENABLE_OLD = (
    '            extra_kwargs["enable_thinking"] = '
    'self.reasoning_effort != "none"\n'
)
ENABLE_NEW = (
    '            extra_kwargs["enable_thinking"] = '
    'self.reasoning_effort not in ("none", "off")\n'
)


def main() -> None:
    import vllm

    path = os.path.join(
        os.path.dirname(vllm.__file__),
        "entrypoints",
        "openai",
        "chat_completion",
        "protocol.py",
    )
    with open(path, encoding="utf-8") as source:
        text = source.read()

    if MARKER in text:
        print(f"already patched {path}")
        return

    for anchor, name in (
        (TYPE_OLD, "reasoning_effort type"),
        (KWARG_OLD, "reasoning_effort normalization"),
        (ENABLE_OLD, "enable_thinking mapping"),
    ):
        if text.count(anchor) != 1:
            sys.exit(f"expected one {name} anchor in {path}")

    text = text.replace(TYPE_OLD, TYPE_NEW, 1)
    text = text.replace(KWARG_OLD, KWARG_NEW, 1)
    text = text.replace(ENABLE_OLD, ENABLE_NEW, 1)
    compile(text, path, "exec")

    with open(path, "w", encoding="utf-8") as destination:
        destination.write(text)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
