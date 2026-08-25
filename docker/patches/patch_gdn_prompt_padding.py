#!/usr/bin/env python3
"""Install an opt-in one-token guard for the Xe2 GDN ``64*N+5`` failure.

The guard runs after chat templating/tokenization and immediately before the
rendered engine input is returned. Only token prompts whose final length has
remainder five modulo 64 are changed. For chat, one tokenizer-verified space
token is inserted before the last ``<|im_end|>`` so the assistant-generation
prefix remains unchanged. Raw completions append the token.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


MARKER = "B70_GDN_PROMPT_PADDING_V1"


def target_file() -> Path:
    override = os.getenv("B70_ONLINE_RENDERER_PATH")
    if override:
        return Path(override)
    spec = importlib.util.find_spec("vllm.renderers.online_renderer")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate vllm.renderers.online_renderer")
    return Path(spec.origin)


def main() -> None:
    path = target_file()
    text = path.read_text()
    if MARKER in text:
        print(f"already patched {path}")
        return

    import_anchor = """from collections.abc import Sequence
from http import HTTPStatus
from typing import Any
"""
    import_replacement = """from collections.abc import Sequence
from http import HTTPStatus
import os
from typing import Any
"""

    logger_anchor = """logger = init_logger(__name__)


def _reused_prompt_token_ids(request: Any) -> list[int] | None:
"""
    logger_replacement = """logger = init_logger(__name__)

# B70_GDN_PROMPT_PADDING_V1: containment for upstream XPU-kernels issue #548.
_B70_GDN_PROMPT_PADDING = os.getenv("B70_GDN_PROMPT_PADDING", "0") == "1"


def _reused_prompt_token_ids(request: Any) -> list[int] | None:
"""

    method_anchor = """    def warmup(self) -> None:
        self.renderer.warmup(
            ChatParams(
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                chat_template_kwargs=self.default_chat_template_kwargs,
            )
        )

    async def render_chat(
"""
    method_replacement = r'''    def warmup(self) -> None:
        self.renderer.warmup(
            ChatParams(
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                chat_template_kwargs=self.default_chat_template_kwargs,
            )
        )

    def _b70_guard_gdn_prompt_tail(
        self, engine_inputs: list[EngineInput], *, chat: bool
    ) -> None:
        if not _B70_GDN_PROMPT_PADDING:
            return
        pad_ids = self.renderer.get_tokenizer().encode(
            " ", add_special_tokens=False
        )
        if len(pad_ids) != 1:
            raise RuntimeError(
                "B70 GDN prompt guard requires space to encode as one token"
            )
        pad_id = int(pad_ids[0])
        im_end_id = None
        if chat:
            im_end_ids = self.renderer.get_tokenizer().encode(
                "<|im_end|>", add_special_tokens=False
            )
            if len(im_end_ids) != 1:
                raise RuntimeError(
                    "B70 GDN prompt guard requires <|im_end|> to be one token"
                )
            im_end_id = int(im_end_ids[0])
        for engine_input in engine_inputs:
            # Prompt embeddings cannot be extended with a token ID. The local
            # Qwen service is text-only, but fail closed if that ever changes.
            if engine_input.get("type") == "embeds":
                continue
            token_ids = engine_input.get("prompt_token_ids")
            if not isinstance(token_ids, list) or len(token_ids) % 64 != 5:
                continue
            original_length = len(token_ids)
            if original_length + 1 > self.model_config.max_model_len:
                raise ValueError("B70 GDN prompt guard would exceed max_model_len")
            if chat:
                assert im_end_id is not None
                try:
                    insert_at = len(token_ids) - 1 - token_ids[::-1].index(im_end_id)
                except ValueError as error:
                    raise ValueError(
                        "B70 GDN chat guard found no <|im_end|> boundary"
                    ) from error
                token_ids.insert(insert_at, pad_id)
            else:
                insert_at = len(token_ids)
                token_ids.append(pad_id)
            prompt = engine_input.get("prompt")
            if isinstance(prompt, str):
                if chat:
                    marker = "<|im_end|>"
                    char_at = prompt.rfind(marker)
                    if char_at < 0:
                        raise ValueError(
                            "B70 GDN chat guard found no text <|im_end|> boundary"
                        )
                    engine_input["prompt"] = (
                        prompt[:char_at] + " " + prompt[char_at:]
                    )
                else:
                    engine_input["prompt"] = prompt + " "
            engine_input.pop("prompt_token_offsets", None)
            assistant_mask = engine_input.get("assistant_tokens_mask")
            if isinstance(assistant_mask, list):
                assistant_mask.insert(insert_at, 0)
            logger.warning(
                "B70 GDN prompt guard inserted one space token: %d -> %d (%s)",
                original_length,
                len(token_ids),
                "chat-final-message" if chat else "completion-tail",
            )

    async def render_chat(
'''

    chat_anchor = """        return conversation, engine_inputs

    def _make_request_with_harmony(
"""
    chat_replacement = """        self._b70_guard_gdn_prompt_tail(
            engine_inputs, chat=True
        )
        return conversation, engine_inputs

    def _make_request_with_harmony(
"""

    completion_anchor = """        engine_inputs = await self.preprocess_completion(
            request,
            prompt_input=request.prompt,
            prompt_embeds=request.prompt_embeds,
            skip_mm_cache=skip_mm_cache,
        )

        return engine_inputs
"""
    completion_replacement = """        engine_inputs = await self.preprocess_completion(
            request,
            prompt_input=request.prompt,
            prompt_embeds=request.prompt_embeds,
            skip_mm_cache=skip_mm_cache,
        )

        self._b70_guard_gdn_prompt_tail(engine_inputs, chat=False)
        return engine_inputs
"""

    for anchor, replacement, label in (
        (import_anchor, import_replacement, "imports"),
        (logger_anchor, logger_replacement, "configuration"),
        (method_anchor, method_replacement, "guard method"),
        (chat_anchor, chat_replacement, "chat application"),
        (completion_anchor, completion_replacement, "completion application"),
    ):
        if text.count(anchor) != 1:
            raise RuntimeError(f"unexpected {label} anchor in {path}")
        text = text.replace(anchor, replacement, 1)

    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
