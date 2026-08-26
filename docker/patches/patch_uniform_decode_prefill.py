#!/usr/bin/env python3
"""Prevent prompt chunks from being dispatched as FULL decode graphs.

vLLM's legacy GPUModelRunner classified a uniform decode from batch shape
alone.  With MTP4, a one-request five-token prompt chunk has exactly the same
shape as a speculative decode step and was therefore sent through the FULL
decode graph.  The newer runner already includes ``has_prefill`` in this
decision via ``get_uniform_decode_token_count``.
"""

from pathlib import Path

import vllm


path = Path(vllm.__file__).parent / "v1/worker/gpu_model_runner.py"
text = path.read_text()
marker = "B70_FIX_UNIFORM_DECODE_PREFILL"
if marker in text:
    raise SystemExit(0)

old_classifier = '''    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
'''
new_classifier = '''    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        has_prefill: bool = False,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        return (
            (
                not has_prefill  # B70_FIX_UNIFORM_DECODE_PREFILL
                and (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
'''
if text.count(old_classifier) != 1:
    raise RuntimeError("uniform-decode classifier anchor changed")
text = text.replace(old_classifier, new_classifier, 1)

old_call = '''            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
'''
new_call = '''            num_tokens=num_tokens,
            num_reqs=num_reqs,
            has_prefill=bool(
                (self.input_batch.num_computed_tokens_cpu[:num_reqs]
                 < self.input_batch.num_prompt_tokens[:num_reqs]).any()
            ),
            force_uniform_decode=force_uniform_decode,
'''
if text.count(old_call) != 1:
    raise RuntimeError("uniform-decode call anchor changed")
text = text.replace(old_call, new_call, 1)

compile(text, str(path), "exec")
path.write_text(text)
print(f"patched {path}")
