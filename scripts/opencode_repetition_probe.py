#!/usr/bin/env python3
"""Capture and replay an OpenCode chat request without exposing its contents.

``capture`` runs a localhost-only HTTP endpoint compatible with the OpenAI
chat-completions route.  It records the request with mode 0600 and returns a
small synthetic completion; it never forwards transcript data.

``replay`` sends that recorded request to the local B70 endpoint, applies
explicit sampling overrides, and reports only token/character statistics and
hashes.  Transcript text is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _atomic_private_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, stat.S_IRWXU)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)


def _max_run(text: str, needle: str = "!") -> int:
    maximum = current = 0
    for char in text:
        if char == needle:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") or []
    message_text = list(_walk_strings(messages))
    joined = "\n".join(message_text)
    return {
        "model": payload.get("model"),
        "messages": len(messages),
        "tools": len(payload.get("tools") or []),
        "message_string_chars": sum(map(len, message_text)),
        "message_bangs": joined.count("!"),
        "message_max_bang_run": _max_run(joined),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "top_k": payload.get("top_k"),
        "repetition_penalty": payload.get("repetition_penalty"),
        "presence_penalty": payload.get("presence_penalty"),
        "frequency_penalty": payload.get("frequency_penalty"),
        "max_tokens": payload.get("max_tokens"),
        "stream": payload.get("stream"),
    }


class CaptureHandler(BaseHTTPRequestHandler):
    capture_path: Path

    def log_message(self, format_string: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        if self.path.split("?", 1)[0] == "/v1/models":
            encoded = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen38",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local-b70",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/v1/chat/completions":
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(size)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return
        _atomic_private_write(
            self.capture_path,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        )
        print(json.dumps({"captured": str(self.capture_path), **_payload_summary(payload)}), flush=True)

        now = int(time.time())
        chunk_id = "chatcmpl-b70-capture"
        chunks = [
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": now,
                "model": payload.get("model", "qwen38"),
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "CAPTURED"}, "finish_reason": None}],
            },
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": now,
                "model": payload.get("model", "qwen38"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def capture(args: argparse.Namespace) -> None:
    CaptureHandler.capture_path = args.output
    server = ThreadingHTTPServer((args.host, args.port), CaptureHandler)
    print(json.dumps({"listening": f"http://{args.host}:{args.port}/v1", "output": str(args.output)}), flush=True)
    server.serve_forever()


def _parse_sse(
    raw: bytes,
) -> tuple[str, str, dict[str, Any] | None, dict[str, Any]]:
    content: list[str] = []
    reasoning: list[str] = []
    usage = None
    chosen_logprobs: list[float] = []
    top_logprobs: list[float] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            for item in ((choice.get("logprobs") or {}).get("content") or []):
                value = item.get("logprob")
                if isinstance(value, (int, float)):
                    chosen_logprobs.append(float(value))
                for top in item.get("top_logprobs") or []:
                    value = top.get("logprob")
                    if isinstance(value, (int, float)):
                        top_logprobs.append(float(value))

    def numeric_summary(values: list[float]) -> dict[str, Any]:
        finite = [value for value in values if math.isfinite(value)]
        return {
            "count": len(values),
            "finite": len(finite),
            "nan": sum(math.isnan(value) for value in values),
            "positive_inf": sum(value == math.inf for value in values),
            "negative_inf": sum(value == -math.inf for value in values),
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
        }

    return (
        "".join(content),
        "".join(reasoning),
        usage,
        {
            "chosen": numeric_summary(chosen_logprobs),
            "top": numeric_summary(top_logprobs),
        },
    )


def replay(args: argparse.Namespace) -> None:
    payload = json.loads(args.request.read_text())
    if args.message_count is not None:
        payload["messages"] = payload.get("messages", [])[: args.message_count]
    overrides = json.loads(args.overrides)
    payload.update(overrides)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    payload["max_tokens"] = args.max_tokens
    if args.seed is not None:
        payload["seed"] = args.seed

    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-b70"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise SystemExit(f"HTTP {error.code}: {detail}") from error
    elapsed = time.monotonic() - started
    content, reasoning, usage, logprobs = _parse_sse(raw)
    combined = reasoning + content
    result = {
        **_payload_summary(payload),
        "seed": payload.get("seed"),
        "elapsed_seconds": round(elapsed, 3),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "output_chars": len(combined),
        "output_bangs": combined.count("!"),
        "output_max_bang_run": _max_run(combined),
        "output_unique_chars": len(set(combined)),
        "output_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        "usage": usage,
        "logprobs": logprobs,
    }
    print(json.dumps(result, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    cap = commands.add_parser("capture")
    cap.add_argument("--host", default="127.0.0.1")
    cap.add_argument("--port", type=int, default=19623)
    cap.add_argument("--output", type=Path, required=True)
    cap.set_defaults(function=capture)

    rep = commands.add_parser("replay")
    rep.add_argument("--request", type=Path, required=True)
    rep.add_argument("--url", default="http://127.0.0.1:19622/v1/chat/completions")
    rep.add_argument("--overrides", default="{}")
    rep.add_argument("--max-tokens", type=int, default=128)
    rep.add_argument(
        "--message-count",
        type=int,
        help="Replay only the first N serialized messages",
    )
    rep.add_argument("--seed", type=int)
    rep.add_argument("--timeout", type=float, default=300)
    rep.set_defaults(function=replay)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
