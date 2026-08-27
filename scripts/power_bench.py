#!/usr/bin/env python3
"""Measure B70 request latency, energy, temperature, and clock telemetry."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import statistics
import threading
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from typing import Any


PCI_DEVICE = "/sys/bus/pci/devices/0000:03:00.0"
DEFAULT_URL = "http://127.0.0.1:19622/v1/chat/completions"

SPEC_COUNTERS = {
    "vllm:spec_decode_num_drafts_total": "mtp_drafts",
    "vllm:spec_decode_num_draft_tokens_total": "mtp_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "mtp_accepted_tokens",
}


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text(encoding="ascii").strip())
    except (OSError, TypeError, UnicodeDecodeError, ValueError):
        return None


def metrics_url(request_url: str) -> str:
    parsed = urllib.parse.urlsplit(request_url)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "/metrics", "", "")
    )


def read_spec_counters(url: str, timeout_s: float) -> dict[str, float]:
    """Read aggregate speculative-decoding counters from vLLM metrics."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            text = response.read().decode("utf-8")
    except OSError:
        return {}

    counters: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric = line.split("{", 1)[0].split(None, 1)[0]
        output_name = SPEC_COUNTERS.get(metric)
        if output_name is None:
            continue
        try:
            counters[output_name] = float(line.rsplit(None, 1)[1])
        except (IndexError, ValueError):
            continue
    return counters


def spec_counter_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name in SPEC_COUNTERS.values():
        start = before.get(name)
        end = after.get(name)
        values[name] = None if start is None or end is None else end - start

    drafts = values["mtp_drafts"]
    draft_tokens = values["mtp_draft_tokens"]
    accepted = values["mtp_accepted_tokens"]
    values["mtp_acceptance_rate"] = (
        None
        if not draft_tokens or accepted is None
        else accepted / draft_tokens
    )
    values["mtp_mean_acceptance_length"] = (
        None if not drafts or accepted is None else 1.0 + accepted / drafts
    )
    return values


def locate_hwmon() -> Path:
    for candidate in glob.glob(f"{PCI_DEVICE}/hwmon/hwmon*"):
        path = Path(candidate)
        if (path / "name").read_text(encoding="ascii").strip() == "xe":
            if (path / "energy1_input").exists():
                return path
    raise RuntimeError("B70 Xe hwmon with energy1_input was not found")


class TelemetrySampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.hwmon = locate_hwmon()
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> dict[str, Any]:
        temperatures: dict[str, float] = {}
        for input_path in self.hwmon.glob("temp*_input"):
            stem = input_path.name.removesuffix("_input")
            label_path = self.hwmon / f"{stem}_label"
            label = (
                label_path.read_text(encoding="ascii").strip()
                if label_path.exists()
                else stem
            )
            value = read_int(str(input_path))
            if value is not None:
                temperatures[label] = value / 1000.0

        return {
            "monotonic_ns": time.monotonic_ns(),
            "card_energy_uj": read_int(str(self.hwmon / "energy1_input")),
            "package_energy_uj": read_int(str(self.hwmon / "energy2_input")),
            "power_cap_w": (
                read_int(str(self.hwmon / "power1_cap")) or 0
            )
            / 1_000_000.0,
            "fan_rpm": read_int(str(self.hwmon / "fan1_input")),
            "gt0_act_mhz": read_int(
                f"{PCI_DEVICE}/tile0/gt0/freq0/act_freq"
            ),
            "gt0_cur_mhz": read_int(
                f"{PCI_DEVICE}/tile0/gt0/freq0/cur_freq"
            ),
            "temperatures_c": temperatures,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.sample())
            self._stop.wait(self.interval_s)

    def start(self) -> dict[str, Any]:
        initial = self.sample()
        self.samples.append(initial)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return initial

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        final = self.sample()
        self.samples.append(final)
        return final


def energy_delta_j(start: dict[str, Any], end: dict[str, Any], key: str) -> float:
    first = start.get(key)
    last = end.get(key)
    if first is None or last is None:
        return float("nan")
    delta = last - first
    if delta < 0:
        raise RuntimeError(f"energy counter wrapped during request: {key}")
    return delta / 1_000_000.0


def summarize_telemetry(samples: list[dict[str, Any]]) -> dict[str, Any]:
    temperature_labels = sorted(
        {
            label
            for sample in samples
            for label in sample["temperatures_c"].keys()
        }
    )
    max_temperatures = {
        label: max(
            sample["temperatures_c"][label]
            for sample in samples
            if label in sample["temperatures_c"]
        )
        for label in temperature_labels
    }
    active_clocks = [
        sample["gt0_act_mhz"]
        for sample in samples
        if sample["gt0_act_mhz"] is not None and sample["gt0_act_mhz"] > 0
    ]
    fans = [
        sample["fan_rpm"]
        for sample in samples
        if sample["fan_rpm"] is not None
    ]
    return {
        "max_temperatures_c": max_temperatures,
        "max_temperature_c": max(max_temperatures.values()),
        "gt0_active_mhz_median": (
            statistics.median(active_clocks) if active_clocks else None
        ),
        "gt0_active_mhz_max": max(active_clocks) if active_clocks else None,
        "fan_rpm_max": max(fans) if fans else None,
        "sample_count": len(samples),
    }


def load_messages(
    prompt_file: Path,
    index: int,
    cold: bool,
    run_marker: str | None,
    content_multiplier: int,
    content_chars: int | None,
) -> tuple[list[dict], dict]:
    prompt_set = json.loads(prompt_file.read_text(encoding="utf-8"))
    prompts = prompt_set["prompts"]
    prompt = prompts[index % len(prompts)]
    messages = copy.deepcopy(prompt["messages"])
    if content_chars is not None:
        messages[-1]["content"] = messages[-1]["content"][:content_chars]
    if content_multiplier > 1:
        messages[-1]["content"] = messages[-1]["content"] * content_multiplier
    if cold:
        marker = f"POWERCOLD-{uuid.uuid4().hex} "
        messages[0]["content"] = marker + messages[0]["content"]
    elif run_marker is not None:
        messages[0]["content"] = run_marker + messages[0]["content"]
    metadata = {
        "prompt_file": str(prompt_file),
        "prompt_index": index % len(prompts),
        "target_prompt_tokens": prompt.get("target_tokens"),
        "family": prompt.get("family"),
        "scenario": prompt.get("scenario"),
        "source_content_sha256": prompt.get("content_sha256"),
        "cold_prefix": cold,
        "isolated_run_prefix": run_marker is not None,
        "content_multiplier": content_multiplier,
        "content_chars": content_chars,
    }
    return messages, metadata


def run_request(
    url: str,
    model: str,
    messages: list[dict],
    output_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "reasoning_effort": "none",
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    sampler = TelemetrySampler()
    request_start_ns = time.monotonic_ns()
    start_sample = sampler.start()
    first_token_ns: int | None = None
    first_sample: dict[str, Any] | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    generated = any(
                        delta.get(key)
                        for key in ("content", "reasoning_content", "tool_calls")
                    )
                    if generated and first_token_ns is None:
                        first_token_ns = time.monotonic_ns()
                        first_sample = sampler.sample()
    finally:
        end_sample = sampler.stop()
    request_end_ns = time.monotonic_ns()

    if first_token_ns is None or first_sample is None:
        raise RuntimeError("stream ended without a generated token")

    total_s = (request_end_ns - request_start_ns) / 1e9
    ttft_s = (first_token_ns - request_start_ns) / 1e9
    decode_s = (request_end_ns - first_token_ns) / 1e9
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    prompt_token_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(prompt_token_details.get("cached_tokens") or 0)
    card_energy_j = energy_delta_j(start_sample, end_sample, "card_energy_uj")
    prefill_card_energy_j = energy_delta_j(
        start_sample, first_sample, "card_energy_uj"
    )
    decode_card_energy_j = energy_delta_j(
        first_sample, end_sample, "card_energy_uj"
    )

    return {
        "request_start_ns": request_start_ns,
        "request_end_ns": request_end_ns,
        "power_cap_w": start_sample["power_cap_w"],
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "uncached_prompt_tokens": prompt_tokens - cached_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_s": ttft_s,
        "total_s": total_s,
        "decode_s": decode_s,
        "prefill_tokens_per_s": prompt_tokens / ttft_s,
        "decode_tokens_per_s": (
            (completion_tokens - 1) / decode_s
            if completion_tokens > 1 and decode_s > 0
            else None
        ),
        "card_energy_j": card_energy_j,
        "package_energy_j": energy_delta_j(
            start_sample, end_sample, "package_energy_uj"
        ),
        "average_card_power_w": card_energy_j / total_s,
        "prefill_card_energy_j": prefill_card_energy_j,
        "prefill_j_per_input_token": (
            prefill_card_energy_j / prompt_tokens if prompt_tokens else None
        ),
        "decode_card_energy_j": decode_card_energy_j,
        "decode_j_per_output_token": (
            decode_card_energy_j / max(completion_tokens - 1, 1)
        ),
        "telemetry": summarize_telemetry(sampler.samples),
    }


def median_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--fixed-index", action="store_true")
    parser.add_argument("--isolate-run", action="store_true")
    parser.add_argument("--content-multiplier", type=int, default=1)
    parser.add_argument("--content-chars", type=int)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default="qwen38")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.content_multiplier < 1:
        parser.error("--content-multiplier must be at least 1")
    if args.content_chars is not None and args.content_chars < 1:
        parser.error("--content-chars must be at least 1")

    records: list[dict[str, Any]] = []
    spec_metrics_url = metrics_url(args.url)
    run_marker = (
        f"POWERRUN-{uuid.uuid4().hex} " if args.isolate_run else None
    )
    for repeat in range(args.repeats):
        prompt_index = (
            args.start_index if args.fixed_index else args.start_index + repeat
        )
        messages, metadata = load_messages(
            args.prompt_file,
            prompt_index,
            args.cold,
            run_marker,
            args.content_multiplier,
            args.content_chars,
        )
        spec_before = read_spec_counters(spec_metrics_url, args.timeout)
        record = run_request(
            args.url,
            args.model,
            messages,
            args.output_tokens,
            args.timeout,
        )
        spec_after = read_spec_counters(spec_metrics_url, args.timeout)
        record.update(metadata)
        record.update(spec_counter_delta(spec_before, spec_after))
        record["repeat"] = repeat + 1
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if repeat + 1 < args.repeats:
            time.sleep(args.settle)

    summary = {
        "power_cap_w": median_or_none([record["power_cap_w"] for record in records]),
        "repeats": len(records),
        "output_tokens_requested": args.output_tokens,
        "prompt_tokens_median": median_or_none(
            [record["prompt_tokens"] for record in records]
        ),
        "cached_prompt_tokens": [
            record["cached_prompt_tokens"] for record in records
        ],
        "ttft_s_median": median_or_none([record["ttft_s"] for record in records]),
        "prefill_tokens_per_s_median": median_or_none(
            [record["prefill_tokens_per_s"] for record in records]
        ),
        "decode_tokens_per_s_median": median_or_none(
            [record["decode_tokens_per_s"] for record in records]
        ),
        "average_card_power_w_median": median_or_none(
            [record["average_card_power_w"] for record in records]
        ),
        "prefill_j_per_input_token_median": median_or_none(
            [record["prefill_j_per_input_token"] for record in records]
        ),
        "decode_j_per_output_token_median": median_or_none(
            [record["decode_j_per_output_token"] for record in records]
        ),
        "mtp_acceptance_rate_median": median_or_none(
            [record["mtp_acceptance_rate"] for record in records]
        ),
        "mtp_mean_acceptance_length_median": median_or_none(
            [record["mtp_mean_acceptance_length"] for record in records]
        ),
        "max_temperature_c": max(
            record["telemetry"]["max_temperature_c"] for record in records
        ),
        "gt0_active_mhz_median": median_or_none(
            [record["telemetry"]["gt0_active_mhz_median"] for record in records]
        ),
    }
    document = {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "url": args.url,
        "model": args.model,
        "prompt_file": str(args.prompt_file),
        "cold_prefix": args.cold,
        "records": records,
        "summary": summary,
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
