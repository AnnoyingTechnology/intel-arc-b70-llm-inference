#!/usr/bin/env python3
"""Compare frozen workloads without a background telemetry polling thread."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
import urllib.request


def available_gib() -> float:
    text = Path('/proc/meminfo').read_text()
    return int(re.search(r'^MemAvailable:\s+(\d+)', text, re.M)[1]) / 1048576


def energy_snapshot() -> int | None:
    for p in Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('hwmon*/energy1_input'):
        return int(p.read_text())
    return None


def measure(api: str, payload: dict) -> dict:
    assert available_gib() >= 12, 'Less than 12 GiB host memory available'
    p = dict(payload, stream=True, stream_options={'include_usage': True})
    request = urllib.request.Request(api + '/v1/chat/completions', data=json.dumps(p).encode(),
                                     headers={'Content-Type': 'application/json'})
    content, reasoning, usage, server_metrics = [], [], {}, {}
    first = last = None
    finish = None
    energy_start = energy_snapshot()
    started = time.monotonic()
    energy_first = None
    with urllib.request.urlopen(request, timeout=1200) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                break
            event = json.loads(raw)
            if event.get('error'):
                raise RuntimeError(event['error'])
            usage = event.get('usage') or usage
            server_metrics = event.get('metrics') or server_metrics
            for choice in event.get('choices') or []:
                finish = choice.get('finish_reason') or finish
                delta = choice.get('delta') or {}
                if any(delta.get(k) for k in ('content', 'reasoning_content', 'reasoning')):
                    now = time.monotonic()
                    if first is None:
                        first = now
                        energy_first = energy_snapshot()
                    last = now
                content.append(delta.get('content') or '')
                reasoning.append(delta.get('reasoning_content') or delta.get('reasoning') or '')
    ended = time.monotonic()
    energy_end = energy_snapshot()
    assert first is not None and last is not None and usage and finish
    generated = usage['completion_tokens']
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
    assert cached == 0, 'Cold fixture unexpectedly reused a prefix'
    semantic = {k: v for k, v in payload.items() if k != 'cache_salt'}
    return {'input_sha256': hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest(),
            'output_sha256': hashlib.sha256((''.join(reasoning) + '\n' + ''.join(content)).encode()).hexdigest(),
            'usage': usage, 'finish_reason': finish, 'ttft_s': first - started,
            'total_s': ended - started, 'decode_s': last - first,
            'decode_tok_s': (generated - 1) / (last - first) if generated > 1 else None,
            'prefill_tok_s': usage['prompt_tokens'] / (first - started), 'metrics': server_metrics,
            'energy_total_j': (energy_end - energy_start) / 1e6 if energy_end is not None and energy_start is not None else None,
            'energy_decode_j': (energy_end - energy_first) / 1e6 if energy_end is not None and energy_first is not None else None,
            'host_available_gib_after': available_gib()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--api', default='http://127.0.0.1:19622')
    ap.add_argument('--model', default='qwen-3.8-27b')
    ap.add_argument('--namespace', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists(), 'Use a fresh result file'
    deck = {}
    for size in (512, 8192):
        p = a.root / f'results/frozen-gptq-mtp4-p{size}-g128/prompts.json'
        deck[size] = json.loads(p.read_text())['prompts']
    report = {'protocol': 'frozen-six-family-512-output-and-cold-8k-32k-v1',
              'power_polling': False, 'namespace': a.namespace, 'rows': []}
    cells = [('decode', index, 1, repeat) for repeat in range(2)
             for index in (list(range(6)) if repeat == 0 else list(reversed(range(6))))]
    cells += [('prefill', 0, mult, repeat) for mult in (1, 4) for repeat in range(3)]
    for kind, index, mult, repeat in cells:
        source = deck[512 if kind == 'decode' else 8192][index]
        messages = copy.deepcopy(source['messages'])
        if mult > 1:
            messages[-1]['content'] *= mult
        payload = {'model': a.model, 'messages': messages, 'temperature': 0, 'seed': 4242,
                   'max_tokens': 512 if kind == 'decode' else 8, 'ignore_eos': True,
                   'reasoning_effort': 'none',
                   'cache_salt': f'{a.namespace}-{kind}-{index}-{mult}-{repeat}'}
        row = measure(a.api, payload)
        assert row['usage']['completion_tokens'] == payload['max_tokens']
        row.update(kind=kind, family=source.get('family'), index=index, multiplier=mult, repeat=repeat)
        report['rows'].append(row)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + '\n')
        print(kind, index, mult, row['usage']['prompt_tokens'], round(row['prefill_tok_s'], 2),
              round(row['decode_tok_s'], 2), flush=True)
    report['summary'] = {'decode_median_tok_s': statistics.median(r['decode_tok_s'] for r in report['rows'] if r['kind'] == 'decode'),
                         'prefill_median_tok_s': {str(m): statistics.median(r['prefill_tok_s'] for r in report['rows'] if r['kind'] == 'prefill' and r['multiplier'] == m) for m in (1, 4)}}
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
