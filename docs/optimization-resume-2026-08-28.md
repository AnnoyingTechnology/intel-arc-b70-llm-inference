# Optimization resume checkpoint — 2026-08-28

## Immediate incident

The wheel build launched on 2026-08-27 used `MAX_JOBS=8` without a Docker
memory limit. Several oneAPI compiler processes each consumed roughly 6 GiB.
At 18:13 CEST the host OOM killer terminated Chromium, Spotify, Mattermost,
the user D-Bus, and finally the GNOME session. The kernel and Docker daemon did
not reboot; the builder survived until it was stopped on 2026-08-28.

Stopping container `quirky_napier` reduced host memory use from approximately
47 GiB to 7 GiB. Its bind-mounted source and object tree remains under
`/tmp/b70-xpu-autotune`.

## Durable resource guardrail

[`../AGENTS.md`](../AGENTS.md) now requires B70 compiler containers to use:

- at most 26 GiB RAM;
- no container swap (`--memory-swap=26g`);
- at most four build jobs;
- only the targets required by the experiment;
- at least 12 GiB host memory reserved before starting work.

[`../scripts/run_bounded_xpu_build.sh`](../scripts/run_bounded_xpu_build.sh)
implements those limits.

## Completed optimization

The investigation completed successfully. The final selector uses the existing
32x64 strategy only for four exact Qwen3.8 W4A16 projection shapes at 1,024 or
more tokens. oneDNN exposes the API token dimension as internal `N`, not `M`;
the corrected predicate and source patch are recorded in
[`xpu-w4a16-prefill64-tuning-2026-08-28.md`](xpu-w4a16-prefill64-tuning-2026-08-28.md).

The weighted exact-shape microbenchmark improved 6.14%. Three-run cold service
medians improved from 5.2160 to 4.9195 seconds at 8K and from 24.6192 to
22.9631 seconds at 32K. Correctness, energy, neighboring-shape, and GDN gates
all passed.

The first integration image omitted GDN from `_xpu_C` and failed startup with a
missing `gdn_attention` registration. It was never benchmarked or promoted.
The corrected focused build enables both `GDN_KERNELS_ENABLED` and
`XPU_SPECIFIC_KERNELS_ENABLED`; its matched `_xpu_C` and GDN companion are
packaged together.

## Final service state

Compose now runs image
`b70-vllm-latest-xpu:a397c58-head256-k64-prefill64`, ID
`sha256:547811943b8b78a48a17ca17e1a16e8927ae98e982e444640f7e4035b02d7d68`.
The service is healthy with the established 196,608-token, graph-enabled,
FP8-KV, MTP4 configuration and no `/tmp` bind mounts. The B70 power cap was
restored to 210 W after the crash had reset it to zero.
