# Runtime update qualification — 2026-11-08

Baseline: selected tuned XPU image, Frozenlock GPTQ target, draft-only INT4,
MTP4, FP8 KV, 262,144 server tokens, 210 W. No driver change.

The harness uses the same frozen six-family deck in forward/reverse order,
12 requests with 512 forced output tokens, and three cold 8K/32K prefills.
Counters are read at request boundaries; there is no periodic power polling.
These fixed-output speed cells do not establish answer quality.

| Candidate | Decode tok/s | 8K prefill tok/s | 32K prefill tok/s |
|---|---:|---:|---:|
| Baseline | 83.49 | 1,672.29 | 1,429.46 |
| Official vLLM 0.32 XPU, MTP4 | 84.32 (+1.0%) | 1,609.46 (-3.8%) | 1,322.15 (-7.5%) |
| Tuned baseline image, MTP5 | 82.85 (-0.8%) | 1,665.35 (-0.4%) | 1,427.59 (-0.1%) |
| Tuned baseline image, MTP3 | 79.23 (-5.1%) | 1,706.78 (+2.1%) | 1,409.40 (-1.4%) |

The stock upgrade is rejected: it loses the deployment's tuned prefill kernels
and does not meet the 3% repeatable end-to-end promotion threshold.
It passed the 143 bounded answer/API/neighboring-length checks.
Greedy outputs are not all stable across repeats, even on the baseline;
headline decode medians must not be interpreted as pure kernel attribution.

MTP5 needed 10.5 GB KV allocation to preserve 262,144 tokens (10.3 GB was
rejected by the capacity check). MTP3 used the baseline 10.3 GB allocation.
Both draft-depth candidates are rejected. Restoring MTP4 measured 82.22 tok/s
decode, 1,665.34 tok/s at 8K and 1,428.49 tok/s at 32K; it passed 143 checks.
The selected MTP4 image and Compose
profile remain unchanged. No driver upgrade is needed for the tested runtime.

The 0.32-compatible patch anchors retain the old-runtime branches and fail
closed on unknown source. The disabled prompt-padding patch was omitted from
the 0.32 trial; prompt padding remained off. These changes prepare a future
trial and are not evidence that the 0.32 candidate is deployed.


The newer compatible XPU kernel wheel 0.1.16.1 was also tested in an isolated
vLLM 0.32 image (Torch 2.14 unchanged). It failed during vision initialization:
its precompiled attention set lacks `80,false,false,false,false,false`.
The generic fallback attempted a 128 GiB GPU allocation and cleanly raised
XPU OOM. No device fault/reset occurred; the original service was restored.
No throughput figure exists for this failed start. Reducing image capability
or context to make the candidate start was not accepted.

Wheel: `vllm_xpu_kernels-0.1.16.1-cp38-abi3-manylinux_2_28_x86_64.whl`,
SHA-256 `20da3b3953d9617afcd2de191c64e869c2d85dc706ae6372cc9873789a5951ea`.
The official image digest was
`sha256:20509fabe50dda62c09c0786d1a6effa299d3dbdddcc49331b93e65144652a8c`.
It contained vLLM 0.32.0+xpu, Torch 2.14.0+xpu, XPU kernels 0.1.15.4,
and Triton XPU 3.8.0. The retained image is
`sha256:a5ab5fc08fe7a1755eedc01e45749190a8a53b4f41c0db1430f65df70d2b4a50`.

These are bounded update trials, not an exhaustive search of custom rebuilds.
No production performance improvement is claimed and no blog update is needed.
