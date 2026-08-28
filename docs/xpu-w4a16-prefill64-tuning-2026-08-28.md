# Xe2 W4A16 prefill selector tuning — 2026-08-28

## Decision

Promoted. For four exact Qwen3.8 large-prefill projections, the contained
oneDNN selector uses an existing 32x64 XeHPC strategy instead of the Xe2
catalog's 32x32 strategy. The change starts at 1,024 tokens and preserves the
catalog selection for decode, short tails, output heads, and unrelated shapes.

## Observations

oneDNN 3.13.0 at `0e2a5bfeef1bfbffc3137464606540233086ce9b`
exposes these matmuls internally in column-major order: `(M, N, K)` means
`(output features, tokens, input features)`. The selector therefore matches:

```text
Xe2, N >= 1024, A=u4, B=f16, C=f16
K=5120 and M in {34816, 16384, 14336}
or K=6144 and M=5120
```

Verbose generation confirmed 32x64 for a 4,992-token target and the stock
32x32 recipe at 512 tokens. The five-token path retained its stock 16x8 recipe.

Two alternating 100-iteration exact-shape runs measured:

| Weighted modeled projection cycle | Time |
|---|---:|
| Stock mean | 2,021.37 ms |
| Candidate mean | 1,897.34 ms |
| Reduction | 6.14% |

The largest projection improved at every redirected neighboring token count:

| Tokens | Stock | Candidate | Throughput gain |
|---:|---:|---:|---:|
| 1,024 | 2.4930 ms | 2.3879 ms | 4.40% |
| 2,048 | 5.5525 ms | 5.0774 ms | 9.36% |
| 3,136 | 9.0306 ms | 8.6221 ms | 4.74% |
| 4,992 | 14.6422 ms | 13.4428 ms | 8.92% |
| 6,656 | 19.8681 ms | 17.6677 ms | 12.45% |
| 8,192 | 24.5138 ms | 21.7075 ms | 12.93% |

At 512 tokens both libraries emitted the same stock 32x32 kernel; its apparent
single-run timing difference was order noise and is not attributed to the
candidate.

## End-to-end and energy result

All service cells used the same model, prompt fixture, MTP4, FP8 KV, 8,192-token
scheduler, unique cold prefixes, zero cached tokens, and the 210 W cap. Each
value is a three-run median.

| Cell | Stock TTFT | Candidate TTFT | TTFT reduction | Prompt rate gain | Energy reduction |
|---|---:|---:|---:|---:|---:|
| 8K | 5.2160 s | 4.9195 s | 5.68% | 6.04% | 5.78% |
| 32K | 24.6192 s | 22.9631 s | 6.73% | 7.22% | 6.73% |

Raw records are retained in `.artifacts/w4a16-prefill64-{stock,candidate}-{8k,32k}.json`.

## Correctness and neighboring gates

- SHA-256 digests of the complete FP16 output tensors matched stock bit for bit
  for all six exact production shapes at 4,992 tokens.
- All outputs were finite.
- Seven deterministic canaries passed and matched the accepted baseline hashes.
- Eight repeated deterministic outputs produced one hash.
- The full 131-request GDN sweep passed lengths 1–128, 133, 197, and 261 with
  finite logprobs and no unexpected lengths.
- MTP acceptance for the service A/B fixture was unchanged.

## Interpretation

The measured fact is that 32x64 is faster for these exact large prefill shapes.
Better tile reuse or occupancy is a plausible explanation, not a measured root
cause. No broader Xe2 dispatch claim is made; the predicate deliberately avoids
generalizing beyond tested shapes and neighboring token counts.

## Build safety and integration pitfall

All corrected compiler runs used a 26 GiB Docker memory limit, no container
swap, four jobs, and focused targets. Host memory remained well above the
12 GiB reserve. The earlier uncapped eight-job build caused the desktop OOM
incident documented in
[`optimization-resume-2026-08-28.md`](optimization-resume-2026-08-28.md).

Replacing `_xpu_C` also replaces its operator registry. The first focused
integration build enabled W4A16 but disabled GDN, so service graph capture
failed on the missing `gdn_attention` operator. The accepted build enables both
`XPU_SPECIFIC_KERNELS_ENABLED=ON` and `GDN_KERNELS_ENABLED=ON` and packages
`_xpu_C` with its matched GDN shared library.

## Reproduction and deployment

Apply
[`onednn-qwen38-w4a16-prefill64.patch`](../docker/patches/onednn-qwen38-w4a16-prefill64.patch)
to the oneDNN source fetched by `vllm-xpu-kernels` commit
`a397c58eb7781e6fe0d6b3fb7c25d21b5f658784`, then use
[`run_bounded_xpu_build.sh`](../scripts/run_bounded_xpu_build.sh).

Validated artifacts:

```text
0ef0a2aa5a9a2ecc5b6ec51adf99fe2349f606030a739568d58cb0f7dc70c93c  _xpu_C.abi3.so
3c6a9931f7373abb3cda6bcea5cc13f984580bd5f424edeceeb8e16a35c2a80e  libgdn_attn_kernels_xe_2.so
561ec4416398b2d2344d5840778a724531ca4410b2de17ab8f95eacb42cc50ef  vllm_xpu_kernels-0.1.14.dev16+ga397c58.prefill64-cp38-abi3-linux_x86_64.whl
```

The promoted image is
`b70-vllm-latest-xpu:a397c58-head256-k64-prefill64`, ID
`sha256:547811943b8b78a48a17ca17e1a16e8927ae98e982e444640f7e4035b02d7d68`.
