# Qwen3.8 vision enablement — 2026-09-01

## Outcome

Qwen3.8-27B now accepts one image per OpenAI-compatible chat request while
retaining the complete 196,608-token serving contract, the exact
8,500,000,000-byte FP8 KV reservation, one sequence, MTP4 and every existing
runtime patch. Video remains disabled.

The healthy engine reports 209,523 KV-cache tokens and 1.0656934306569343x
maximum concurrency at 196,608 tokens. Loading the vision-capable model used
17.66 GiB; the encoder cache was initialized with a 16,384-token budget. The
first cold vision startup spent 93.87 seconds compiling and 101.98 seconds in
engine initialization; graph capture used 0.24 GiB.

## Root cause

The model overlay already contained 333 BF16 vision tensors: 460,730,096
parameters and 921,460,192 payload bytes. They were not loaded because Compose
passed `--language-model-only`.

Removing that flag exposed a second, independent issue. The locally optimized
Xe2 FlashAttention library had deliberately been built with only the language
model's head-256 prefill tuple and its decode companion. Vision initialization
requested the upstream-supported non-causal tuple:

```text
96,false,false,false,false,false
```

The missing tuple made the wrapper fall back to a 128 GiB allocation and fail;
this was not evidence that the configured context or vision projector did not
fit. Rebuilding the focused attention library with that one additional tuple
removed the failure.

## Exact build

- `vllm-xpu-kernels`: `a397c58eb7781e6fe0d6b3fb7c25d21b5f658784`
- oneAPI compiler: 2026.0.0
- runtime-matched `ocloc`: `26.27.39122.11`
- runtime-matched IGC: `2.38.2`
- PyTorch: `2.13.0+xpu`
- build families: FA2 only; all basic, MoE, GDN, MHC, MQA-logits,
  XPU-specific and allocator targets disabled
- generated sources: four chunk-prefill objects and one paged-decode object
- compiler container: 26 GiB RAM, no swap, eight CPUs, three jobs

Four compiler jobs exceeded the 26 GiB cgroup limit. Three jobs peaked at
19.91 GiB and completed while the host retained more than the required 12 GiB
reserve. The pinned `setup.py` did not forward `MHC_KERNELS_ENABLED`; the
temporary build tree added that option to its forwarded CMake list so MHC
remained off.

The exact retained configs are
[`qwen38-vision-chunk-prefill.conf`](../docker/kernel-configs/qwen38-vision-chunk-prefill.conf)
and [`qwen38-paged-decode.conf`](../docker/kernel-configs/qwen38-paged-decode.conf).
The existing
[`vllm-xpu-kernels-xe2-head256-k64.patch`](../docker/patches/vllm-xpu-kernels-xe2-head256-k64.patch)
was applied unchanged.

Built artifacts:

```text
685eab45ed06aa9fd519e883eeffb847c6a83acfa7196e32e8d44fb79cb54e73  _vllm_fa2_C.abi3.so
30145d4984926a7ed395cf916bbb7aa10e63f4c4e37e9384f3b8012929ba6d1c  libattn_kernels_xe_2.so
e7bdd1c5d06e20948d84428343d9899b164271a169e59ee490faf9a88c92fc52  vllm_xpu_kernels-0.1.14.dev16+ga397c58.vision1-cp38-abi3-linux_x86_64.whl
```

The two attention libraries were overlaid on the prior production image. The
result is
`b70-vllm-latest-xpu:a397c58-head256-k64-prefill64-vision96`, image ID
`sha256:a5ab5fc08fe7a1755eedc01e45749190a8a53b4f41c0db1430f65df70d2b4a50`.

## Runtime validation

Ten sequential local files completed through `/v1/chat/completions`: two JPEG,
two PNG, two WebP, one GIF, one AVIF and one TIFF. Dimensions ranged from
432x346 to 4048x2961, file sizes from 27,150 to 4,315,296 bytes, and image-token
counts from 154 to 12,740. Every response returned a description, keywords and
raw OCR text with `reasoning_tokens=0`. A separate text-only canary returned
exactly `TEXT_OK` with zero reasoning tokens.

After the run the container was healthy, had zero restarts, was not OOM-killed,
and reported no request-time errors. No output-quality gate was run because no
weights, logits, precision, sampler or generation policy changed.

## Rollback

Restore image `b70-vllm-latest-xpu:a397c58-head256-k64-prefill64` and replace
