# References and pinned sources

## Hardware and runtime

- [Intel Arc Pro B70 specifications](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html) — 230 W TBP; configurable 160–290 W range.
- [Intel Xe KMD supported GPUs](https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html)
- [Intel Arc Resizable BAR requirements](https://www.intel.com/content/www/us/en/support/articles/000091128/graphics/intel-arc-dedicated-graphics-family.html)
- [vLLM Intel XPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
- [vLLM prefix-caching design](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [Intel LLM Scaler vLLM reference commands](https://github.com/intel/llm-scaler/blob/main/vllm/README.md/)
- [vLLM XPU Qwen3.5 optimization plan](https://github.com/vllm-project/vllm-xpu-kernels/issues/172)
- [XPU-kernels SYCL-TLA update](https://github.com/vllm-project/vllm-xpu-kernels/pull/517) — unreleased main change advancing the kernel template dependency.
- [SYCL-TLA Battlemage block-2D load fix](https://github.com/intel/sycl-tla/pull/846) — included by the unreleased dependency update, but it restores BMG behavior already present in the pinned older revision and is not a known `64*N+5` fix.
- [XPU-kernels main wheel used for the pending binary A/B](https://github.com/vllm-project/vllm-xpu-kernels/actions/runs/32692290527) — commit `baaa05bb4`, artifact `9508328924`; source parity and hashes are recorded in the source audit.
- [LM Studio runtime management](https://lmstudio.ai/docs/cli/runtime/runtime)
- [humble-b70-llm community reference](https://github.com/JP-devv/humble-b70-llm)

## Base-model artifacts

- [Frozenlock Qwen3.8-27B AutoRound INT4](https://huggingface.co/Frozenlock/Qwen3.8-27B-int4-AutoRound), pinned revision `b4c61732c4f2d8af323d75ba5702b5c7f3361539`.
- [SergiioB Qwen3.8-27B GPTQ INT4 with BF16 MTP](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16), pinned revision `e28c5f952bdd5d814297a07d85a064a87af26a3f`.
- [Unsloth Qwen3.8-27B GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), pinned local reference revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`.

Exact local paths and hashes are in [`architecture.md`](architecture.md) and [`benchmarks-and-quality.md`](benchmarks-and-quality.md).

## Abliterated candidates

- [Official Huihui BF16 source](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated), pinned inspected revision `739e3c5b89849f6c238ce1e5b70008612ae42cdd`.
- [Official Huihui GGUF reference](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF), pinned inspected revision `5a159a02d382ed01dd352e0fdb599f3c33bef0dd`.
- [letechlead AutoRound INT4 candidate](https://huggingface.co/letechlead/Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound), pinned inspected revision `2e8f02468010a96bae5d5a12b2ea472c17f780ad`.
- [TelperionAI AWQ/GPTQ candidate](https://huggingface.co/TelperionAI/Huihui-Qwen3.8-27B-abliterated-INT4-AWQ-GPTQ), an alternate compressed-tensors artifact not yet proven on the selected XPU path.

Recheck upstream revisions and model cards before downloading or quantizing. The recorded identifiers deliberately prevent a mutable repository name from being mistaken for reproducible provenance.
