# Huihui abliterated model plan

## Decision rule

The desirable final state is one model, always loaded and available to trusted LAN clients. It may be the Huihui abliterated/uncensored variant only if paired tests show no material capability, long-context or performance regression against the current base service.

No Huihui artifact is currently promoted, downloaded by this work, exposed to the LAN or configured for automatic restart.

## Canonical candidate

Use the official [`huihui-ai/Huihui-Qwen3.8-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) as the source of truth:

- Pinned revision inspected on 2026-08-24: `739e3c5b89849f6c238ce1e5b70008612ae42cdd`.
- Based on `Qwen/Qwen3.8-27B` and Apache-2.0 licensed.
- The current card says only layers 18–51 are ablated; other layers are unchanged.
- MTP and visual components were not modified.
- The publisher warns that the model can produce sensitive or inappropriate output and is unsuitable for public, underage or high-security use.

The official [`Huihui GGUF repository`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF), revision `5a159a02d382ed01dd352e0fdb599f3c33bef0dd`, is the preferred independent quality-reference track. It is not expected to be the fastest vLLM/XPU serving artifact.

## Existing INT4 candidate and pitfall

[`letechlead/Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound`](https://huggingface.co/letechlead/Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound), revision `2e8f02468010a96bae5d5a12b2ea472c17f780ad`, is technically closest to the selected B70 format:

- AutoRound 0.14.2; INT4 W4A16; symmetric; group size 128.
- Packing is documented as `auto_round:auto_gptq`.
- GDN `in_proj_a` and `in_proj_b` are retained at 16-bit.
- It includes an MTP export, but most of its MTP path is quantized rather than kept as the quality-controlled BF16 sidecar used by the current service.

It predates the official Huihui revision above by four days. Therefore it may not contain the current layer-18–51-only ablation. Do not promote it based on name, format or speed.

The clean path is to pin the official current source and either:

1. Quantize it locally to the same AutoRound W4A16 symmetric G128, GPTQ-compatible target format while excluding the GDN input projections and retaining MTP in BF16; or
2. Use a third-party artifact only after proving its exact source revision and quantization recipe match that contract.

## Promotion gate

Run base and candidate with the same pinned vLLM image, context, prompt corpus, seeds, sampling, output budgets and cache state. Keep raw outputs and counters under `results/huihui-ab/`.

Required gates:

- Artifact provenance: source revision, file hashes and recipe recorded; no opaque INT4 accepted.
- Existing corruption gate: 7/7 exact canaries, eight-repeat stability, JSON/tool formatting and code execution.
- Quality reference: compare both candidate XPU and official Huihui GGUF against the same prompts; compare the base XPU service against its Unsloth GGUF reference.
- Capability: version-pinned reasoning, coding, factual, instruction-following, structured-output and multilingual sets.
- Long context: retrieval and reasoning at short, 32K, 100K and the 196,608-token boundary.
- Performance: paired p512/p8192 × g128/g256 cells, full-boundary run, TTFT, decode rate and MTP acceptance.
- Resource stability: cold startup, VRAM headroom, repeated runs, container recreation and reboot.
- Refusal behavior: verify the intended reduction in refusals using a private, lawful prompt set; also measure ordinary-task regressions and pathological output.

Promotion requires no material regression. For deterministic tests, require exact pass parity. For benchmark sets, predefine acceptable statistical tolerance before looking at results; do not excuse a regression merely because the model is uncensored. Performance must be at least the selected base within run-to-run variance.

## Final deployment, only after promotion

If the Huihui candidate passes, replace the base rather than keeping both resident. Then:

- Give it a stable served name and update `local-b70` once.
- Bind only to the intended LAN address.
- Restrict access to trusted subnets and add authentication or an authenticated reverse proxy.
- Change the restart policy to `unless-stopped` and validate after reboot.
- Retain the base model and its Compose revision on disk as rollback until the new service has an agreed soak period.

An uncensored endpoint must not be anonymously reachable. The absence of model refusals transfers policy enforcement to network access, clients and users.
