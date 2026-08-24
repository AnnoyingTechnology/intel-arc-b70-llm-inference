# Selected architecture

## Host and accelerator

- Debian forky/sid; no Ubuntu repositories or host oneAPI stack.
- Intel Arc Pro B70, PCI ID `8086:e223`, 32 GiB VRAM.
- Upstream `xe` kernel driver on kernel `7.1.8+deb14.1-amd64`.
- B70 device nodes passed to the container: `/dev/dri/card0` and `/dev/dri/renderD128`.
- PCIe 5.0 x16 host link and a full 32 GiB Resizable BAR were observed during bring-up.

## Runtime

- Compose definition: `/home/julien/Documents/B70/docker/compose.yaml`.
- Container: `b70-vllm-qwen38`.
- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- vLLM: `0.27.2rc1.dev77+gac7509e2b`.
- PyTorch: `2.13.0+xpu`.
- `vllm-xpu-kernels`: `0.1.12.3`.
- XPU graphs enabled; one request sequence; MTP with four speculative tokens.
- FP8 KV cache with an explicit 8,500,000,000-byte reservation.
- Maximum model length: 196,608 tokens.
- SHA-256 automatic prefix caching enabled for multi-round agent and tool-call workloads, with a 64-token match unit inside the XPU hybrid cache's larger physical blocks. The power sweep retained cold/fixed controls so historical numbers remain comparable.
- Scheduler budget: 8,192 tokens. A 16,384-token 100K-prefill A/B regressed both TTFT and energy efficiency.
- The persistent Compose volume `b70-vllm_vllm-cache` retains compiled graphs.

With prefix caching and aligned hybrid-cache pages, the engine reports 209,523 tokens of KV capacity and 1.07x maximum concurrency at the configured 196,608-token contract. The earlier 214,214-token figure and exact 196,480+128 boundary completion were measured before prefix caching changed the cache layout. At the user's direction, that five-minute mechanical request was not repeated; current capacity headroom is observed, while exact-boundary completion under the final cache mode remains unverified. A cold-cache recreation completed its two main compilation phases in about 58 and 14 seconds.

## Target model

The served model is an overlay at:

`/srv/models/vllm/Frozenlock--Qwen3.8-27B-GPTQ-MTP-BF16`

It combines:

1. Target weights from `Frozenlock/Qwen3.8-27B-int4-AutoRound`, revision `b4c61732c4f2d8af323d75ba5702b5c7f3361539`.
2. The checkpoint's documented `auto_round:auto_gptq` packing, declared as GPTQ to select Intel's working XPU W4A16 verification path.
3. Fifteen BF16 `mtp.*` tensors extracted from the same official-base `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, revision `e28c5f952bdd5d814297a07d85a064a87af26a3f`.
4. Runtime INT4 G128 symmetric copies of only the draft LM head and the five draft MTP linears.

The original target verifier, including its FP16 target LM head, is unchanged. Draft quantization may change proposal acceptance, but every proposed token is checked by the target before emission.

Generated overlay hashes:

- BF16 MTP sidecar: `814dee3d5904e82ae37c9c7ef4c7c1bec44fe2f5ea6184ce6301e723a341f068`.
- Merged index: `08195c40d1c7d622e51a41c7fef9b89a7603ebabd828902b014fc618f888185a`.
- `config.json`: `0c397dd7c79d2e7ccd29068085763f5dc280debd26c85c7065da817218e565ed`.
- `quantize_config.json`: `a1087a8ea34f3d17a1a0a2154243f85263d786d957521a8fec3b2d73506fac39`.

## Runtime patches

Applied in this order from `/home/julien/Documents/B70/docker/patches`:

1. `patch_mtp_nightly.py` — `4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14`.
2. `patch_mtp_boundary.py` — `41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50`.
3. `patch_draft_lmhead_int4.py` — `ffae41926d5f05f4f38bb985301b5e572092441d06d6063c8820a63a39b8cefc`.
4. `patch_draft_mtp_int4.py` — `4df179c3e77fd7a248f9b9c0b60217c60caea14ebfd16b7860536fbff3b2a1e9`.
5. `patch_reasoning_effort_off.py` — `2ac1af03723e80eab2a05cb5a4ea8ebfc1021082607d43ffc54071da008e0641`.

The fifth patch is a frontend compatibility shim: OpenCode 1.18.4 sends its visible `off` reasoning variant literally, while vLLM accepts `none`. It admits only that alias and normalizes it to `none`. The image is digest-pinned because all patches depend on exact source anchors. Revalidate and update every patch before changing the image.

## Network and lifecycle

- Host endpoint: `127.0.0.1:19622`; container endpoint: `0.0.0.0:8000`.
- Served name: `qwen38`.
- Automatic tool choice enabled with vLLM's `qwen3_coder` tool-call parser.
- `restart: "no"`; it does not reserve the GPU after a host reboot until explicitly started.
- Container capabilities are dropped and `no-new-privileges` is enabled.
- Model directories are read-only inside the container.

## Power policy

- Persistent cap: 210 W, selected by the sweep in [`power-efficiency.md`](power-efficiency.md).
- Source unit: `/home/julien/Documents/B70/systemd/b70-power-limit.service`.
- Installed unit: `/etc/systemd/system/b70-power-limit.service`.
- Installed helper: `/usr/local/sbin/b70-power-limit`, root-owned mode `0755`.
- The helper verifies PCI `8086:e223`, refuses values outside Intel's documented 160–290 W range, and verifies the applied Xe hwmon value.
- Stopping the unit restores the board's observed pre-sweep 275 W cap.

LAN exposure and automatic restart are intentionally deferred until an abliterated candidate passes the promotion gate and access controls are decided.
