# Selected architecture

## Host and accelerator

- Debian forky/sid; no Ubuntu repositories or host oneAPI stack.
- Intel Arc Pro B70, PCI ID `8086:e223`, 32 GiB VRAM.
- Upstream `xe` kernel driver on kernel `7.1.8+deb14.1-amd64`.
- B70 device nodes passed to the container: `/dev/dri/card0` and `/dev/dri/renderD128`.
- PCIe 5.0 x16 host link and a full 32 GiB Resizable BAR were observed during bring-up.

## Runtime

- Compose definition: `/home/julien/Documents/B70/docker/compose.yaml`.
- Container: `b70-vllm-qwen38-latest`.
- Image: local derivative `b70-vllm-latest-xpu:a397c58-head256-k64-prefill64-vision96`, ID `sha256:a5ab5fc08fe7a1755eedc01e45749190a8a53b4f41c0db1430f65df70d2b4a50`, based on `sha256:547811943b8b78a48a17ca17e1a16e8927ae98e982e444640f7e4035b02d7d68`.
- vLLM source: `46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3`.
- PyTorch: `2.13.0+xpu`.
- `vllm-xpu-kernels` source: `a397c58eb7781e6fe0d6b3fb7c25d21b5f658784`, with the local head-256 Q256/K64 language-attention policy, head-96 non-causal vision attention, and the contained oneDNN W4A16 prefill selector.
- XPU graphs enabled; one request sequence; MTP with four speculative tokens.
- `interactivity` performance mode captures exact graph sizes 1–10. This avoids
  padding MTP4's five-token verifier to the balanced profile's eight-row graph.
- FP8 KV cache with an explicit 8,500,000,000-byte reservation.
- Maximum model length: 196,608 tokens.
- Vision modules are loaded. Each request may contain at most one image; video
  inputs are disabled.
- SHA-256 automatic prefix caching enabled for multi-round agent and tool-call workloads, with a 64-token match unit inside the XPU hybrid cache's larger physical blocks. The power sweep retained cold/fixed controls so historical numbers remain comparable.
- Scheduler budget: 8,192 tokens. A 16,384-token 100K-prefill A/B regressed both TTFT and energy efficiency.
- The persistent Compose volume `b70-vllm-cache` retains compiled graphs.

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
3. `patch_draft_lmhead_int4.py` — `e299c8fd4f9a6ae9fe6f42ac3cfda451654d9fcd2b183fb1e6255b0680575951`.
4. `patch_draft_mtp_int4.py` — `2dde6d58863cfc1b871daa2f871ce719c9526dd39607d1f325217b6f20a5ff3d`.
5. `patch_reasoning_effort_contract.py` — `7f1e1d1cc3f66ec2a5f67890c3ef683ad1fc41b61689e8b81a1305d37a9da858`.
6. `patch_gdn_prompt_padding.py` — `0188d68fdad6ebbf8946bd330617c07d1348c21c789b1fa37bce8a0ec6a01eff`.
7. `patch_uniform_decode_prefill.py` — `1c9d3db9ef1a9abfdd00dd488bc45bda692174589ed62c94ad95f0b6eccec951`.

The fifth patch narrows vLLM's generic reasoning-effort request types to this
single model's `none`, `low`, `medium` and `xhigh` contract across chat, batch
chat, render and Responses routes. It also narrows the Anthropic-compatible
Messages route to its supported enabled profiles (`low`, `medium`, `xhigh`).
FastAPI therefore advertises only effort values that the Qwen template accepts.
The sixth remains mounted but is disabled. The seventh fixes vLLM's legacy
graph classifier by forbidding uniform-decode dispatch whenever scheduler state
identifies prefill; it replaces the earlier
prompt-padding containment without changing prompts. The local image contains
the validated Q256/K64 Xe2 language-attention pair, the head-96 non-causal
vision-attention pair, and the matched `_xpu_C` and GDN companion carrying the
contained W4A16 prefill selector. Revalidate every runtime patch and rebuild
both kernel pairs before changing either source
revision.

## Network and lifecycle

- Host endpoint: `0.0.0.0:19622`; container endpoint: `0.0.0.0:8000`.
  The unauthenticated API is reachable through every host IPv4 interface, so
  the host firewall must restrict it to trusted networks.
- Served name: `qwen-3.8-27b`.
- Automatic tool choice enabled with vLLM's `qwen3_coder` tool-call parser.
- `restart: unless-stopped`; Docker starts inference after a host reboot unless
  the service was explicitly stopped.
- Container capabilities are dropped and `no-new-privileges` is enabled.
- Model directories are read-only inside the container.

## Power policy

- Persistent cap: 210 W, selected by the sweep in [`power-efficiency.md`](power-efficiency.md).
- Source unit: `/home/julien/Documents/B70/systemd/b70-power-limit.service`.
- Installed unit: `/etc/systemd/system/b70-power-limit.service`.
- Installed helper: `/usr/local/sbin/b70-power-limit`, root-owned mode `0755`.
- The helper verifies PCI `8086:e223`, refuses values outside Intel's documented 160–290 W range, and verifies the applied Xe hwmon value.
- Stopping the unit restores the board's observed pre-sweep 275 W cap.

LAN exposure is enabled on every host IPv4 interface. Automatic restart is
enabled and must remain paired with trusted-network firewall restrictions.
