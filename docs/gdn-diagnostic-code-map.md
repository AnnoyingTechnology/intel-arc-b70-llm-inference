# GDN diagnostic code map

This map prevents experimental instrumentation from being mistaken for a fix
or for trusted evidence. None of these patches is mounted by the stock service.
The authoritative conclusions are in
[`gdn-64n5-investigation-pause-2026-08-25.md`](gdn-64n5-investigation-pause-2026-08-25.md).

## Trusted evidence producers

| File | Purpose |
|---|---|
| `docker/patches/patch_model_stage_capture.py` | Persistent model-stage boundary capture used to establish that earlier layers remain finite. |
| `docker/patches/patch_gdn_stage_capture.py` | Length-keyed persistent GDN stage bank with active/physical-row analysis. |
| `docker/patches/patch_layer4_projection_decisive.py` | Final producer-side layer-4 input, projection, `z`, and `core_attn_out` capture with separate mutation contract and markers. |
| `scripts/analyze_model_stage_capture.py` | Value-free model-stage summaries and hashes. |
| `scripts/analyze_gdn_stage_capture.py` | Value-free GDN-stage summaries and hashes. |
| `scripts/analyze_decisive_gdn.py` | Finiteness analysis for the final layer-4 capture. |

## Replay and negative controls

| File | Purpose |
|---|---|
| `scripts/replay_layer4_projection.py` | Replays exact layer-4 projection inputs in eager and XPU-graph modes. |
| `scripts/replay_layer4_causal_conv.py` | Replays exact projections through fused/split GDN with physical/padded and compact/live state layouts. |
| `scripts/replay_gdn_stage_capture.py` | Replays earlier full GDN-stage captures. |
| `scripts/replay_gdn_multilayer_graph.py` | Checks graph/lifetime behavior across multiple GDN layers. |
| `docker/patches/patch_gdn_entry_sync_diagnostic.py` | Pre-GDN synchronization A/B; did not cure n5. |
| `docker/patches/patch_gdn_split_probe.py` | Split-path and retained-intermediate A/B; did not cure n5. |
| `docker/Dockerfile.xpu-kernels-build` | Reproducible local XPU-kernel build environment used for source candidates. |

## Superseded or observer-invalid probes

The following files are retained only as an experiment log. Results obtained
from their debug-copy outputs are excluded from the upstream evidence. Do not
deploy them or infer the first failing internal operator from them.

| File | Reason |
|---|---|
| `docker/patches/patch_gdn_capture.py` | Early broad capture; shared buffers and observer ordering were not sufficiently constrained. |
| `docker/patches/patch_flash_attn_capture.py` | Earlier cross-layer survey, superseded by the producer-side model-stage capture. |
| `docker/patches/patch_layer4_projection_capture.py` | Superseded by the length-keyed decisive projection capture. |
| `docker/patches/patch_layer4_gdn_entry_decisive.py` | In-outer-op copy path contradicted the trusted producer-side capture. |
| `docker/patches/patch_layer4_gdn_inputs_decisive.py` | C++ optional-debug-output path proved observer-invalid. |
| `docker/patches/patch_layer4_gdn_inop_persistent.py` | Finite sentinels and synchronization did not establish valid producer ordering. |
| `docker/patches/patch_layer4_gdn_explicit_capture.py` | Expanding the outer custom op mutation set perturbed compiled memory planning. |
| `docker/patches/patch_layer4_gdn_pointer_probe.py` | Metadata probe only; synchronization/printing makes it unsuitable as numerical evidence. |
| `scripts/analyze_flash_capture.py` | Analyzer for the superseded FlashAttention survey. |
| `scripts/analyze_gdn_capture.py` | Analyzer for the early broad capture. |

`docker/patches/patch_gdn_stage_capture.py` contains earlier causal-stage fields
whose debug-copy values are not independently trusted. Its persistent
producer-side `core_pre`, `core_output`, `z_output`, model inputs, metadata,
and recurrent-state layout remain useful. The authoritative handoff uses only
the validated subset.

## Paused source build

The ignored `.build/` tree contains an incomplete experiment that changes the
five q/k/v/b/a diagnostic copies from ATen `copy_` to
`vllmGetQueue().memcpy`. It changes no inference arithmetic. The build was
stopped when the reduced build unexpectedly reconfigured oneDNN. It is not a
result, is not deployed, and should be resumed only if maintainers consider the
capture useful.
