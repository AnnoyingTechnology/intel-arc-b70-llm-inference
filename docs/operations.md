# Operations

Run all Compose commands from `/home/julien/Documents/B70/docker` so relative patch mounts resolve correctly.

## Start and verify

```bash
cd /home/julien/Documents/B70/docker
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:19622/health
curl -fsS http://127.0.0.1:19622/v1/models | jq '.data[] | {id, root, max_model_len}'
```

Expected model ID: `qwen38`. Expected `max_model_len`: `196608`. Initial compilation can take roughly two minutes after the compile cache is empty.

Follow startup or failure logs:

```bash
cd /home/julien/Documents/B70/docker
docker compose logs -f --tail=200
```

Confirm the memory contract in the logs:

```bash
docker logs b70-vllm-qwen38 2>&1 \
  | rg 'max_seq_len|GPU KV cache size|Maximum concurrency|init engine'
```

## Request example

Non-thinking request:

```bash
curl -fsS http://127.0.0.1:19622/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen38",
    "messages":[{"role":"user","content":"Reply with exactly OK"}],
    "temperature":0,
    "max_tokens":8,
    "chat_template_kwargs":{"enable_thinking":false}
  }' | jq .
```

Without `chat_template_kwargs.enable_thinking=false`, the Qwen reasoning parser can return reasoning separately in compatible clients.

OpenCode selector: `local-b70/qwen38`. The provider points directly to the loopback API and is not the global default. Its default reasoning effort is `low`; selectable variants are `off`, `low`, `medium` and `xhigh`, matching the other Qwen3.8 entry. OpenCode sends `off` literally while vLLM expects `none`, so the pinned startup patch normalizes that alias before chat-template rendering.

Automatic tool selection is enabled server-side with vLLM's `qwen3_coder` parser. The parser matches this model's XML `<tool_call><function=...>` chat-template contract.

Automatic prefix caching is enabled with SHA-256 keys and a 64-token prefix-match unit. Repeated conversations and tool-call follow-ups can reuse cached context inside the XPU hybrid cache's larger physical blocks; cold, unrelated prompts still pay normal prefill cost. Cache entries are in-memory and disappear when the service restarts.

## Power cap

The selected 210 W cap is independent of the container and persists at boot:

```bash
systemctl is-enabled b70-power-limit.service
systemctl is-active b70-power-limit.service
cat /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*/power1_cap
```

Expected value: `210000000` microwatts. Full method and evidence: [`power-efficiency.md`](power-efficiency.md).

Rollback to the observed pre-sweep 275 W state:

```bash
sudo systemctl disable --now b70-power-limit.service
sudo /usr/local/sbin/b70-power-limit 275
```

Re-enable 210 W with `sudo systemctl enable --now b70-power-limit.service`.

## Stop or recreate

Stop and remove only the B70 service:

```bash
cd /home/julien/Documents/B70/docker
docker compose down
```

Apply a Compose change:

```bash
cd /home/julien/Documents/B70/docker
docker compose config --quiet
docker compose up -d --force-recreate
docker compose ps
curl -fsS http://127.0.0.1:19622/health
```

The persistent compile-cache volume is retained by `docker compose down`. Do not add `--volumes` unless deliberately discarding it.

## Reboot behavior

The 210 W power cap is applied automatically after reboot. `restart: "no"` for inference is deliberate: start the model with `docker compose up -d`. This avoids silently reserving the B70 or exposing a future LAN endpoint.

Do not run LM Studio inference concurrently. LM Studio and vLLM compete for the same 32 GiB device and can produce misleading failures or performance results. Merely running `lms runtime ls` can start LM Studio's background service.

## Rollback of the draft-only speed overlay

The measured BF16-draft rollback profile keeps MTP4 and the same verifier target but disables the two draft INT4 features. Before editing, preserve the current Compose file or use a version-control diff. Then change:

```yaml
B70_DRAFT_LMHEAD_INT4: "0"
B70_DRAFT_MTP_INT4: "0"
```

The patch hooks are environment-gated, so the patch invocations and mounts can remain. Recreate the service and rerun the quality gate. The measured rollback medians were 82.98 tok/s at p512/g128 and 81.98 tok/s at p8192/g128.

Rollback from a failed Compose edit is to restore the previous file, validate with `docker compose config --quiet`, force-recreate, and check `/health` plus the canaries.

## LAN and always-loaded mode

Not enabled. A future promotion needs all of the following as one reviewed change:

- A candidate that passes [`huihui-plan.md`](huihui-plan.md).
- An explicit bind address, preferably the B70 host's LAN address rather than `0.0.0.0`.
- Host firewall scoping to trusted local subnets.
- Authentication or a trusted authenticated reverse proxy; the vLLM API is otherwise unauthenticated.
- `restart: unless-stopped` only after the model and exposure policy are final.
- Validation from an authorized LAN client and after one reboot.
