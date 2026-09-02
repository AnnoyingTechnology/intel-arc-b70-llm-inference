# B70 Agent Instructions

## Safety and scope

- This is an experimental workstation, but preserve the interactive desktop and
  the established inference service unless the user explicitly authorizes an
  outage.
- Before a compile, benchmark, or profiler run, estimate its CPU, RAM, swap,
  storage, and GPU impact. Keep resource use proportionate to the experiment.
- Never run an uncapped compiler container. Use a Docker memory cgroup limit so
  a failed experiment is killed inside its container instead of invoking the
  host OOM killer.
- On this 48 GB host, compiler containers are limited to 26 GiB RAM with no
  swap allowance, at most four build jobs, and only the targets required by the
  experiment. Raise any limit only with explicit user approval.
- Do not run a build alongside a benchmark. Confirm at least 12 GiB host memory
  remains available before starting either workload and stop if the desktop,
  inference service, or system responsiveness degrades.

## Optimization contract

- Preserve target weights, target logits, KV precision, sampling semantics, and
  the 262,144-token server context contract.
- Pursue the highest-gain or lowest-effort candidates first. Require a credible
  and repeatable end-to-end gain of at least 3% on the target workload.
- Auto-tuning is allowed, but use exact production shapes, unbiased cache and
  memory conditions, bounded search spaces, and the same resource limits.
- Record observations separately from hypotheses. Retain a candidate only after
  correctness, neighboring-shape, quality, energy, and end-to-end gates pass.
