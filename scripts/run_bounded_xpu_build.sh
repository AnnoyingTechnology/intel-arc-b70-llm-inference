#!/usr/bin/env bash
set -euo pipefail

readonly source_tree="${1:-/tmp/b70-xpu-autotune}"
readonly builder_image="${B70_BUILDER_IMAGE:-b70-xpu-autotune-builder:2026.0}"

if [[ ! -d "${source_tree}" ]]; then
    printf 'Missing source tree: %s\n' "${source_tree}" >&2
    exit 1
fi

# Keep the workstation responsive. Equal memory and memory-swap limits disable
# container swap; an oversized build is terminated in its cgroup, not by the
# host-wide OOM killer.
exec docker run --rm --ipc=host \
    --name=b70-xpu-w4a16-build \
    --memory=26g \
    --memory-swap=26g \
    --cpus=8 \
    --pids-limit=1024 \
    -v "${source_tree}:/workspace/xpu" \
    -w /workspace/xpu \
    -e VLLM_VERSION_OVERRIDE=0.1.14.dev16+ga397c58.prefill64 \
    -e BASIC_KERNELS_ENABLED=OFF \
    -e FA2_KERNELS_ENABLED=OFF \
    -e GDN_KERNELS_ENABLED=ON \
    -e MHC_KERNELS_ENABLED=OFF \
    -e MOE_KERNELS_ENABLED=OFF \
    -e MQA_LOGITS_KERNELS_ENABLED=OFF \
    -e XPU_SPECIFIC_KERNELS_ENABLED=ON \
    -e XPUMEM_ALLOCATOR_ENABLED=OFF \
    -e CMAKE_BUILD_PARALLEL_LEVEL=4 \
    -e MAX_JOBS=4 \
    "${builder_image}" \
    bash -lc \
    'python setup.py bdist_wheel --dist-dir /workspace/xpu/dist --py-limited-api=cp38'
