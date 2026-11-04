#!/usr/bin/env bash
set -euo pipefail
cd /data4/r1/workspace/cwork/projects/IntrinsicSecurity
mkdir -p outputs/intern_s2
server_log=${INTERN_S2_SERVER_LOG:-server.log}
exec > >(tee -a "outputs/intern_s2/$server_log") 2>&1
printf '%s\n' "$server_log" > outputs/intern_s2/server.active
export PATH="$PWD/.venv-intern-s2/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${INTERN_S2_GPUS:-0,1,2,3}
tp=${INTERN_S2_TP:-4}
port=${INTERN_S2_PORT:-22100}
sequences=${INTERN_S2_MAX_SEQS:-16}
memory=${INTERN_S2_MEMORY:-0.90}
extra_args=()
if [[ -n ${INTERN_S2_STRUCTURED_BACKEND:-} ]]; then
    extra_args+=(--structured-outputs-config "{\"backend\":\"$INTERN_S2_STRUCTURED_BACKEND\"}")
fi
if [[ ${INTERN_S2_STRUCTURED_BACKEND:-} == outlines ]]; then
    .venv-intern-s2/bin/python scripts/intern_s2/patch_outlines_eos.py
fi
if [[ -n ${INTERN_S2_MAX_BATCHED_TOKENS:-} ]]; then
    extra_args+=(--max-num-batched-tokens "$INTERN_S2_MAX_BATCHED_TOKENS")
fi
if [[ ${INTERN_S2_TEXT_ONLY:-0} == 1 ]]; then
    extra_args+=(--language-model-only)
    if [[ -z ${INTERN_S2_MAX_BATCHED_TOKENS:-} ]]; then
        extra_args+=(--max-num-batched-tokens 1024)
    fi
else
    extra_args+=(--allowed-local-media-path /data4/r1/workspace/cwork/datasets/MuSLR
      --limit-mm-per-prompt '{"image":4,"video":0}')
fi
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# This host has PCIe-only GPU links; bypass the P2P path that stalled startup.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
command -v ninja
ninja --version
echo "[$(date -Is)] Checking $tp-GPU NCCL communication before loading weights"
timeout --kill-after=5s 120s .venv-intern-s2/bin/torchrun \
    --standalone --nproc-per-node="$tp" scripts/intern_s2/check_nccl.py
echo "[$(date -Is)] Starting Intern-S2-Preview, TP=$tp, GPUs=$CUDA_VISIBLE_DEVICES, BF16"
exec .venv-intern-s2/bin/vllm serve /data3/Intern-S2-Preview \
    --served-model-name Intern-S2-Preview \
    --host 127.0.0.1 --port "$port" \
    --trust-remote-code --tensor-parallel-size "$tp" \
    --dtype bfloat16 --reasoning-parser qwen3 \
    --max-model-len 32768 --max-num-seqs "$sequences" \
    --gpu-memory-utilization "$memory" --enforce-eager \
    --disable-custom-all-reduce \
    "${extra_args[@]}"
