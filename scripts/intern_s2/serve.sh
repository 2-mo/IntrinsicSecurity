#!/usr/bin/env bash
set -euo pipefail
cd /data4/r1/workspace/cwork/projects/IntrinsicSecurity
mkdir -p outputs/intern_s2
exec > >(tee -a outputs/intern_s2/server.log) 2>&1
export PATH="$PWD/.venv-intern-s2/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# This host has PCIe-only GPU links; bypass the P2P path that stalled startup.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
command -v ninja
ninja --version
echo "[$(date -Is)] Checking four-GPU NCCL communication before loading weights"
timeout --kill-after=5s 120s .venv-intern-s2/bin/torchrun \
    --standalone --nproc-per-node=4 scripts/intern_s2/check_nccl.py
echo "[$(date -Is)] Starting Intern-S2-Preview, TP=4, BF16"
exec .venv-intern-s2/bin/vllm serve /data3/Intern-S2-Preview \
    --served-model-name Intern-S2-Preview \
    --host 127.0.0.1 --port 22100 \
    --trust-remote-code --tensor-parallel-size 4 \
    --dtype bfloat16 --reasoning-parser qwen3 \
    --max-model-len 32768 --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 --enforce-eager \
    --disable-custom-all-reduce \
    --allowed-local-media-path /data4/r1/workspace/cwork/datasets/MuSLR \
    --limit-mm-per-prompt '{"image":4,"video":0}'
