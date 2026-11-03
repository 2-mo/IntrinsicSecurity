#!/usr/bin/env bash
set -euo pipefail
cd /data4/r1/workspace/cwork/projects/IntrinsicSecurity
mkdir -p outputs/intern_s2
exec > >(tee -a outputs/intern_s2/setup.log) 2>&1
export UV_CACHE_DIR=/tmp/intern-s2-uv-cache
export UV_HTTP_TIMEOUT=180
echo "[$(date -Is)] Preparing isolated Intern-S2 serving environment"
rm -f outputs/intern_s2/setup.complete
if [[ ! -x .venv-intern-s2/bin/python ]]; then
    /data4/r1/.local/bin/uv venv --python .venv/bin/python .venv-intern-s2
fi
/data4/r1/.local/bin/uv pip install --python .venv-intern-s2/bin/python 'vllm==0.22.1' 'transformers==5.8.1'
.venv-intern-s2/bin/python -c 'import torch,vllm; print("vLLM",vllm.__version__,"torch",torch.__version__,"GPUs",torch.cuda.device_count())'
.venv-intern-s2/bin/python -c 'from transformers import AutoConfig; c=AutoConfig.from_pretrained("/data3/Intern-S2-Preview",trust_remote_code=True); print("Model config loaded:",c.model_type)'
echo "[$(date -Is)] SETUP COMPLETE"
touch outputs/intern_s2/setup.complete
