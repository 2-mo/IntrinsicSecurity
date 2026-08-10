# MuSLR 本地推理

本目录使用 RIPO 的 `PIVLMConfig + PIVLMPool + run_batch` 调用本地
OpenAI-compatible vLLM 服务，运行 MuSLR baseline 或 LogiCAM 提示词。默认配置已经对应
当前机器：

- 工作目录：`/data1/nemo/my_projects/IntrinsicSecurity/myMuSLR`
- 数据集：`/data1/nemo/data/MuSLR`
- 官方提示词：`/data1/nemo/ex_methods/MuSLR/prompts`
- RIPO：`/data1/nemo/my_projects/RIPO`
- 输出目录：`/data1/nemo/my_projects/IntrinsicSecurity/outputs`
- vLLM endpoint：`http://127.0.0.1:22000/v1`

## 环境

项目环境位于本目录的 `.venv`：

```bash
cd /data1/nemo/my_projects/IntrinsicSecurity/myMuSLR
source .venv/bin/activate
```

如需重建环境：

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

## 修改配置

运行参数集中在 `.env`，可直接编辑。`.env.example` 是可提交的模板，真实 `.env` 已被
Git 忽略。系统环境变量优先于 `.env`，命令行参数优先于两者。

常用项：

```dotenv
MUSLR_DATA=/data1/nemo/data/MuSLR
MUSLR_OUTPUT_DIR=/data1/nemo/my_projects/IntrinsicSecurity/outputs
RIPO_ROOT=/data1/nemo/my_projects/RIPO
RIPO_ENDPOINTS=http://127.0.0.1:22000/v1
RIPO_MODEL=anom-pi
RIPO_IMAGE_URL_MODE=data
```

`RIPO_IMAGE_URL_MODE=data` 会把本地图像编码进请求，不要求 vLLM 的
`--allowed-local-media-path` 覆盖 MuSLR 数据目录；当前服务配置应保留此值。

## 运行

先只检查路径、数据和样本选择，不调用模型：

```bash
.venv/bin/python run.py --dry-run
```

跑一条样本验证服务：

```bash
.venv/bin/python run.py --limit 1 \
  --output /data1/nemo/my_projects/IntrinsicSecurity/outputs/muslr_smoke.jsonl
```

跑完整 baseline：

```bash
.venv/bin/python run.py
```

默认输出：

```text
/data1/nemo/my_projects/IntrinsicSecurity/outputs/muslr_baseline.jsonl
/data1/nemo/my_projects/IntrinsicSecurity/outputs/muslr_baseline.jsonl.meta.json
```

跑完整 LogiCAM：

```bash
.venv/bin/python run.py --method logicam --max-tokens 2048
```

输出采用逐条追加的 JSONL checkpoint。重复执行会跳过已有结果；只重跑失败或未解析项：

```bash
.venv/bin/python run.py --retry-statuses failed,unparsed
```

如果启用了多个 endpoint，在 `.env` 中用逗号分隔，并相应提高 worker 数：

```dotenv
RIPO_ENDPOINTS=http://127.0.0.1:22000/v1,http://127.0.0.1:22001/v1
MUSLR_WORKERS=2
```

## 测试

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```
