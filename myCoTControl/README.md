# CoT-Control QA 独立运行

通过 OpenAI 兼容 HTTP 接口运行 CoT-Control QA，无需 RIPO 或本地模型权重。
模型部署与评测客户端分开，接口需要支持 `/v1/models` 和聊天补全。
导出包包含官方 QA 代码、提示词、三个 CSV 数据集、许可证和下载来源记录。

在解压目录运行，推荐 Python 3.12：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r myCoTControl/requirements.txt
cp .env.example .env
```

编辑 `.env` 中的 `OPENAI_BASE_URL`（包括 `/v1`）、`OPENAI_MODEL` 和
`OPENAI_API_KEY`。在原始仓库使用时，复制 `myCoTControl/.env.example` 到仓库根目录的 `.env`。
模型名称必须与服务公布的名称一致，也可通过 `--model`、`--endpoints`、`--api-key` 覆盖。
多个端点用逗号分隔，必须服务同一模型。

先检查数据和提示词，再运行：

```bash
.venv/bin/python myCoTControl/run.py --dry-run --output outputs/check
.venv/bin/python myCoTControl/run.py --output outputs/smoke
.venv/bin/python myCoTControl/run.py --limit 0 --output outputs/full
# 已有结果离线重新汇总，保持与推理时相同的筛选和模型参数：
.venv/bin/python myCoTControl/run.py --limit 0 --evaluate-only --output outputs/full
```

默认每个数据集、每个模式选择三道题，运行 baseline 和 word_suppression。
使用 `--datasets gpqa hle mmlu_pro_mini` 选择数据集，`--modes` 选择控制模式，
`--data-dir /path/to/csv-directory` 指定其他数据位置。
输出包括 `predictions.jsonl`、`metadata.json`、`prompts.jsonl` 和 `summary.json`。
相同配置可断点续跑；更换模型或实验参数时使用新的输出目录。

规则服从率需要服务返回推理通道（`reasoning_content` 或 `reasoning`），
也支持文本中的 `<think>…</think>` / `<output_reasoning>…</output_reasoning>`。
没有推理文本或输出被截断时，该条规则分数记为不可评估，全请求口径仍计入分母。
本客户端评测 QA 答案和规则服从率，不包含 Agent 任务及外部 meta-discussion LLM 裁判。

共享工具函数位于 `benchmark_common/`，随导出包提供；无需安装另外两个评测项目。
`--model-path` 只用于可选的来源记录和故障诊断，客户端不会加载模型。

```bash
.venv/bin/python -m unittest discover -s myCoTControl -p test_run.py -v
```

从原始 IntrinsicSecurity 仓库重新导出：`python scripts/package_text_benchmarks.py`。
导出仅包含列明的源码和公共数据，不包含 `.env`、模型、实验结果或本机部署脚本。
官方文件及许可证见 `datasets/CoT-Control/`，下载版本见其中的 `download_manifest.json`。
