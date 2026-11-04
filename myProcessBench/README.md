# ProcessBench 独立运行

通过 OpenAI 兼容 HTTP 接口评测数学推理中的首个错误步骤，无需 RIPO、
CoT-Control、本地模型权重或 vLLM Python 环境。模型服务独立部署，
接口需要支持 `/v1/models` 和聊天补全。
导出包包含官方四个 JSON 数据分片、critique 提示词和数据来源记录。

在解压目录运行，推荐 Python 3.12：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r myProcessBench/requirements.txt
cp .env.example .env
```

编辑 `.env` 中的 `OPENAI_BASE_URL`（包括 `/v1`）、`OPENAI_MODEL` 和
`OPENAI_API_KEY`。在原始仓库使用时，复制 `myProcessBench/.env.example` 到仓库根目录的 `.env`。
模型名称必须与服务公布的名称一致，也可通过 `--model`、`--base-url`、`--api-key` 覆盖。

```bash
.venv/bin/python myProcessBench/run.py --dry-run --limit 4 --output outputs/check
.venv/bin/python myProcessBench/run.py --limit 4 --output outputs/smoke
.venv/bin/python myProcessBench/run.py --output outputs/full
# 离线重新汇总已有结果，保持与推理时相同的模型和筛选参数：
.venv/bin/python myProcessBench/run.py --evaluate-only --output outputs/full
```

`--limit` 为每个分片的题数，默认 0 表示全量；小样本运行会兼顾有错和无错题。
可用 `--data /path/to/ProcessBench` 指定数据目录，`--workers` 设置并发，
`--max-tokens` 设置输出上限，`--timeout` 设置请求超时秒数。
输出包括 `predictions.jsonl`、`metadata.json` 和 `summary.json`。
相同配置可断点续跑；更换模型或实验参数时使用新的输出目录。

使用官方 critique 提示词，解析最终文本中的最后一个 `\boxed{整数}`。
有错样本的标签为从 0 开始的首错索引，无错标签为 -1。
报告有错题准确率、无错题准确率及二者的调和均值（官方 F1）；失败请求保留在分母中。

共享的 checkpoint 和推理通道处理函数位于 `benchmark_common/`，随导出包提供。
`--model-path` 只用于可选的来源记录，客户端不会加载模型。

```bash
.venv/bin/python -m unittest discover -s myProcessBench -p test_run.py -v
```

从原始 IntrinsicSecurity 仓库重新导出：`python scripts/package_text_benchmarks.py`。
导出仅包含列明的源码和公共数据，不包含 `.env`、模型、实验结果或本机部署脚本。
数据来源及许可声明见 `datasets/ProcessBench/README.md` 和 `download_manifest.json`，
提示词来自 QwenLM/ProcessBench，版本记录在评测脚本和下载清单中。
