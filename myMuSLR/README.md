# MuSLR 本地评测

使用本地 `Qwen3-VL-8B` 模型复现 MuSLR baseline 和 LogiCAM。推理通过 RIPO 的
`PIVLMPool + run_batch` 分发到三个 vLLM endpoint，图像使用本地路径。

## 路径

- Python：`/data4/r1/workspace/cwork/projects/IntrinsicSecurity/.venv`
- 数据集：`/data4/r1/workspace/cwork/datasets/MuSLR`
- 官方提示词：`/data4/r1/workspace/cwork/projects/MuSLR/prompts`
- RIPO：`/data4/r1/workspace/cwork/projects/RIPO`

以下命令均在本目录执行：

```bash
cd /data4/r1/workspace/cwork/projects/IntrinsicSecurity/myMuSLR
```

## 全集推理

Baseline：

```bash
../.venv/bin/python run.py \
  --data /data4/r1/workspace/cwork/datasets/MuSLR \
  --method baseline \
  --output outputs/baseline_ripo_3gpu.jsonl \
  --endpoints http://127.0.0.1:22000/v1,http://127.0.0.1:22001/v1,http://127.0.0.1:22002/v1 \
  --workers 3 \
  --endpoint-concurrency 1
```

LogiCAM：

```bash
../.venv/bin/python run.py \
  --data /data4/r1/workspace/cwork/datasets/MuSLR \
  --method logicam \
  --output outputs/logicam_ripo_3gpu.jsonl \
  --endpoints http://127.0.0.1:22000/v1,http://127.0.0.1:22001/v1,http://127.0.0.1:22002/v1 \
  --workers 3 \
  --endpoint-concurrency 1 \
  --max-tokens 2048
```

只重跑未解析结果并原位替换（例如将输出上限提高到 4096）：

```bash
../.venv/bin/python run.py \
  --data /data4/r1/workspace/cwork/datasets/MuSLR \
  --method logicam \
  --output outputs/logicam_ripo_3gpu.jsonl \
  --endpoints http://127.0.0.1:22000/v1,http://127.0.0.1:22001/v1,http://127.0.0.1:22002/v1 \
  --workers 3 \
  --endpoint-concurrency 1 \
  --max-tokens 4096 \
  --retry-statuses unparsed
```

数据集共 1093 条。输出逐条写入 JSONL；重复执行同一命令会跳过已有结果并继续运行。

## 评估

首次使用安装评估依赖：

```bash
uv pip install --python ../.venv/bin/python -r requirements.txt
```

Baseline 全集、论文全部自动指标：

```bash
HF_ENDPOINT=https://hf-mirror.com ../.venv/bin/python evaluate.py \
  --data /data4/r1/workspace/cwork/datasets/MuSLR \
  --predictions outputs/baseline_ripo_3gpu.jsonl \
  --output outputs/baseline_ripo_3gpu.summary.json \
  --report outputs/baseline_ripo_3gpu.report.md \
  --model Qwen3-VL-8B \
  --reasoning-metrics \
  --reasoning-device cuda:1 \
  --table-output outputs/baseline_ripo_3gpu.table.md
```

LogiCAM 全集、论文全部自动指标：

```bash
HF_ENDPOINT=https://hf-mirror.com ../.venv/bin/python evaluate.py \
  --data /data4/r1/workspace/cwork/datasets/MuSLR \
  --predictions outputs/logicam_ripo_3gpu.jsonl \
  --output outputs/logicam_ripo_3gpu.summary.json \
  --report outputs/logicam_ripo_3gpu.report.md \
  --model "Qwen3-VL-8B + LogiCAM" \
  --reasoning-metrics \
  --reasoning-device cuda:1 \
  --table-output outputs/logicam_ripo_3gpu.table.md
```

报告按以下层次组织：

1. 主结果：严格端到端准确率、已解析准确率、答案覆盖率。
2. 推理质量：ROUGE-L、BERTScore-F1、ROSCOE、三项均值。
3. 准确率细分：Symbol、Domain、`Symbol x Domain` 论文主表。
4. 深度鲁棒性：2-3/4-5/6-7/8-9 分桶、逐深度结果、浅层到深层降幅。
5. 诊断：未解析、截断、失败、重复 ID、endpoint 分布。

论文没有发布 ROSCOE 的具体配置。本实现使用 ROSCOE 官方默认句向量模型
`all-mpnet-base-v2`，并将论文中的单个 ROSCOE 值映射为官方
`reasoning_alignment`；JSON 同时保存 grounding、informativeness、missing-step
等语义分项。论文中的跨模型 Pearson 相关性、LogiCAM 消融降幅和六类人工错误分布
需要额外模型运行或人工标注，报告会明确显示为未计算，不会用单次 baseline 推断。

两种方法都完成后进行对比：

```bash
../.venv/bin/python compare.py \
  --data /data4/r1/workspace/cwork/datasets/MuSLR \
  --baseline outputs/baseline_ripo_3gpu.jsonl \
  --logicam outputs/logicam_ripo_3gpu.jsonl \
  --output outputs/baseline_vs_logicam.summary.json \
  --report outputs/baseline_vs_logicam.report.md
```

## 可视化审阅

启动 Flask 页面查看 baseline 和 LogiCAM 的逐样本对比：

```bash
../.venv/bin/python dashboard.py --host 0.0.0.0 --port 5058
```

- 首页 `/`：一屏一个样本，展示本地图像、题目、标准答案、baseline/LogiCAM 输出、思维链类型、token、延迟和当前样本所属 domain/symbol 的指标。
- 汇总页 `/dashboard`：保留整体、domain、symbol、depth 和 reasoning quality 的图表与样本表。
- 论文对比页 `/paper-table`：列出论文 Table 1 的 Qwen、Llava、InternVL、InstructBLIP、Claude、GPT-4o、GPT-4.1、LogiCAM，以及本地 `Qwen3-VL-8B` baseline/LogiCAM 的完整 `Symbol x Domain` 表。
- 样本默认按 `LogiCAM fixes -> regressions -> unparsed -> both incorrect -> both correct` 排序，方便先看最有诊断价值的例子。

重新生成论文对比表：

```bash
../.venv/bin/python paper_comparison.py
```

输出为 `outputs/paper_vs_ours.table.md` 和 `outputs/paper_vs_ours.table.json`。

## 当前结果

| 指标 | Baseline | LogiCAM | 差值 |
| :--- | ---: | ---: | ---: |
| 端到端准确率 | 47.03% | 57.64% | +10.61 pp |
| 已解析准确率 | 51.66% | 78.85% | +27.19 pp |
| 答案覆盖率 | 91.03% | 73.10% | -17.93 pp |
| ROUGE-L | 0.3041 | 0.2550 | -0.0491 |
| BERTScore-F1 | 0.8455 | 0.8333 | -0.0122 |
| ROSCOE | 0.8099 | 0.7514 | -0.0585 |

LogiCAM 多答对 116 条，但有 293 条未解析输出触及 2048-token 上限。完整联合报告见
`outputs/baseline_vs_logicam.report.md`；单方法完整报告分别见
`outputs/baseline_ripo_3gpu.report.md` 和 `outputs/logicam_ripo_3gpu.report.md`。
论文 Table 1 表格按 `Symbol x Domain` 计算端到端准确率，未解析结果按错误计入。

## 测试

```bash
PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -v
```
