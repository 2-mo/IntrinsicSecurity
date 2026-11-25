# MuSLR · Qwen3-VL-8B 原图推理与评测

输入为 MuSLR 官方 1,093 道题的原图、题干、前提和选项。运行一次即可生成逐题答案与推理，并计算准确率和参考推理的 ROSCOE 对齐分数。

## 运行

已有 Qwen3-VL-8B-Instruct 的 vLLM 服务及客户端依赖时，在本目录执行：

```bash
bash run.sh --dataset /path/to/MuSLR --endpoint http://127.0.0.1:8000/v1
```

程序先写出 `runs/official/predictions.jsonl`，每行包含题号、预测答案、推理文本和判定方式。随后读取数据集 `metadata.csv` 里的官方答案和参考推理逐题评分，写出 `runs/official/scores.json`。多服务可重复传入 `--endpoint`。再次从头运行时，使用 `--output runs/second` 指定新的输出目录。

## 官方全量实测

使用三个 Qwen3-VL-8B-Instruct 副本、36 个题目工作线程，从固定版本的原图重新生成全部答案和解释。下表按原始 1,093 题计分。

| 参数设置 | 答对 | 准确率 | ROSCOE reasoning_alignment |
| --- | ---: | ---: | ---: |
| 官方全量选定的固定系数 | 860 / 1,093 | 78.6825% | 0.899306 |

## 在新机器上准备

进入本目录，用 Python 3.12 建立客户端环境并取得固定版本的官方题目：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf download Aiden0526/MuSLR --repo-type dataset \
  --revision 16dbb73e00bfc49011f645bcb97510f0ad7de962 \
  --include 'metadata.csv' 'images/*' --local-dir assets/MuSLR
```

在有 GPU 的终端启动以下后端。实测配置为一张 A100 40 GB 承载一个 BF16 副本。

```bash
python3.12 -m venv .venv-server
source .venv-server/bin/activate
pip install vllm==0.22.1 transformers==5.8.1
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --revision cadac78306af287f801b75a5565ede58f323f472 \
  --served-model-name Qwen3-VL-8B-Instruct --port 8000 \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 16 \
  --max-num-batched-tokens 8192 --limit-mm-per-prompt '{"image":2,"video":0}' \
  --gpu-memory-utilization 0.85 --enforce-eager
```

回到客户端终端，执行 `bash run.sh --dataset assets/MuSLR`。评分器首次运行时下载固定版本的 `sentence-transformers/all-mpnet-base-v2`。已有本地权重可加 `--encoder-path /path/to/all-mpnet-base-v2`。首次试运行可加 `--limit 4 --output runs/smoke`。

## 每道题怎样得到答案

1. 将前提里的条件句和析取句转换成命题子句。对每个可询问的视觉命题分别假设“真”和“假”，选取最能缩小未决结论范围的命题，让 8B 查看原图并回答“是、否、看不出”。最多主动问三次。只有当图像事实与规则共同推出唯一候选时，采用这个结论。
2. 其余题由同一个 8B 模型比较候选。模型轮换选项顺序，检查每个结论的支持和反驳证据。不同判断发生分歧时可放大相关区域。选择题结合规则允许的可能情形打分，判断题结合候选句在有、无图文前提时的似然差打分。默认使用随代码发布的固定系数。
3. 模型也逐项读取题干中可解析的原子命题，供可能情形打分和解释生成使用。根据本题的视觉事实、可用规则和最终候选生成英文推理。评估阶段读取官方答案和参考步骤，核对所有题号均有一份有效输出。本文的 ROSCOE 对应 `reasoning_alignment`，论文未公布具体评分器配置。

默认系数曾使用官方全量答案选定，8B 模型参数始终冻结。需要独立选参时，可用外部图片和问题确定系数，再以同一条命令评估相同的官方全集。[固定系数](official_weights.json) · [外部选参](external/README.md)
