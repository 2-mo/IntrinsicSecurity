# 使用外部样本确定系数

生成器从 A-OKVQA validation 的两组独立照片各取 180 张。每张图保留一道人注视觉选择题，并构造一道条件推理判断题。后者将视觉选项映射到两组后续事件，加入条件、排他约束和可选的未知前提。答案由人注视觉选项与预设规则计算。两组使用固定的来源题号，照片互不重复。

已选定的系数随代码发布，可直接运行官方全量推理及评估：

```bash
bash run.sh --dataset assets/MuSLR --weights external/weights.json --output runs/official-external
```

需要重新生成外部题并选参时，在项目的客户端环境、项目入口所在目录运行：

```bash
pip install pyarrow==25.0.0
python external/build.py --output assets/external
bash run.sh --dataset assets/external/calibration --external \
  --stage calibrate --output runs/external-fit
bash run.sh --dataset assets/MuSLR --weights runs/external-fit/weights.json \
  --output runs/official-external
```

第一条推理命令对 360 道外部题执行与官方测试相同的候选打分和证明判定，再用外部答案按题型选择非负整数系数，各系数之和为 8。准确率并列时依次优先非零项较少、较早分量较大的组合。第二条命令固定这些系数，从官方原图生成预测并评分。外部选参沿用项目已有的提示词和打分项，只有系数在外部样本上重新确定。

另有 360 道外部确认题。将 `--dataset` 指向 `assets/external/confirmation` 并加 `--external` 即可用固定系数评估新样本。其外部参考推理来自 A-OKVQA 人注解释和生成器的规则推导。

使用官方选定的默认系数，在独立的 180 张确认照片上答对 227/360 题，准确率 63.0556%，与这套外部参考推理的 ROSCOE 对齐分数为 0.831097。
