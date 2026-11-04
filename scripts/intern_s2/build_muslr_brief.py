"""Audit completed MuSLR runs and export a Chinese presentation brief."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'outputs/intern_s2'
FINAL_LINE = re.compile(
    r'^Answer\s*:\s*(?:\{\s*(true|false|unknown|[a-d])\s*\}|'
    r'(true|false|unknown|[a-d]))\s*\.?$', re.I)


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else 0


def audit(directory, structured):
    source = ROOT / directory
    summary = json.loads((source / 'summary.json').read_text())
    with (source / 'predictions.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    assert len(rows) == len({r['id'] for r in rows}) == 1093
    assert summary['reasoning_quality']['status'] == 'computed'
    valid = nonempty = syntax_valid = 0
    for row in rows:
        text = row.get('output') or ''
        nonempty += bool(text.strip())
        allowed = ['A', 'B', 'C', 'D'] if row['task_type'] == 'multiple_choice' else ['True', 'False', 'Unknown']
        if structured:
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                good = False
            else:
                syntax_valid += 1
                good = (isinstance(value, dict) and set(value) == {'reasoning', 'final_answer'}
                        and isinstance(value['reasoning'], str)
                        and isinstance(value['final_answer'], str)
                        and value['final_answer'] in allowed)
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            match = FINAL_LINE.fullmatch(lines[-1]) if lines else None
            good = bool(match and (match[1] or match[2]).upper() in {v.upper() for v in allowed})
        valid += good
        assert bool(good) == bool(row.get('prediction')), row['id']
    result = {
        'total': len(rows), 'strict_format_valid': valid, 'nonempty_final_outputs': nonempty,
        'empty_final_outputs': len(rows) - nonempty,
        'nonempty_format_invalid': nonempty - valid,
        'format_valid_percent_all_samples': percent(valid, len(rows)),
        'format_valid_percent_nonempty_outputs': percent(valid, nonempty),
        'json_syntax_valid': syntax_valid if structured else None,
        'summary': summary,
        'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ('summary.json', 'predictions.jsonl')},
    }
    assert valid == summary['overall']['parsed']
    return result


def main():
    now = datetime.now(ZoneInfo('Asia/Singapore'))
    output = ROOT / f'muslr_brief_{now:%Y%m%d}'
    output.mkdir(parents=True, exist_ok=True)
    text, structured = audit('muslr', False), audit('muslr_json_tp2', True)
    runs = [text, structured]
    accuracy_gain = (structured['summary']['overall']['accuracy_end_to_end_percent']
                     - text['summary']['overall']['accuracy_end_to_end_percent'])
    format_gain = structured['format_valid_percent_all_samples'] - text['format_valid_percent_all_samples']
    rows = []

    def add(label, values, definition):
        rows.append([label, *values, definition])

    add('样本数', [str(r['total']) for r in runs], '同一 MuSLR 全量样本')
    add('端到端准确率', [f"{r['summary']['overall']['accuracy_end_to_end_percent']:.2f}% ({r['summary']['overall']['correct']}/1093)" for r in runs], '正确答案数 / 1093；无有效答案计错')
    add('格式正确率（全样本）', [f"{r['format_valid_percent_all_samples']:.2f}% ({r['strict_format_valid']}/1093)" for r in runs], '满足各自严格输出协议的样本数 / 1093')
    add('可解析样本准确率', [f"{r['summary']['overall']['accuracy_scored_percent']:.2f}%" for r in runs], '正确答案数 / 可解析答案数')
    add('格式正确率（非空最终输出）', [f"{r['format_valid_percent_nonempty_outputs']:.2f}% ({r['strict_format_valid']}/{r['nonempty_final_outputs']})" for r in runs], '满足输出协议数 / 非空最终输出数')
    add('空最终输出', [str(r['empty_final_outputs']) for r in runs], 'output 字段为空或只有空白，不含 reasoning 通道')
    add('非空但格式不合规', [str(r['nonempty_format_invalid']) for r in runs], '最终输出非空，但未满足各自协议')
    for key, label in [('rouge_l', 'ROUGE-L F1'), ('bertscore_f1', 'BERTScore-F1'), ('roscoe', 'ROSCOE')]:
        add(label, [f"{r['summary']['reasoning_quality'][key]['score']:.4f}" for r in runs],
            '推理文本质量，分数范围 0–1；覆盖样本不同')
    add('推理指标三项均值', [f"{r['summary']['reasoning_quality']['overall_mean']:.4f}" for r in runs], 'ROUGE-L、BERTScore、ROSCOE 的算术平均')
    add('推理质量评分样本数', [str(r['summary']['reasoning_quality']['samples']) for r in runs], '具有可用生成推理和参考推理的样本')
    domain_rows = []
    for key in ('Overall', 'Traffic', 'Science', 'Finance', 'Others'):
        values = [r['summary']['compact_domain_table']['row'][key] for r in runs]
        domain_rows.append([key, *[f'{v:.2f}%' for v in values], f'{values[1]-values[0]:+.2f} pp'])

    conclusion = (f'在 MuSLR 全量 1,093 题上，JSON 方案的端到端准确率为 54.44%，'
                  f'原始文本方案为 49.41%，提高 {accuracy_gain:.2f} 个百分点；'
                  f'全样本格式正确率由 80.51% 提升至 88.01%，提高 {format_gain:.2f} 个百分点。'
                  '可解析样本准确率分别为 61.85% 和 61.36%，观察到的改善主要伴随有效最终答案数量增加。')
    notes = [
        '格式正确率按全量 1,093 题计算，空最终输出也计为格式无效；不是“只在有输出的样本上”计算。',
        'JSON 严格协议：完整 JSON 对象，仅含 reasoning（字符串）与 final_answer；final_answer 必须属于对应题型的枚举。',
        '原始文本严格协议：最后一个非空行是 Answer: 加合法标签；忽略大小写，允许空白、配对大括号和句末英文句号。逐条核验的有效数与现有评分器解析数一致。',
        'JSON 的 131 条无效记录均为空最终输出，不应表述为“131 条非空 JSON 语法错误”；962 条非空最终输出全部满足 Schema。',
        '原始文本有 201 条空最终输出、12 条非空但格式不合规；其中共 190 条无有效答案的记录达到输出上限。JSON 对应达到上限的记录为 9 条，不能据此解释其余空输出的原因。',
        '两次均为 Intern-S2-Preview、BF16、temperature=0、输出上限 8192 token；原始文本采用 TP4，JSON 采用 TP2，提示词/输出约束也不同，属于方案对比而非仅改变 JSON 格式的单变量实验。',
        '推理指标覆盖原始文本 1093 题、JSON 962 题。原始文本主要使用保存的模型 reasoning 通道，JSON 使用对象内 reasoning 字段；评分文本来源和覆盖不同，不能将数值差异直接归因为格式本身。',
        'ROSCOE 使用 reference-based reasoning_alignment 映射；MuSLR 论文未公开完整 scorer 配置，因此不是保证配置完全一致的官方分数。',
    ]
    markdown = [
        '# MuSLR 汇报对照', '', f'生成时间：{now:%Y-%m-%d %H:%M:%S}（UTC+08:00）。两组推理及评分均已完成。',
        '', '## 可直接用于汇报的结论', '', conclusion,
        '', '## 核心结果', '', '| 指标 | 原始文本 | JSON | 口径 |', '| :--- | ---: | ---: | :--- |',
        *['| ' + ' | '.join(row) + ' |' for row in rows],
        '', '## 分领域准确率', '', '| 领域 | 原始文本 | JSON | JSON − 原始 |', '| :--- | ---: | ---: | ---: |',
        *['| ' + ' | '.join(row) + ' |' for row in domain_rows],
        '', '## 统计口径与汇报边界', '', *[f'{i}. {note}' for i, note in enumerate(notes, 1)],
        '', '## 数据来源', '',
        '- 原始文本：`../muslr/summary.json`、`../muslr/predictions.jsonl`。',
        '- JSON：`../muslr_json_tp2/summary.json`、`../muslr_json_tp2/predictions.jsonl`。',
        '- 审计计数与文件 SHA-256：`format_audit.json`。源预测未修改。', '',
    ]
    (output / 'MuSLR汇报对照.md').write_text('\n'.join(markdown), encoding='utf-8')
    with (output / 'MuSLR结果对照.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['指标', '原始文本', 'JSON', '统计口径'])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(['领域', '原始文本', 'JSON', '差值'])
        writer.writerows(domain_rows)
    (output / 'format_audit.json').write_text(json.dumps(
        {'generated_at': now.isoformat(), 'original_text': text, 'json': structured,
         'notes': notes, 'accuracy_gain_pp': round(accuracy_gain, 2),
         'format_gain_pp': round(format_gain, 2)}, ensure_ascii=False, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'],
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout='constrained')
    x = [0, 1]
    colors = ['#718096', '#1670c5']
    values = [[49.41, 80.51], [54.44, 88.01]]
    for i, (label, series) in enumerate(zip(['原始文本（TP4）', 'JSON（TP2）'], values)):
        bars = ax.bar([p + (i-.5)*.32 for p in x], series, width=.30, label=label, color=colors[i])
        ax.bar_label(bars, labels=[f'{v:.2f}%' for v in series], padding=4, fontsize=11)
    ax.set_xticks(x, ['端到端准确率', '格式正确率（全样本）'], fontsize=12)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_title('MuSLR 两种输出方案对照（N = 1,093）', fontsize=15, pad=14)
    ax.legend(loc='upper left', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    for index, gain in enumerate([accuracy_gain, format_gain]):
        ax.text(index, max(values[0][index], values[1][index])+10,
                f'+{gain:.2f} 个百分点', ha='center', color=colors[1], fontsize=10)
    fig.savefig(output / 'MuSLR准确率与格式正确率.png', dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    fig.text(.06, .965, 'MuSLR 两种输出方案评估对照', fontsize=19, color='#16324f', va='top')
    fig.text(.06, .929, f'Intern-S2-Preview · 全量 1,093 题 · {now:%Y-%m-%d} · 两组评分均完成', fontsize=9, color='#526274')
    fig.text(.06, .887, f'端到端准确率 +{accuracy_gain:.2f} pp     格式正确率 +{format_gain:.2f} pp',
             fontsize=15, color='#1670c5')
    ax = fig.add_axes([.06, .40, .88, .45]); ax.axis('off')
    table = ax.table(cellText=[row[:3] for row in rows], colLabels=['指标', '原始文本（TP4）', 'JSON（TP2）'],
                     colWidths=[.42, .29, .29], cellLoc='left', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#dbe2ea'); cell.set_linewidth(.6)
        if row == 0: cell.set_facecolor('#16324f'); cell.set_text_props(color='white', weight='bold')
        elif row % 2 == 0: cell.set_facecolor('#f0f5fa')
    fig.text(.06, .37, '分领域端到端准确率', fontsize=12, color='#16324f')
    ax = fig.add_axes([.06, .225, .88, .13]); ax.axis('off')
    table = ax.table(cellText=domain_rows, colLabels=['领域', '原始文本', 'JSON', '差值'],
                     colWidths=[.25, .25, .25, .25], cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#dbe2ea'); cell.set_linewidth(.6)
        if row == 0: cell.set_facecolor('#e6eff9')
    footnotes = [
        '格式正确率：满足各自输出协议的条数 / 1093，空最终输出计入分母。',
        'JSON：完整对象、两个指定字段、类型正确且答案标签合法；962 条非空输出全部合规。',
        '原始文本：末行 Answer: 加合法标签；880 条合规，201 条空输出、12 条非空但不合规。',
        '两方案硬件并行度、提示词及输出约束不同，不是只改变格式的单变量实验。',
        '推理指标覆盖数与文本来源不同；ROSCOE 为 reasoning_alignment 复现映射。',
        '详细定义、汇报结论及源数据 SHA-256 见同目录 Markdown 与 format_audit.json。',
    ]
    for i, note in enumerate(footnotes):
        fig.text(.06, .19-i*.025, note, fontsize=8, color='#526274')
    fig.savefig(output / 'MuSLR汇报对照.pdf')
    plt.close(fig)
    files = [p for p in output.iterdir() if p.suffix in {'.md', '.csv', '.json', '.pdf', '.png'}]
    with zipfile.ZipFile(output / 'MuSLR汇报材料.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files): archive.write(path, path.name)
    print(output)
    for path in sorted(output.iterdir()): print(path.name, path.stat().st_size)


if __name__ == '__main__':
    main()
