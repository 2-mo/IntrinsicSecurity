"""Live checkpoint progress and estimated remaining generation time for PiMux."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[2] / "outputs/intern_s2"
BENCHMARKS = {
    "muslr": ("MuSLR", "muslr", 1093),
    "muslr-json": ("MuSLR JSON (TP2)", "muslr_json_tp2", 1093),
    "cot-control": ("CoT-Control", "cot_control", 2383),
    "processbench": ("ProcessBench", "processbench", 3400),
}


class Checkpoint:
    def __init__(self, path):
        self.path, self.offset, self.identity, self.rows = path, 0, None, {}

    def read(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return self.rows
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.rows, self.offset, self.identity = {}, 0, identity
        with self.path.open() as stream:
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line or not line.endswith("\n"):
                    stream.seek(start)
                    break
                try:
                    row = json.loads(line)
                    self.rows[row["id"]] = row
                except (json.JSONDecodeError, KeyError):
                    continue
            self.offset = stream.tell()
        return self.rows


def service_speed():
    try:
        active = ROOT / "server.active"
        log_name = active.read_text().strip() if active.exists() else "server.log"
        with (ROOT / log_name).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            lines = stream.read().decode(errors="replace").splitlines()
        for line in reversed(lines):
            match = re.search(r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs", line)
            if match:
                return float(match[1]), int(match[2])
    except OSError:
        pass
    return 0.0, 0


def phase(name):
    try:
        text = (ROOT / f"{name}.phase").read_text().strip()
    except FileNotFoundError:
        return "未启动", "white"
    if "paused" in text:
        return "已暂停", "yellow"
    if "queued" in text:
        queue = ROOT / "queue.phase"
        if queue.exists() and queue.read_text().startswith("FAILED:"):
            return "队列异常，查看 queue 窗口", "red"
        return "排队：等待双卡服务", "yellow"
    if text.startswith("complete"):
        return "已完成", "green"
    if "exit_code=" in text and "exit_code=0 " not in text:
        return "异常退出，请查看日志", "red"
    if text.startswith("scoring"):
        return "CPU 评分中", "cyan"
    if text.startswith("full"):
        return "全量推理中", "cyan"
    if text.startswith("smoke"):
        return "小样本测试", "cyan"
    return "等待服务", "yellow"


def duration(seconds):
    minutes = round(seconds / 60)
    return f"约 {minutes // 60}小时{minutes % 60:02d}分"


def gpu_status():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode:
            return "GPU 状态暂不可用"
        parts = []
        for line in result.stdout.splitlines():
            index, memory, utilization = [part.strip() for part in line.split(",")]
            parts.append(f"GPU{index} {int(memory)/1024:.1f}GB {utilization}%")
        return "  |  ".join(parts)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return "GPU 状态暂不可用"


def render(selected, readers):
    speed, running = service_speed()
    active = [key for key in BENCHMARKS if phase(key)[0] == "全量推理中"]
    bars = Progress(TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(),
                    TaskProgressColumn(), TextColumn("{task.fields[state]}"), expand=True)
    details = Table(expand=True, box=None)
    for column in ("数据集", "已解析", "请求失败", "输出截断", "预计剩余生成时间"):
        details.add_column(column)
    for key in selected:
        name, _, total = BENCHMARKS[key]
        rows = list(readers[key].read().values())
        state, color = phase(key)
        bars.add_task(name, total=total, completed=len(rows), state=f"[{color}]{state}[/]")
        failed = sum(r.get("status") in {"failed", "skipped_missing_image"} for r in rows)
        parsed = sum((r.get("prediction") is not None) if key == "processbench" else
                     bool(r.get("prediction") or r.get("extracted_answer")) for r in rows)
        truncated = sum(bool(r.get("truncated")) or r.get("finish_reason") == "length" for r in rows)
        tokens = [(r.get("usage") or {}).get("completion_tokens", 0) for r in rows[-30:]]
        tokens = [value for value in tokens if value > 0]
        eta = "—"
        if state == "全量推理中" and speed > 0 and tokens and active == [key]:
            eta = duration((total - len(rows)) * sum(tokens) / len(tokens) / speed)
        elif state == "全量推理中":
            eta = "多任务共享服务，暂不估计"
        details.add_row(name, str(parsed), str(failed), str(truncated), eta)
    return Group(
        Panel(Group(bars, Text(""), details), title="Intern-S2-Preview · 实时实验进度", border_style="cyan"),
        Text(f"刷新时间 {time.strftime('%H:%M:%S')} · 总生成速度 {speed:.1f} tokens/s · 服务处理中 {running} 个请求"),
        Text(gpu_status(), style="dim"),
        Text("每 2 秒刷新；ETA 按最近 30 题输出长度估算，另需预留最终评分时间。", style="dim"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS) + ["all"], default="all")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    selected = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]
    readers = {key: Checkpoint(ROOT / BENCHMARKS[key][1] / "predictions.jsonl") for key in selected}
    console = Console()
    if args.once:
        console.print(render(selected, readers))
        return
    with Live(render(selected, readers), console=console, auto_refresh=False) as live:
        while True:
            time.sleep(2)
            live.update(render(selected, readers), refresh=True)


if __name__ == "__main__":
    main()
