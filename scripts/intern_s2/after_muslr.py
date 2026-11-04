"""After four-GPU MuSLR completes, resume the text benchmarks on GPUs 0 and 1."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

import psutil

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "outputs/intern_s2"
SESSION = "intern-s2-eval"
BASE_URL = "http://127.0.0.1:22102/v1"
DEPLOYMENT = {
    "INTERN_S2_GPUS": "0,1", "INTERN_S2_TP": "2", "INTERN_S2_PORT": "22102",
    "INTERN_S2_MAX_SEQS": "4", "INTERN_S2_MEMORY": "0.95",
    "INTERN_S2_TEXT_ONLY": "1", "INTERN_S2_SERVER_LOG": "server_tp2.log",
}


def state(message):
    (OUTPUT / "queue.phase").write_text(message + "\n")
    print(f"[{datetime.now().astimezone().isoformat()}] {message}", flush=True)


def tmux(*args):
    return subprocess.check_output(["pimux", "tmux", *args], text=True).strip()


def stop_server(port, pane=None):
    target = pane or f"{SESSION}:server"
    pid = int(tmux("display-message", "-p", "-t", target, "#{pane_pid}"))
    root = psutil.Process(pid)
    cmd = root.cmdline()
    if "/data3/Intern-S2-Preview" not in cmd or str(port) not in cmd:
        raise RuntimeError(f"Server pane PID {pid} does not match the expected model/port; leaving it untouched")
    processes = root.children(recursive=True) + [root]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=10)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


def wait_complete(benchmark):
    path = OUTPUT / f"{benchmark}.phase"
    while True:
        text = path.read_text() if path.exists() else ""
        if text.startswith("complete"):
            return
        if "exit_code=" in text and "exit_code=0 " not in text:
            raise RuntimeError(f"{benchmark} exited unsuccessfully; see its PiMux log")
        time.sleep(15)


def wait_free_gpus():
    for _ in range(40):
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
        memory = {int(line.split(",")[0]): int(line.split(",")[1]) for line in result.splitlines()}
        if all(memory.get(index, 0) >= 39000 for index in (0, 1)):
            return
        state("Waiting for GPUs 0 and 1 to have enough free memory; unrelated GPU jobs are left untouched")
        time.sleep(15)
    raise RuntimeError("GPUs 0 and 1 do not have enough free memory for the BF16 two-GPU deployment")


def wait_ready_and_probe():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(90):
        try:
            with opener.open(BASE_URL.removesuffix("/v1") + "/health", timeout=3) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if tmux("display-message", "-p", "-t", f"{SESSION}:server", "#{pane_dead}") == "1":
            raise RuntimeError("Two-GPU model service exited; see server_tp2.log")
        time.sleep(10)
    else:
        raise RuntimeError("Two-GPU service did not become ready within 15 minutes")
    payload = {"model": "Intern-S2-Preview", "messages": [{"role": "user", "content": "Reply only OK."}],
               "max_tokens": 128, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(BASE_URL + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=120) as response:
        result = json.load(response)
    if not result.get("choices") or not result["choices"][0]["message"].get("content"):
        raise RuntimeError("Two-GPU probe did not return final answer text")
    (OUTPUT / "tp2_smoke.json").write_text(json.dumps(result, indent=2) + "\n")


def record_transition():
    record = {"time": datetime.now().astimezone().isoformat(), "endpoint": BASE_URL,
              "deployment": DEPLOYMENT, "existing_ids_from_four_gpu_run": {}}
    for directory in ("cot_control", "processbench"):
        path = OUTPUT / directory / "predictions.jsonl"
        ids = []
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("status") == "ok":
                    ids.append(row["id"])
        record["existing_ids_from_four_gpu_run"][directory] = ids
    with (OUTPUT / "deployment_history.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({"wait_for": "MuSLR inference and evaluation complete", "deployment": DEPLOYMENT,
                          "order": ["cot-control", "processbench"], "workers": 4,
                          "completed_predictions": "preserved; record original TP4 IDs before resuming",
                          "on_success": "stop the TP2 service and release its GPUs"}, indent=2))
        return
    os.chdir(PROJECT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "tp2_queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            state("Waiting for four-GPU MuSLR to finish; then CoT-Control -> ProcessBench on GPUs 0 and 1")
            wait_complete("muslr")
            summary = json.loads((OUTPUT / "muslr/summary.json").read_text())
            if summary["overall"]["prediction_records"] != 1093 or summary["overall"]["status_counts"].get("failed", 0):
                raise RuntimeError("MuSLR summary is incomplete or contains failed requests")
            state("MuSLR complete. Stopping TP4 service and preparing TP2 text-only service")
            stop_server(22100)
            wait_free_gpus()
            command = shlex.join(["env", *[f"{key}={value}" for key, value in DEPLOYMENT.items()],
                                   "bash", "scripts/intern_s2/serve.sh"])
            tmux("respawn-pane", "-k", "-t", f"{SESSION}:server", command)
            state("Starting and validating TP2 service; GPUs 2 and 3 remain available")
            wait_ready_and_probe()
            record_transition()
            for benchmark in ("cot-control", "processbench"):
                state(f"Running {benchmark} on two GPUs, four concurrent requests")
                command = shlex.join(["env", "INTERN_S2_WORKERS=4", f"INTERN_S2_BASE_URL={BASE_URL}",
                                       "bash", "scripts/intern_s2/run_dataset.sh", benchmark])
                pane = tmux("new-window", "-d", "-t", SESSION, "-n", benchmark,
                            "-c", str(PROJECT), "-P", "-F", "#{pane_id}", command)
                tmux("set-window-option", "-t", pane, "remain-on-exit", "on")
                wait_complete(benchmark)
            stop_server(22102)
            state("COMPLETE: all benchmarks evaluated; the two-GPU service was stopped to release GPUs")
        except Exception as error:
            state(f"FAILED: {type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
