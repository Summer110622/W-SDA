"""Real localhost HTTP + two worker processes. CPU integration, NOT a WAN benchmark."""
from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsda.cli import evaluate
from wsda.config import ModelConfig, RunConfig
from wsda.coordinator import Coordinator
from wsda.data import prepare


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    runtime = root / "artifacts" / "http_runtime"
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    text = runtime / "corpus.txt"
    text.write_text(("summer trains a small language model.\n"
                     "rope gives position. delta stores associations. nope attends causally.\n") * 600)
    prepare(text, runtime / "data")
    cfg = RunConfig(model=ModelConfig(d_model=32, num_heads=4, num_layers=3,
                                      intermediate_size=64, max_seq_len=32, archive_capacity=8),
                     seq_len=16, local_steps=4, min_local_steps=2, max_local_steps=8,
                     target_seconds=0.25, server_alpha=0.8, max_staleness=8,
                     learning_rate=0.002)
    cfg.save(runtime / "config.json")
    c = Coordinator(runtime / "state", cfg)
    c.export(runtime / "initial")
    c.close()
    before = evaluate(runtime / "initial", runtime / "data", batches=12, batch_size=2)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    token = secrets.token_urlsafe(32)
    env = dict(os.environ, WSDA_TOKEN=token, PYTHONPATH=str(root), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    server_log = (runtime / "server.log").open("w")
    workers, logs = [], []

    def start_server():
        return subprocess.Popen([sys.executable, "-m", "wsda.cli", "coordinator", "--state", str(runtime / "state"),
                                 "--port", str(port)], cwd=root, env=env, stdout=server_log, stderr=subprocess.STDOUT)

    def wait_ready(proc):
        for _ in range(200):
            if proc.poll() is not None:
                raise RuntimeError("coordinator exited; inspect server.log")
            try:
                if httpx.get(url + "/healthz", timeout=0.3, trust_env=False).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise TimeoutError("coordinator not ready")

    server = start_server()
    try:
        wait_ready(server)
        started = time.perf_counter()
        for i, batch in enumerate((1, 2)):
            worker_id = f"cpu-{i}"
            log = (runtime / f"{worker_id}.log").open("w")
            logs.append(log)
            cmd = [sys.executable, "-m", "wsda.cli", "worker", "--server", url,
                   "--data", str(runtime / "data"), "--worker-id", worker_id,
                   "--jobs", "4", "--batch-size", str(batch), "--device", "cpu",
                   "--work-dir", str(runtime / worker_id), "--num-shards", "2", "--shard-index", str(i)]
            workers.append(subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT))
        for proc in workers:
            if proc.wait(timeout=90) != 0:
                raise RuntimeError("worker failed; inspect worker logs")
        elapsed = time.perf_counter() - started
        headers = {"Authorization": "Bearer " + token}
        status = httpx.get(url + "/v1/status", headers=headers, trust_env=False).json()
        server.terminate()
        server.wait(timeout=10)
        server = start_server()
        wait_ready(server)
        resumed = httpx.get(url + "/v1/status", headers=headers, trust_env=False).json()
        assert status["version"] == resumed["version"]
        assert status["workers"] == resumed["workers"]
        c = Coordinator(runtime / "state")
        c.export(runtime / "final")
        c.close()
        after = evaluate(runtime / "final", runtime / "data", batches=12, batch_size=2)
        rows = []
        for i in range(2):
            rows.extend(json.loads(line) for line in (runtime / f"cpu-{i}" / "metrics.jsonl").read_text().splitlines())
        accepted = [r for r in rows if r["accepted"]]
        assert status["version"] == len(accepted) == 8
        assert set(status["workers"]) == {"cpu-0", "cpu-1"}
        assert after["validation_loss"] < before["validation_loss"]
        report = {"scope": "localhost HTTP, 2 CPU subprocess workers; no GPU/WAN measurement",
                  "torch_version": torch.__version__, "python_version": sys.version.split()[0],
                  "parameters": sum(p.numel() for p in __import__('wsda').HybridLM(cfg.model).parameters()),
                  "accepted_updates": len(accepted), "rejected_updates": len(rows)-len(accepted),
                  "server_restart_preserved_state": True, "wall_seconds_including_worker_startup": elapsed,
                  "initial": before, "final": after,
                  "train_tokens": sum(r["tokens"] for r in accepted),
                  "total_download_bytes": sum(r["download_bytes"] for r in rows),
                  "total_upload_bytes": sum(r.get("upload_bytes", 0) for r in rows),
                  "observed_local_steps": [r["local_steps"] for r in rows],
                  "max_observed_staleness": max(r["staleness"] for r in accepted),
                  "run_config": cfg.to_dict()}
        out = root / "artifacts" / "http_smoke_results.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        for proc in workers:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=10)
        for log in logs:
            log.close()
        server_log.close()


if __name__ == "__main__":
    main()
