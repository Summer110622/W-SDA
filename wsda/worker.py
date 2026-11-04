from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import torch

from .codec import metadata, pack, unpack
from .config import RunConfig
from .data import TokenDataset
from .model import HybridLM


def secure_endpoint(url: str, allow_http=False):
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("URL must not contain credentials, query, or fragment")
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("invalid coordinator URL")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"} and not allow_http:
        raise ValueError("WAN requires HTTPS; --allow-http is only for trusted private networks")
    return url.rstrip("/")


class Client:
    def __init__(self, endpoint, token, *, allow_http=False, timeout=180, max_payload_mb=256):
        self.endpoint = secure_endpoint(endpoint, allow_http)
        if len(token) < 32:
            raise ValueError("WSDA_TOKEN must contain at least 32 characters")
        self.client = httpx.Client(base_url=self.endpoint, timeout=timeout, follow_redirects=False,
                                   headers={"Authorization": "Bearer " + token}, trust_env=False)
        self.limit = max_payload_mb * 1024**2

    def request(self, method, path, **kwargs):
        # Retried updates must reuse identical bytes. A retry after an ACK loss
        # receives the persisted receipt, not a second application of the delta.
        for attempt in range(5):
            try:
                with self.client.stream(method, path, **kwargs) as response:
                    if response.status_code in {429, 502, 503, 504}:
                        raise httpx.TransportError(f"retryable status {response.status_code}")
                    response.raise_for_status()
                    payload = bytearray()
                    for chunk in response.iter_bytes():
                        if len(payload) + len(chunk) > self.limit:
                            raise ValueError("response exceeds max_payload_mb")
                        payload.extend(chunk)
                    return bytes(payload)
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 16) + random.random() * 0.2)
        raise AssertionError("unreachable")

    def close(self):
        self.client.close()


def train_local(model, base, dataset, cfg, *, steps, batch_size, grad_accum, device, precision="auto"):
    if steps < 1 or batch_size < 1 or grad_accum < 1:
        raise ValueError("steps, batch_size, and grad_accum must be positive")
    model.load_state_dict(base, strict=True)
    model.to(device).train()
    # Optimizer is reset after each rebase. Momentum from a different base model
    # is not silently reused. This is NOT DiLoCo's two-optimizer algorithm.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("unknown precision")
    if device.type != "cuda" and precision != "fp32":
        raise ValueError("mixed precision is supported on CUDA only")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("device does not support bf16; use fp16 or fp32")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    context = (lambda: torch.autocast("cuda", dtype=dtype)) if dtype else contextlib.nullcontext
    if device.type == "cuda":
        torch.cuda.synchronize()
    start, total_loss, done, attempts = time.perf_counter(), 0.0, 0, 0
    while done < steps:
        attempts += 1
        if attempts > steps * 3 + 10:
            raise FloatingPointError("too many fp16 overflow retries")
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(grad_accum):
            x, y = dataset.batch(batch_size, device)
            with context():
                loss = model(x, y)["loss"] / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not torch.isfinite(norm) and precision != "fp16":
            raise FloatingPointError("non-finite gradients")
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < before:  # skipped fp16 overflow; do not claim a completed step
            continue
        total_loss += loss_sum
        done += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    delta = {k: p.detach().cpu().float() - base[k] for k, p in model.state_dict().items()}
    return delta, {"steps": done, "tokens": done * batch_size * grad_accum * cfg.seq_len,
                   "loss": total_loss / done, "compute_seconds": seconds}


def run_worker(*, server, data, worker_id, token, jobs=10, batch_size=1, grad_accum=1,
               device="auto", precision="auto", wire="int8", work_dir=None,
               shard_index=0, num_shards=1, allow_http=False, max_payload_mb=256):
    if jobs < 1 or batch_size < 1 or grad_accum < 1:
        raise ValueError("jobs, batch_size, and grad_accum must be positive")
    import re
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", worker_id):
        raise ValueError("worker_id must be 1..64 alphanumeric, '-' or '_' characters")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    work_dir = Path(work_dir or f".wsda-workers/{worker_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    # Avoid two processes writing the same outbox on Linux/Colab. A clear error is
    # preferable to silently applying two different payloads for one lease.
    lock = (work_dir / ".lock").open("a")
    try:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        lock.close()
        raise RuntimeError("worker locking requires Linux/macOS (Docker on Windows is supported)")
    except OSError:
        lock.close()
        raise RuntimeError("worker work_dir is already in use")
    pending = work_dir / "pending.safetensors"
    identity = hashlib.sha256(secure_endpoint(server, allow_http).encode()).hexdigest()
    client = Client(server, token, allow_http=allow_http, max_payload_mb=max_payload_mb)
    manifest = json.loads((Path(data) / "manifest.json").read_text())
    last_comm, model, dataset, results = 0.0, None, None, []
    try:
        if pending.exists():
            blob = pending.read_bytes()
            info = metadata(blob)["info"]
            if info.get("endpoint_hash") != identity or info.get("worker_id") != worker_id:
                raise ValueError("outbox belongs to another endpoint/worker; use a separate work_dir")
            result = json.loads(client.request("POST", f"/v1/update/{info['lease_id']}", content=blob))
            pending.unlink()
            print(json.dumps({"recovered_outbox": result}), flush=True)
        for _ in range(jobs):
            start = time.perf_counter()
            payload = client.request("POST", "/v1/lease", json={
                "worker_id": worker_id, "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
                "vocab_size": manifest["vocab_size"], "batch_size": batch_size,
                "grad_accum": grad_accum, "last_comm_seconds": last_comm})
            pull_seconds = time.perf_counter() - start
            base, job = unpack(payload)
            cfg = RunConfig(**job["config"])
            if dataset is None:
                seed = cfg.seed + int.from_bytes(hashlib.sha256(worker_id.encode()).digest()[:4], "little")
                torch.manual_seed(seed)
                dataset = TokenDataset(data, cfg.seq_len, shard_index=shard_index,
                                       num_shards=num_shards, seed=seed)
                model = HybridLM(cfg.model)
            effective_batch = batch_size * grad_accum
            while True:
                try:
                    delta, metrics = train_local(model, base, dataset, cfg, steps=job["steps"],
                        batch_size=batch_size, grad_accum=grad_accum, device=device, precision=precision)
                    break
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise RuntimeError("model/sequence cannot fit this GPU; reduce model or seq_len") from None
                    model.cpu()
                    torch.cuda.empty_cache()
                    batch_size = max(1, math.gcd(effective_batch, batch_size // 2))
                    grad_accum = effective_batch // batch_size
                    print(json.dumps({"oom_retry_microbatch": batch_size, "grad_accum": grad_accum}), flush=True)
            info = {k: job[k] for k in ("run_id", "lease_id", "worker_id", "base_version")}
            info.update(metrics, endpoint_hash=identity)
            blob = pack(delta, info, mode=wire)
            if len(blob) > client.limit:
                raise ValueError("update exceeds max_payload_mb")
            # Atomic, fsynced outbox before any POST. Keep on retry exhaustion.
            tmp = pending.with_suffix(".tmp")
            with tmp.open("wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(pending)
            start = time.perf_counter()
            result = json.loads(client.request("POST", f"/v1/update/{job['lease_id']}", content=blob))
            last_comm = pull_seconds + time.perf_counter() - start
            pending.unlink()
            result.update(download_bytes=len(payload), communication_seconds=last_comm,
                          compute_seconds=metrics["compute_seconds"], local_steps=metrics["steps"])
            with (work_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(result, allow_nan=False) + "\n")
            print(json.dumps(result, allow_nan=False), flush=True)
            results.append(result)
    finally:
        client.close()
        lock.close()
    return results
