"""Single-coordinator asynchronous local training with durable, idempotent updates.

SQLite commits the new full-precision snapshot and its receipt atomically.
Use one uvicorn process. Trusted participants only; this is not Byzantine-robust.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from safetensors import SafetensorError

from .codec import pack, unpack
from .config import RunConfig
from .model import HybridLM


class LeaseRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    tokenizer_fingerprint: str = Field(max_length=128)
    vocab_size: int = Field(gt=0)
    batch_size: int = Field(ge=1, le=1024)
    grad_accum: int = Field(ge=1, le=1024)
    last_comm_seconds: float = Field(default=0.0, ge=0, le=86400, allow_inf_nan=False)


def choose_steps(cfg: RunConfig, batch_tokens: int, stats: dict, comm_seconds: float) -> int:
    """EWMA throughput, target compute window, and a bounded communication budget.

    The communication floor targets <=20% communication only when max_local_steps
    and lease duration allow it. It is a heuristic, not an optimal scheduler.
    """
    tps = stats.get("tokens_per_second")
    if not tps:
        return cfg.local_steps
    desired_compute = min(max(cfg.target_seconds, 4 * comm_seconds), cfg.lease_seconds / 2)
    steps = round(desired_compute * tps / batch_tokens)
    return max(cfg.min_local_steps, min(cfg.max_local_steps, steps))


class Coordinator:
    def __init__(self, state_dir: str | Path, cfg: RunConfig | None = None):
        self.path = Path(state_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path / "coordinator.sqlite3", check_same_thread=False,
                                  isolation_level=None, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS snapshots (version INTEGER PRIMARY KEY, blob BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY, worker TEXT NOT NULL,
            version INTEGER NOT NULL, expires REAL NOT NULL, steps INTEGER NOT NULL,
            batch_tokens INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS receipts (id TEXT PRIMARY KEY, digest TEXT NOT NULL,
            response TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS workers (id TEXT PRIMARY KEY, stats TEXT NOT NULL);
        """)
        saved = self.db.execute("SELECT value FROM settings WHERE key='config'").fetchone()
        if saved:
            self.cfg = RunConfig(**json.loads(saved[0]))
            if cfg is not None and cfg.to_dict() != self.cfg.to_dict():
                raise ValueError("saved run config differs; resume without --config or use a new state directory")
            self.run_id = self.db.execute("SELECT value FROM settings WHERE key='run_id'").fetchone()[0]
        else:
            self.cfg = cfg or RunConfig()
            self.run_id = uuid.uuid4().hex
            with torch.random.fork_rng():
                torch.manual_seed(self.cfg.seed)
                initial = HybridLM(self.cfg.model).state_dict()
            blob = pack(initial, {}, "fp32")
            if len(blob) > self.cfg.max_payload_mb * 1024**2:
                raise ValueError("model snapshot exceeds max_payload_mb")
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute("INSERT INTO settings VALUES ('config',?)", (json.dumps(self.cfg.to_dict()),))
                self.db.execute("INSERT INTO settings VALUES ('run_id',?)", (self.run_id,))
                self.db.execute("INSERT INTO snapshots VALUES (0,?)", (blob,))
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def close(self):
        self.db.close()

    def latest(self):
        with self.lock:
            version, blob = self.db.execute("SELECT version,blob FROM snapshots ORDER BY version DESC LIMIT 1").fetchone()
            weights, _ = unpack(blob)
            return version, weights

    def lease(self, request: LeaseRequest) -> bytes:
        if (request.tokenizer_fingerprint != self.cfg.tokenizer_fingerprint
                or request.vocab_size != self.cfg.model.vocab_size):
            raise ValueError("tokenizer/vocabulary mismatch")
        with self.lock:
            now = time.time()
            self.db.execute("DELETE FROM leases WHERE expires < ?", (now,))
            if self.db.execute("SELECT COUNT(*) FROM leases WHERE worker != ?", (request.worker_id,)).fetchone()[0] >= self.cfg.max_leases:
                raise RuntimeError("too many outstanding leases; wait for expiry")
            # One in-flight lease per worker ID. A fresh request supersedes a lost
            # lease response; use different IDs for concurrently running workers.
            self.db.execute("DELETE FROM leases WHERE worker=?", (request.worker_id,))
            version, weights = self.latest()
            row = self.db.execute("SELECT stats FROM workers WHERE id=?", (request.worker_id,)).fetchone()
            stats = json.loads(row[0]) if row else {}
            batch_tokens = request.batch_size * request.grad_accum * self.cfg.seq_len
            steps = choose_steps(self.cfg, batch_tokens, stats, request.last_comm_seconds)
            lease_id = uuid.uuid4().hex
            self.db.execute("INSERT INTO leases VALUES (?,?,?,?,?,?)",
                            (lease_id, request.worker_id, version, now + self.cfg.lease_seconds, steps, batch_tokens))
            info = {"run_id": self.run_id, "lease_id": lease_id, "worker_id": request.worker_id,
                    "base_version": version, "steps": steps, "config": self.cfg.to_dict()}
            return pack(weights, info, "fp32")

    def submit(self, lease_id: str, blob: bytes) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}", lease_id):
            raise ValueError("invalid lease ID")
        digest = hashlib.sha256(blob).hexdigest()
        with self.lock:
            receipt = self.db.execute("SELECT digest,response FROM receipts WHERE id=?", (lease_id,)).fetchone()
            if receipt:
                if not hmac.compare_digest(receipt[0], digest):
                    raise ValueError("same lease ID submitted with different payload")
                return json.loads(receipt[1])
            row = self.db.execute("SELECT worker,version,expires,steps,batch_tokens FROM leases WHERE id=?", (lease_id,)).fetchone()
            current_version, current = self.latest()
            if not row:
                return {"accepted": False, "reason": "unknown_or_expired_lease", "version": current_version}
            worker, base_version, expires, max_steps, batch_tokens = row
            stale = current_version - base_version
            if time.time() > expires or stale > self.cfg.max_staleness:
                self.db.execute("DELETE FROM leases WHERE id=?", (lease_id,))
                return {"accepted": False, "reason": "expired" if time.time() > expires else "stale",
                        "version": current_version, "staleness": stale}
            delta, info = unpack(blob, expected=current)
            if (info.get("lease_id") != lease_id or info.get("worker_id") != worker
                    or info.get("base_version") != base_version or info.get("run_id") != self.run_id):
                raise ValueError("update metadata does not match lease")
            steps, tokens, seconds, loss = (info.get(k) for k in ("steps", "tokens", "compute_seconds", "loss"))
            if type(steps) is not int or not 1 <= steps <= max_steps or tokens != steps * batch_tokens:
                raise ValueError("invalid step/token counts")
            if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid compute duration")
            if not isinstance(loss, (float, int)) or not math.isfinite(loss) or loss < 0:
                raise ValueError("invalid loss")
            base_blob = self.db.execute("SELECT blob FROM snapshots WHERE version=?", (base_version,)).fetchone()
            if base_blob is None:
                return {"accepted": False, "reason": "base_pruned", "version": current_version}
            base, _ = unpack(base_blob[0], expected=current)
            # Float64 norm avoids overflow before clipping. Invalid payloads were
            # rejected above; clipping is stability protection, not poisoning defense.
            norm = math.sqrt(sum(float(d.double().square().sum()) for d in delta.values()))
            clip = min(1.0, self.cfg.max_delta_norm / max(norm, 1e-12))
            alpha = self.cfg.server_alpha / (1 + stale)
            updated = {k: torch.lerp(current[k], base[k] + clip * delta[k], alpha) for k in current}
            version = current_version + 1
            response = {"accepted": True, "version": version, "base_version": base_version,
                        "staleness": stale, "alpha": alpha, "delta_norm": norm,
                        "clip_factor": clip, "worker_id": worker, "tokens": tokens,
                        "loss": float(loss), "upload_bytes": len(blob)}
            snapshot = pack(updated, {}, "fp32")
            stat_row = self.db.execute("SELECT stats FROM workers WHERE id=?", (worker,)).fetchone()
            old = json.loads(stat_row[0]) if stat_row else {}
            throughput = tokens / seconds
            stats = {"tokens_per_second": 0.3 * throughput + 0.7 * old.get("tokens_per_second", throughput),
                     "accepted_updates": old.get("accepted_updates", 0) + 1,
                     "tokens": old.get("tokens", 0) + tokens, "last_loss": loss,
                     "last_staleness": stale}
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute("INSERT INTO snapshots VALUES (?,?)", (version, snapshot))
                self.db.execute("INSERT INTO receipts VALUES (?,?,?)", (lease_id, digest, json.dumps(response)))
                self.db.execute("INSERT OR REPLACE INTO workers VALUES (?,?)", (worker, json.dumps(stats)))
                self.db.execute("DELETE FROM leases WHERE id=?", (lease_id,))
                self.db.execute("DELETE FROM snapshots WHERE version < ?", (version - self.cfg.max_staleness,))
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            return response

    def status(self):
        with self.lock:
            version = self.db.execute("SELECT MAX(version) FROM snapshots").fetchone()[0]
            workers = {k: json.loads(v) for k, v in self.db.execute("SELECT id,stats FROM workers")}
            return {"run_id": self.run_id, "version": version, "workers": workers,
                    "active_leases": self.db.execute("SELECT COUNT(*) FROM leases").fetchone()[0]}

    def export(self, path):
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        version, weights = self.latest()
        (out / "model.safetensors").write_bytes(pack(weights, {}, "fp32"))
        self.cfg.save(out / "run_config.json")
        (out / "metadata.json").write_text(json.dumps({"version": version, "run_id": self.run_id}))
        return {"version": version, "path": str(out)}


def create_app(state_dir="runs/home", cfg: RunConfig | None = None, token: str | None = None):
    token = token or os.environ.get("WSDA_TOKEN", "")
    if len(token) < 32 or token.startswith("REPLACE_"):
        raise ValueError("WSDA_TOKEN must contain at least 32 characters")
    coordinator = Coordinator(state_dir, cfg)
    app = FastAPI(title="W-SDA", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.coordinator = coordinator

    async def authorize(request: Request):
        auth = request.headers.get("authorization", "")
        if not hmac.compare_digest(auth, "Bearer " + token):
            raise HTTPException(401, "invalid bearer token")

    @app.get("/healthz")
    def health():
        return {"ok": True, "protocol": 1}

    @app.get("/v1/status", dependencies=[Depends(authorize)])
    def status():
        return coordinator.status()

    @app.post("/v1/lease", dependencies=[Depends(authorize)])
    def lease(request: LeaseRequest):
        try:
            return Response(coordinator.lease(request), media_type="application/octet-stream")
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(429, str(error)) from error

    @app.post("/v1/update/{lease_id}", dependencies=[Depends(authorize)])
    async def update(lease_id: str, request: Request):
        limit = coordinator.cfg.max_payload_mb * 1024**2
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > limit:
                raise HTTPException(413, "payload too large")
            data.extend(chunk)
        try:
            # Offload CPU tensor validation/aggregation; lock serializes updates.
            from starlette.concurrency import run_in_threadpool
            return await run_in_threadpool(coordinator.submit, lease_id, bytes(data))
        except (ValueError, KeyError, TypeError, RuntimeError, SafetensorError) as error:
            raise HTTPException(400, "invalid update: " + str(error)[:200]) from error

    return app
