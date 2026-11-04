import json
import time

import pytest
import torch
from fastapi.testclient import TestClient

from wsda.codec import pack, unpack
from wsda.coordinator import Coordinator, LeaseRequest, choose_steps, create_app
from wsda.worker import secure_endpoint


def req(worker="home", **kwargs):
    return LeaseRequest(worker_id=worker, tokenizer_fingerprint="wsda-byte-v1", vocab_size=258,
                        batch_size=1, grad_accum=1, **kwargs)


def update(base, job, amount=0.001):
    delta = {k: torch.full_like(v, amount) for k, v in base.items()}
    info = {k: job[k] for k in ("run_id", "lease_id", "worker_id", "base_version")}
    info.update(steps=1, tokens=job["config"]["seq_len"], loss=2.0, compute_seconds=0.1)
    return pack(delta, info, "fp32")


def test_codec_real_int8():
    torch.manual_seed(1)
    x = {"weight": torch.randn(10001), "small": torch.zeros(3, 2)}
    fp = pack(x, {}, "fp32")
    quant = pack(x, {}, "int8")
    got, _ = unpack(quant, x)
    assert len(quant) < len(fp) * 0.35
    assert (got["weight"] - x["weight"]).abs().max() <= x["weight"].abs().max() / 254 + 1e-6
    torch.testing.assert_close(got["small"], x["small"])
    full, _ = unpack(fp, x)
    torch.testing.assert_close(full["weight"], x["weight"], atol=0, rtol=0)


def test_codec_rejects_bad_input():
    with pytest.raises(ValueError):
        pack({"w": torch.tensor([float("nan")])}, {})
    with pytest.raises(ValueError):
        unpack(b"bad")
    with pytest.raises(ValueError):
        unpack(pack({"w": torch.ones(4)}, {}), {"w": torch.ones(3)})


def test_async_mixing_idempotency_and_resume(tmp_path, cfg):
    c = Coordinator(tmp_path, cfg)
    initial, j1 = unpack(c.lease(req("a")))
    _, j2 = unpack(c.lease(req("b")))
    blob1 = update(initial, j1, 0.002)
    blob2 = update(initial, j2, 0.004)
    a = c.submit(j1["lease_id"], blob1)
    b = c.submit(j2["lease_id"], blob2)
    assert a["alpha"] == 0.5 and b["alpha"] == 0.25
    version, current = c.latest()
    assert version == 2
    for k in current:
        torch.testing.assert_close(current[k], initial[k] + 0.00175, atol=1e-7, rtol=1e-5)
    assert c.submit(j1["lease_id"], blob1) == a
    assert c.latest()[0] == 2
    with pytest.raises(ValueError, match="different payload"):
        c.submit(j1["lease_id"], update(initial, j1, 0.1))
    c.close()
    c = Coordinator(tmp_path)
    assert c.latest()[0] == 2
    assert c.submit(j1["lease_id"], blob1) == a
    c.close()


def test_stale_expired_and_schema(tmp_path, cfg):
    cfg.max_staleness = 0
    c = Coordinator(tmp_path, cfg)
    base, a = unpack(c.lease(req("a")))
    _, b = unpack(c.lease(req("b")))
    assert c.submit(a["lease_id"], update(base, a))["accepted"]
    assert c.submit(b["lease_id"], update(base, b))["reason"] == "stale"
    base, d = unpack(c.lease(req("d")))
    c.db.execute("UPDATE leases SET expires=? WHERE id=?", (time.time()-1, d["lease_id"]))
    assert c.submit(d["lease_id"], update(base, d))["reason"] == "expired"
    base, e = unpack(c.lease(req("e")))
    with pytest.raises(ValueError):
        c.submit(e["lease_id"], pack({"wrong": torch.ones(2)}, {}, "fp32"))
    c.close()


def test_auth_http_and_limits(tmp_path, cfg):
    token = "x" * 40
    cfg.max_payload_mb = 1
    app = create_app(tmp_path, cfg, token)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/status").status_code == 401
        headers = {"Authorization": "Bearer " + token}
        assert client.get("/v1/status", headers=headers).status_code == 200
        response = client.post("/v1/lease", json=req().model_dump(), headers=headers)
        assert response.status_code == 200
        base, job = unpack(response.content)
        response = client.post("/v1/update/" + job["lease_id"], content=update(base, job), headers=headers)
        assert response.json()["accepted"]
        assert client.post("/v1/update/" + "a"*32, content=b"x"*(1024**2+1), headers=headers).status_code == 413
        bad = req().model_dump()
        bad["vocab_size"] = 999
        assert client.post("/v1/lease", json=bad, headers=headers).status_code == 400
    app.state.coordinator.close()


def test_heterogeneous_step_scheduler(cfg):
    cfg.max_local_steps = 64
    assert choose_steps(cfg, 100, {"tokens_per_second": 100}, 0) == 1
    assert choose_steps(cfg, 100, {"tokens_per_second": 400}, 0) == 4
    assert choose_steps(cfg, 100, {"tokens_per_second": 400}, 2) == 32
    assert choose_steps(cfg, 100, {"tokens_per_second": 4000}, 100) == 64


@pytest.mark.parametrize("url", ["http://example.org:8000", "https://token@example.org", "file:///tmp/x"])
def test_insecure_endpoint_rejected(url):
    with pytest.raises(ValueError):
        secure_endpoint(url)


def test_private_http_opt_in():
    assert secure_endpoint("http://coordinator:8000", allow_http=True) == "http://coordinator:8000"
    assert secure_endpoint("http://127.0.0.1:8000")


def test_malformed_safetensors_returns_400(tmp_path, cfg):
    token = "y" * 40
    app = create_app(tmp_path, cfg, token)
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer " + token}
        base, job = unpack(client.post("/v1/lease", json=req().model_dump(), headers=headers).content)
        corrupted = update(base, job)[:-1]
        response = client.post("/v1/update/" + job["lease_id"], content=corrupted, headers=headers)
        assert response.status_code == 400
        assert client.get("/v1/status", headers=headers).json()["version"] == 0
    app.state.coordinator.close()


def test_resume_config_conflict(tmp_path, cfg):
    c = Coordinator(tmp_path, cfg)
    c.close()
    cfg.learning_rate *= 2
    with pytest.raises(ValueError, match="saved run config differs"):
        Coordinator(tmp_path, cfg)


def test_same_worker_lease_retry_with_capacity_one(tmp_path, cfg):
    cfg.max_leases = 1
    c = Coordinator(tmp_path, cfg)
    _, old = unpack(c.lease(req()))
    base, new = unpack(c.lease(req()))
    assert old["lease_id"] != new["lease_id"]
    assert c.submit(old["lease_id"], update(base, old))["reason"] == "unknown_or_expired_lease"
    assert c.submit(new["lease_id"], update(base, new))["accepted"]
    c.close()
