from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from .codec import pack, unpack
from .config import RunConfig
from .data import TokenDataset, Tokenizer, prepare
from .model import HybridLM


def load_checkpoint(path, device="cpu"):
    root = Path(path)
    cfg = RunConfig.load(root / "run_config.json")
    model = HybridLM(cfg.model)
    weights, _ = unpack((root / "model.safetensors").read_bytes(), expected=model.state_dict())
    model.load_state_dict(weights)
    return model.to(device), cfg


@torch.no_grad()
def evaluate(path, data, *, batches=10, batch_size=1, device="cpu"):
    if batches < 1 or batch_size < 1:
        raise ValueError("batches and batch_size must be positive")
    model, cfg = load_checkpoint(path, device)
    model.eval()
    ds = TokenDataset(data, cfg.seq_len, split="validation", seed=12345)
    if ds.manifest["tokenizer_fingerprint"] != cfg.tokenizer_fingerprint:
        raise ValueError("tokenizer mismatch")
    losses = [float(model(*ds.batch(batch_size, device))["loss"]) for _ in range(batches)]
    loss = sum(losses) / len(losses)
    return {"validation_loss": loss, "perplexity": math.exp(loss),
            "tokens": batches * batch_size * cfg.seq_len, "tokenizer": cfg.tokenizer_fingerprint}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wsda")
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads per process")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="prepare a local UTF-8 corpus (no data upload)")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", help="local tokenizer.json; omitted = UTF-8 bytes")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p = sub.add_parser("configure", help="bind a run config to the dataset vocabulary")
    p.add_argument("--template", default="configs/tiny.json")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="run.json")
    p = sub.add_parser("coordinator")
    p.add_argument("--state", default="runs/home")
    p.add_argument("--config", help="omit when resuming an existing run")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("worker")
    p.add_argument("--server", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--worker-id", required=True)
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    p.add_argument("--wire", choices=["int8", "fp32"], default="int8")
    p.add_argument("--work-dir")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--allow-http", action="store_true")
    p.add_argument("--max-payload-mb", type=int, default=256)
    p = sub.add_parser("status")
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--allow-http", action="store_true")
    p = sub.add_parser("export")
    p.add_argument("--state", default="runs/home")
    p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("generate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("local", help="single-worker scratch-training baseline without networking")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    if args.command == "prepare":
        print(json.dumps(prepare(args.input, args.out, tokenizer=args.tokenizer, val_fraction=args.val_fraction), indent=2))
    elif args.command == "configure":
        cfg = RunConfig.load(args.template)
        manifest = json.loads((Path(args.data) / "manifest.json").read_text())
        cfg.model.vocab_size = manifest["vocab_size"]
        cfg.tokenizer_fingerprint = manifest["tokenizer_fingerprint"]
        cfg.save(args.out)
        print(f"Wrote {args.out}")
    elif args.command == "coordinator":
        import uvicorn
        from .coordinator import create_app
        cfg = RunConfig.load(args.config) if args.config else None
        app = create_app(args.state, cfg)
        uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)
    elif args.command == "worker":
        from .worker import run_worker
        kwargs = vars(args).copy()
        kwargs.pop("command")
        kwargs.pop("threads")
        run_worker(**kwargs, token=os.environ.get("WSDA_TOKEN", ""))
    elif args.command == "status":
        from .worker import Client
        c = Client(args.server, os.environ.get("WSDA_TOKEN", ""), allow_http=args.allow_http)
        try:
            print(json.dumps(json.loads(c.request("GET", "/v1/status")), indent=2))
        finally:
            c.close()
    elif args.command == "export":
        from .coordinator import Coordinator
        if not (Path(args.state) / "coordinator.sqlite3").exists():
            parser.error("state directory does not contain an existing run")
        c = Coordinator(args.state)
        try:
            print(json.dumps(c.export(args.out)))
        finally:
            c.close()
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.checkpoint, args.data, batches=args.batches,
                                  batch_size=args.batch_size, device=args.device), indent=2))
    elif args.command == "generate":
        model, cfg = load_checkpoint(args.checkpoint, args.device)
        tok = Tokenizer(args.tokenizer)
        if tok.fingerprint != cfg.tokenizer_fingerprint:
            parser.error("pass the training tokenizer.json (fingerprint mismatch)")
        ids = tok.encode(args.prompt)
        if not ids:
            parser.error("prompt must contain at least one token")
        prompt = torch.tensor([ids], device=args.device)
        result = model.generate(prompt, args.max_new_tokens, args.temperature)
        print(tok.decode(result[0].tolist()))
    elif args.command == "local":
        from .worker import train_local
        cfg = RunConfig.load(args.config)
        ds = TokenDataset(args.data, cfg.seq_len, seed=cfg.seed)
        if ds.manifest["tokenizer_fingerprint"] != cfg.tokenizer_fingerprint:
            parser.error("tokenizer mismatch; run wsda configure first")
        torch.manual_seed(cfg.seed)
        model = HybridLM(cfg.model)
        base = {k: v.clone() for k, v in model.state_dict().items()}
        delta, metrics = train_local(model, base, ds, cfg, steps=args.steps,
                                    batch_size=args.batch_size, grad_accum=1,
                                    device=torch.device(args.device))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.safetensors").write_bytes(pack({k: base[k] + delta[k] for k in base}, {}, "fp32"))
        cfg.save(out / "run_config.json")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
