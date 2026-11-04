from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch


class Tokenizer:
    """UTF-8 bytes for smoke tests, or a local Hugging Face tokenizers JSON."""
    def __init__(self, path: str | Path | None = None):
        self.impl = None
        if path is None:
            self.fingerprint, self.vocab_size, self.eos = "wsda-byte-v1", 258, 256
        else:
            from tokenizers import Tokenizer as HFTokenizer
            raw = Path(path).read_bytes()
            self.impl = HFTokenizer.from_file(str(path))
            self.fingerprint = "sha256:" + hashlib.sha256(raw).hexdigest()
            self.vocab_size = self.impl.get_vocab_size(with_added_tokens=True)
            self.eos = None

    def encode(self, text):
        return list(text.encode("utf-8")) if self.impl is None else self.impl.encode(text).ids

    def decode(self, ids):
        return bytes(x for x in ids if x < 256).decode("utf-8", errors="replace") if self.impl is None else self.impl.decode(ids)


def prepare(input_path: str | Path, output_dir: str | Path, *, tokenizer=None, val_fraction=0.1):
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    tok = Tokenizer(tokenizer)
    text = Path(input_path).read_text(encoding="utf-8")
    ids = np.asarray(tok.encode(text), dtype=np.int32)
    if len(ids) < 32:
        raise ValueError("corpus too short (at least 32 tokens)")
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()):
        raise ValueError("output data directory must be empty")
    path.mkdir(parents=True, exist_ok=True)
    split = int(len(ids) * (1 - val_fraction))
    np.save(path / "train.npy", ids[:split], allow_pickle=False)
    np.save(path / "validation.npy", ids[split:], allow_pickle=False)
    manifest = {"format": 1, "tokenizer_fingerprint": tok.fingerprint,
                "vocab_size": tok.vocab_size, "train_tokens": split,
                "validation_tokens": len(ids) - split,
                "corpus_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if tokenizer is not None:
        shutil.copyfile(tokenizer, path / "tokenizer.json")
    return manifest


class TokenDataset:
    def __init__(self, path, seq_len, *, split="train", shard_index=0, num_shards=1, seed=42):
        if split not in {"train", "validation"} or not 0 <= shard_index < num_shards:
            raise ValueError("invalid split/shard")
        self.path, self.seq_len = Path(path), seq_len
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.tokens = np.load(self.path / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        if self.tokens.ndim != 1 or self.tokens.dtype.kind not in "iu":
            raise ValueError("token file must be a 1D integer array")
        # Adjacent blocks share one boundary input, but not target positions.
        count = (len(self.tokens) - 1) // seq_len
        self.blocks = np.arange(shard_index, count, num_shards)
        if not len(self.blocks):
            raise ValueError("not enough tokens for this sequence length/shard")
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size, device):
        starts = self.rng.choice(self.blocks, size=batch_size) * self.seq_len
        windows = np.stack([self.tokens[s:s + self.seq_len + 1] for s in starts]).astype(np.int64)
        if windows.min() < 0 or windows.max() >= self.manifest["vocab_size"]:
            raise ValueError("out-of-range token ID")
        data = torch.from_numpy(windows).to(device)
        return data[:, :-1].contiguous(), data[:, 1:].contiguous()
