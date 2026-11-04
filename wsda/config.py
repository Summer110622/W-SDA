from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 258
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    intermediate_size: int = 384
    max_seq_len: int = 128
    layer_pattern: list[str] = field(default_factory=lambda: ["rope", "sda", "nope"])
    rope_theta: float = 10000.0
    dropout: float = 0.0
    archive_capacity: int = 32
    sda_variant: str = "base"  # base: residual correction; plus: soft archive/state blend
    sda_threshold: float = 0.5
    sda_gate_temperature: float = 0.1
    sda_top_m: int = 4
    sda_softmax_scale: float = 16.0
    archive_fake_quant: bool = False  # STE simulation, NOT packed training memory
    gradient_checkpointing: bool = False

    def __post_init__(self):
        for name in ("vocab_size", "d_model", "num_heads", "num_layers", "intermediate_size", "max_seq_len"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2:
            raise ValueError("head dimension must be an even integer")
        if not self.layer_pattern or set(self.layer_pattern) - {"rope", "sda", "nope"}:
            raise ValueError("layer_pattern must contain rope, sda, or nope")
        if not 0 <= self.dropout < 1 or self.rope_theta <= 0:
            raise ValueError("invalid dropout or rope_theta")
        if self.archive_capacity < 0 or self.sda_top_m < 1:
            raise ValueError("invalid archive_capacity or sda_top_m")
        if self.sda_variant not in {"base", "plus"}:
            raise ValueError("sda_variant must be base or plus")
        if not -1 <= self.sda_threshold <= 1:
            raise ValueError("sda_threshold must be in [-1,1]")
        if self.sda_gate_temperature <= 0 or self.sda_softmax_scale <= 0:
            raise ValueError("SDA temperatures must be positive")
        for value in asdict(self).values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite configuration")

    def to_dict(self):
        return asdict(self)


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer_fingerprint: str = "wsda-byte-v1"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seq_len: int = 64
    local_steps: int = 8
    min_local_steps: int = 1
    max_local_steps: int = 64
    target_seconds: float = 60.0
    server_alpha: float = 0.5
    max_staleness: int = 4
    lease_seconds: float = 3600.0
    max_delta_norm: float = 100.0
    max_payload_mb: int = 256
    max_leases: int = 16
    seed: int = 42

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if not 1 <= self.seq_len <= self.model.max_seq_len:
            raise ValueError("seq_len must fit model.max_seq_len")
        if not 1 <= self.min_local_steps <= self.local_steps <= self.max_local_steps:
            raise ValueError("invalid local step limits")
        if not 0 < self.server_alpha <= 1 or self.max_staleness < 0:
            raise ValueError("invalid aggregation parameters")
        for name in ("learning_rate", "grad_clip", "target_seconds", "lease_seconds", "max_delta_norm", "max_payload_mb", "max_leases"):
            if getattr(self, name) <= 0 or not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite and positive")
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise ValueError("invalid weight_decay")
        if not self.tokenizer_fingerprint:
            raise ValueError("tokenizer_fingerprint is required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
