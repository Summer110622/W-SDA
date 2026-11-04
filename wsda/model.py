"""Causal trainable reference; SDA loop is intentionally unfused.

No KV/state is transferred over the WAN. Each independent training window starts
with zero delta state. Archive policy is FIFO (not upstream's reservoir sampler).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


def apply_rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """x: [batch, heads, time, even head_dim]; adjacent-coordinate rotation."""
    t, d = x.shape[-2:]
    freq = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angle = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freq
    c, s = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)


def fake_int8(x: torch.Tensor) -> torch.Tensor:
    """Per-vector symmetric fake quantization, straight-through gradients.

    This does NOT reduce training activation memory. Actual WAN compression lives
    in codec.py and stores int8 bytes, not float tensors labelled as int8.
    """
    scale = (x.detach().abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8)
    dequantized = (x.detach() / scale).round().clamp(-127, 127) * scale
    return x + (dequantized - x).detach()


def delta_scan(q, k, v, beta, *, capacity=32, variant="base", threshold=0.5,
               temperature=0.1, top_m=4, softmax_scale=16.0, quantize=False):
    """Differentiable causal SDA scan on [B,H,T,D], beta on [B,H,T,1].

    Read follows write at t (valid causal self-attention). No in-place state or
    detached archive; gradients propagate through the entire training window.
    float64 is preserved for numerical tests; mixed-precision state uses float32.
    """
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, beta = [z.to(dtype) for z in (q, k, v, beta)]
    b, h, t, dk = k.shape
    state = k.new_zeros(b, h, dk, v.shape[-1])
    stored_k = fake_int8(k) if quantize else k
    stored_v = fake_int8(v) if quantize else v
    outputs = []
    for i in range(t):
        kt, vt, qt = k[:, :, i], v[:, :, i], q[:, :, i]
        error = vt - torch.einsum("bhd,bhdv->bhv", kt, state)
        state = state + beta[:, :, i, :, None] * kt[..., :, None] * error[..., None, :]
        fast = torch.einsum("bhd,bhdv->bhv", qt, state)
        if capacity:
            start = max(0, i + 1 - capacity)
            ak, av = stored_k[:, :, start:i + 1], stored_v[:, :, start:i + 1]
            sims = torch.einsum("bhd,bhnd->bhn", qt, ak)
            if variant == "base":
                # Equal similarities prefer the most recent FIFO entry.
                best, reverse_index = sims.flip(-1).max(-1)
                index = sims.shape[-1] - 1 - reverse_index
                key = ak.gather(2, index[..., None, None].expand(-1, -1, 1, dk)).squeeze(2)
                val = av.gather(2, index[..., None, None].expand(-1, -1, 1, v.shape[-1])).squeeze(2)
                residual = val - torch.einsum("bhd,bhdv->bhv", key, state)
                fast = fast + (best >= threshold)[..., None] * residual
            elif variant == "plus":
                scores, index = sims.topk(min(top_m, sims.shape[-1]), dim=-1)
                chosen = av.gather(2, index[..., None].expand(-1, -1, -1, v.shape[-1]))
                archive = (torch.softmax(scores * softmax_scale, -1)[..., None] * chosen).sum(-2)
                gate = torch.sigmoid((scores[..., :1] - threshold) / temperature)
                fast = gate * archive + (1 - gate) * fast
            else:
                raise ValueError("unknown SDA variant")
        outputs.append(fast)
    return torch.stack(outputs, dim=2)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


class DenseAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, rotary: bool):
        super().__init__()
        self.cfg, self.rotary = cfg, rotary
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.cfg.num_heads, d // self.cfg.num_heads).permute(2, 0, 3, 1, 4)
        if self.rotary:
            q, k = apply_rope(q, self.cfg.rope_theta), apply_rope(k, self.cfg.rope_theta)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          dropout_p=self.cfg.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class SummerDeltaAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.beta = nn.Linear(cfg.d_model, cfg.num_heads, bias=True)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.cfg.num_heads, d // self.cfg.num_heads).permute(2, 0, 3, 1, 4)
        q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
        beta = self.beta(x).float().sigmoid().transpose(1, 2).unsqueeze(-1)
        # Disable autocast for recurrent matrix accumulation (including einsum).
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = delta_scan(q, k, v.float(), beta, capacity=self.cfg.archive_capacity,
                           variant=self.cfg.sda_variant, threshold=self.cfg.sda_threshold,
                           temperature=self.cfg.sda_gate_temperature, top_m=self.cfg.sda_top_m,
                           softmax_scale=self.cfg.sda_softmax_scale,
                           quantize=self.cfg.archive_fake_quant)
        return self.out(y.to(x.dtype).transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = SummerDeltaAttention(cfg) if kind == "sda" else DenseAttention(cfg, kind == "rope")
        self.gate_up = nn.Linear(cfg.d_model, cfg.intermediate_size * 2, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        gate, up = self.gate_up(self.norm2(x)).chunk(2, -1)
        return x + self.dropout(self.down(F.silu(gate) * up))


class HybridLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, cfg.layer_pattern[i % len(cfg.layer_pattern)])
                                     for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("input_ids must be [batch,time] within max_seq_len")
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.config.gradient_checkpointing and self.training else block(x)
        # Tie output weights without a second state_dict alias.
        logits = F.linear(self.norm(x), self.embedding.weight)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must be pre-shifted and match input_ids")
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32, temperature=0.8):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(input_ids[:, -self.config.max_seq_len:])["logits"][:, -1].float()
            next_id = torch.multinomial(torch.softmax(logits / temperature, -1), 1)
            input_ids = torch.cat((input_ids, next_id), -1)
        return input_ids
