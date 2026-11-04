"""Actual blockwise int8 WAN payloads. No pickle or torch.load on the network."""
from __future__ import annotations

import json
import math
import struct

import torch
from safetensors.torch import load, save


def metadata(blob: bytes) -> dict:
    if len(blob) < 8:
        raise ValueError("truncated safetensors")
    n = struct.unpack("<Q", blob[:8])[0]
    if n > 4 * 1024 * 1024 or n > len(blob) - 8:
        raise ValueError("invalid safetensors header")
    header = json.loads(blob[8:8 + n])
    return json.loads(header.get("__metadata__", {}).get("wsda", "{}"))


def pack(tensors: dict[str, torch.Tensor], info: dict, mode: str = "int8", block: int = 4096) -> bytes:
    if mode not in {"int8", "fp32"} or block < 1:
        raise ValueError("invalid wire codec")
    payload, shapes = {}, {}
    for name in sorted(tensors):
        x = tensors[name].detach().cpu().float().contiguous()
        if not torch.isfinite(x).all():
            raise ValueError(f"non-finite tensor: {name}")
        shapes[name] = list(x.shape)
        if mode == "fp32":
            payload[name] = x.clone()
        else:
            flat = x.flatten()
            padding = (-flat.numel()) % block
            padded = torch.nn.functional.pad(flat, (0, padding)).view(-1, block)
            scale = (padded.abs().amax(-1) / 127).clamp_min(1e-12)
            payload[name + ".q"] = (padded / scale[:, None]).round().clamp(-127, 127).to(torch.int8).flatten()[:flat.numel()].contiguous()
            payload[name + ".s"] = scale
    meta = {"protocol": 1, "codec": mode, "block": block, "shapes": shapes, "info": info}
    return save(payload, metadata={"wsda": json.dumps(meta, allow_nan=False, separators=(",", ":"))})


def unpack(blob: bytes, expected: dict[str, torch.Tensor] | None = None):
    meta = metadata(blob)
    if meta.get("protocol") != 1 or meta.get("codec") not in {"int8", "fp32"}:
        raise ValueError("unsupported protocol/codec")
    shapes, block = meta["shapes"], meta["block"]
    if type(block) is not int or not 1 <= block <= 65536:
        raise ValueError("invalid block size")
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("invalid tensor schema")
    if expected is not None:
        if set(shapes) != set(expected) or any(shapes[k] != list(v.shape) for k, v in expected.items()):
            raise ValueError("parameter names/shapes do not match")
    tensors = load(blob)
    output = {}
    for name, shape in shapes.items():
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid shape")
        n = math.prod(shape)
        if meta["codec"] == "fp32":
            x = tensors[name]
            if x.dtype != torch.float32 or list(x.shape) != shape:
                raise ValueError("invalid fp32 tensor")
        else:
            q, scales = tensors[name + ".q"], tensors[name + ".s"]
            if q.dtype != torch.int8 or scales.dtype != torch.float32 or q.ndim != 1 or scales.ndim != 1:
                raise ValueError("invalid quantized tensor types")
            if q.numel() != n or scales.numel() != (n + block - 1) // block:
                raise ValueError("invalid quantized tensor length")
            if not torch.isfinite(scales).all() or not (scales > 0).all():
                raise ValueError("invalid quantization scales")
            x = (q.float() * scales.repeat_interleave(block)[:n]).reshape(shape)
        if not torch.isfinite(x).all():
            raise ValueError("non-finite decoded tensor")
        output[name] = x
    wanted = set(shapes) if meta["codec"] == "fp32" else {k + suffix for k in shapes for suffix in (".q", ".s")}
    if set(tensors) != wanted:
        raise ValueError("extra or missing tensors")
    return output, meta["info"]
