import numpy as np
import pytest
import torch
from torch.nn import functional as F

from wsda.model import HybridLM, apply_rope, delta_scan


def test_rope_norm():
    x = torch.randn(2, 3, 7, 8)
    torch.testing.assert_close(apply_rope(x, 10000).square().sum(-1), x.square().sum(-1))


@pytest.mark.parametrize("variant", ["base", "plus"])
def test_causal_and_trainable(cfg, variant):
    cfg.model.sda_variant = variant
    torch.manual_seed(7)
    model = HybridLM(cfg.model).eval()
    x = torch.randint(0, cfg.model.vocab_size, (2, 8))
    altered = x.clone()
    altered[:, 4:] = torch.randint(0, cfg.model.vocab_size, (2, 4))
    a, b = model(x)["logits"], model(altered)["logits"]
    torch.testing.assert_close(a[:, :4], b[:, :4], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(a[:, :4], model(x[:, :4])["logits"], atol=1e-6, rtol=1e-5)
    model(x, x.roll(-1, -1))["loss"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert model.blocks[1].attn.beta.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", ["base", "plus"])
def test_numpy_reference_equations(variant):
    torch.manual_seed(1)
    q = F.normalize(torch.randn(1, 1, 5, 4, dtype=torch.float64), dim=-1)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, 1, 5, 3, dtype=torch.float64)
    beta = torch.full((1, 1, 5, 1), 0.3, dtype=torch.float64)
    got = delta_scan(q, k, v, beta, capacity=3, variant=variant, threshold=0.2,
                     temperature=0.1, top_m=2).numpy()[0, 0]
    state, ref = np.zeros((4, 3)), []
    Q, K, V = (z.numpy()[0, 0] for z in (q, k, v))
    for t in range(5):
        state += 0.3 * np.outer(K[t], V[t] - K[t] @ state)
        start = max(0, t - 2)
        ak, av = K[start:t+1], V[start:t+1]
        sims = ak @ Q[t]
        fast = Q[t] @ state
        if variant == "base":
            idx = np.argmax(sims)
            out = fast + (sims[idx] >= 0.2) * (av[idx] - ak[idx] @ state)
        else:
            idx = np.argsort(sims)[-2:]
            weights = np.exp(16 * (sims[idx] - sims[idx].max()))
            archive = (weights / weights.sum()) @ av[idx]
            gate = 1 / (1 + np.exp(-(sims.max() - 0.2) / 0.1))
            out = gate * archive + (1 - gate) * fast
        ref.append(out)
    np.testing.assert_allclose(got, ref, atol=1e-12)


@pytest.mark.parametrize("variant", ["base", "plus"])
def test_delta_gradcheck(variant):
    torch.manual_seed(31)
    q, k, v = [torch.randn(1, 1, 3, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    beta = torch.full((1, 1, 3, 1), 0.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda q,k,v,b: delta_scan(q,k,v,b,capacity=2,variant=variant,
                                                            threshold=0.17, top_m=2), (q,k,v,beta))


def test_exact_key_residual():
    k = F.normalize(torch.randn(1, 2, 8, 4), dim=-1)
    v = torch.randn(1, 2, 8, 3)
    result = delta_scan(k, k, v, torch.full((1, 2, 8, 1), 0.5), capacity=8, threshold=0.99)
    torch.testing.assert_close(result, v)


def test_checkpoint_and_fake_quant(cfg):
    cfg.model.sda_variant = "plus"
    cfg.model.archive_fake_quant = True
    model = HybridLM(cfg.model).train()
    x = torch.randint(0, cfg.model.vocab_size, (1, 8))
    plain = model(x, x)["loss"]
    cfg.model.gradient_checkpointing = True
    checked = model(x, x)["loss"]
    torch.testing.assert_close(plain, checked)
    checked.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_duplicate_key_base_prefers_latest():
    k = torch.tensor([[[[1., 0.], [1., 0.]]]])
    v = torch.tensor([[[[3.], [9.]]]])
    result = delta_scan(k, k, v, torch.full((1, 1, 2, 1), 0.3), capacity=4)
    torch.testing.assert_close(result, v)
