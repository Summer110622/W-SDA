import torch

from wsda.data import TokenDataset, Tokenizer, prepare
from wsda.model import HybridLM
from wsda.worker import train_local


def test_tokenizer_utf8():
    tok = Tokenizer()
    text = "日本語 RoPE + SDA + NoPE"
    assert tok.decode(tok.encode(text)) == text


def test_shards_and_actual_backward(tmp_path, cfg):
    corpus = tmp_path / "text.txt"
    corpus.write_text("summer trains a language model.\n" * 300)
    root = tmp_path / "data"
    prepare(corpus, root)
    a = TokenDataset(root, cfg.seq_len, shard_index=0, num_shards=2)
    b = TokenDataset(root, cfg.seq_len, shard_index=1, num_shards=2)
    assert not set(a.blocks) & set(b.blocks)
    model = HybridLM(cfg.model)
    base = {k: v.clone() for k,v in model.state_dict().items()}
    delta, metrics = train_local(model, base, a, cfg, steps=2, batch_size=2,
                                grad_accum=2, device=torch.device("cpu"))
    assert metrics["tokens"] == 2 * 2 * 2 * cfg.seq_len
    assert sum(float(x.square().sum()) for x in delta.values()) > 0
    assert all(torch.isfinite(x).all() for x in delta.values())
