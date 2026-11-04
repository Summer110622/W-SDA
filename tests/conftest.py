import pytest
import torch

from wsda import ModelConfig, RunConfig


def pytest_sessionstart(session):
    torch.set_num_threads(1)


@pytest.fixture
def cfg():
    return RunConfig(model=ModelConfig(d_model=16, num_heads=2, num_layers=3,
                                      intermediate_size=32, max_seq_len=16, archive_capacity=4),
                     seq_len=8, local_steps=2, max_local_steps=8, target_seconds=1)
