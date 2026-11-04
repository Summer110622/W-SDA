"""W-SDA: research-grade WAN training, not a pretrained model."""
from .config import ModelConfig, RunConfig
from .model import HybridLM

__version__ = "0.1.0"
__all__ = ["ModelConfig", "RunConfig", "HybridLM"]
