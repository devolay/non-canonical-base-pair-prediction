from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class ModelConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    min_lr: float = 1e-8
    log_neptune: bool = False
    negative_sample_ratio: int = 1
    freeze_embeddings: bool = False

    gradient_clip_value: float = 0.0
    gradient_clip_algorithm: str = "norm"
    
    in_channels: int = 4
    model_type: str = "local"
    gnn_channels: List[int] = field(default_factory=lambda: [64, 64])
    cnn_head_embed_dim: int = 64
    cnn_head_num_blocks: int = 2
    dropout: float = 0.0
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)