from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class ModelConfig:
    # Training
    log_neptune: bool = False
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    min_lr: float = 1e-8
    
    # Negative Sampling
    negative_sample_ratio: int = 1
    hard_negative_sampling: bool = False
    hard_negative_sampling_temperature: float = 1.0
    
    # Gradient Clipping
    use_gradient_clipping: bool = False
    gradient_clip_value: float = 0.0
    gradient_clip_algorithm: str = "norm"

    # Scheduler
    use_scheduler: bool = False
    scheduler_patience: int = 10
    
    # Architecture
    model_type: str = "local"
    freeze_embeddings: bool = False
    in_channels: int = 4
    gnn_channels: List[int] = field(default_factory=lambda: [64, 64])
    out_channels: int = 64
    cnn_head_embed_dim: int = 64
    cnn_head_num_blocks: int = 2
    kernel_size: int = 3
    dropout: float = 0.0
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)