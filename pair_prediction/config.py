from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class ModelConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    log_neptune: bool = False
    negative_sample_ratio: int = 1
    
    in_channels: int = 4
    model_type: str = "local"
    gnn_channels: List[int] = field(default_factory=lambda: [64, 64])
    cnn_channels: Optional[List[int]] = field(default=None)
    dropout: float = 0.0
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)