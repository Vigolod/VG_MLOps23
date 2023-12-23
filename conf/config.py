from dataclasses import dataclass, field

from .model.model_conf import ModelConf
from .train.train_conf import TrainConf


@dataclass
class Config:
    model: ModelConf = field(default_factory=ModelConf)
    train: TrainConf = field(default_factory=TrainConf)
    save_path: str = "checkpoints/mymodel.safetensors"
