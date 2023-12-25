from dataclasses import dataclass, field

from .params.train_params import TrainParams


@dataclass
class TrainConf:
    params: TrainParams = field(default_factory=TrainParams)
    data_path: str = "./data"
    train_folder: str = "train_11k"
    val_folder: str = "val"
    logging_steps: int = 10
    full_train: bool = True
