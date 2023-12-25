from dataclasses import dataclass, field

from .mlflow.mlflow_conf import MlflowConf
from .model.model_conf import ModelConf
from .train.train_conf import TrainConf


@dataclass
class Config:
    model: ModelConf = field(default_factory=ModelConf)
    train: TrainConf = field(default_factory=TrainConf)
    mlflow: MlflowConf = field(default_factory=MlflowConf)
    model_save_path: str = "mymodel.safetensors"
    infer_data_path: str = "data/test_labeled"
    infer_batch_size: int = 32
    predict_save_path: str = "predictions.csv"
