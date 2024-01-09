from dataclasses import dataclass, field


@dataclass
class MlflowConf:
    tracking_uri: str = "http://128.0.1.1:8080"
    exp_name: str = "VGMLOps23"


@dataclass
class ModelConf:
    dropout: float = 0.2
    device: str = "cuda:0"


@dataclass
class TrainParams:
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 20
    weight_decay: float = 0.0


@dataclass
class TrainConf:
    params: TrainParams = field(default_factory=TrainParams)
    data_path: str = "./data"
    train_folder: str = "train_11k"
    val_folder: str = "val"
    logging_steps: int = 10
    full_train: bool = True


@dataclass
class Config:
    model: ModelConf = field(default_factory=ModelConf)
    train: TrainConf = field(default_factory=TrainConf)
    mlflow: MlflowConf = field(default_factory=MlflowConf)
    model_save_path: str = "mymodel.safetensors"
    infer_data_path: str = "data/test_labeled"
    infer_batch_size: int = 32
    predict_save_path: str = "predictions.csv"
