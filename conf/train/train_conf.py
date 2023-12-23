from dataclasses import dataclass


@dataclass
class TrainConf:
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 20
    weight_decay: float = 0.0
    data_path: str = "./data"
    train_folder: str = "train_11k"
    val_folder: str = "val"
    save_path: str = "mymodel_ckpt.pt"
    full_train: bool = True
