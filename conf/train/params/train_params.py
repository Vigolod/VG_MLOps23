from dataclasses import dataclass


@dataclass
class TrainParams:
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 20
    weight_decay: float = 0.0
