from dataclasses import dataclass


@dataclass
class ModelConf:
    dropout: float = 0.2
    device: str = "cuda:0"
