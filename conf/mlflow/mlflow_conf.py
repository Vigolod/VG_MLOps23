from dataclasses import dataclass


@dataclass
class MlflowConf:
    tracking_uri: str = "http://127.0.0.1:8080"
    exp_name: str = "VGMLOps23"
