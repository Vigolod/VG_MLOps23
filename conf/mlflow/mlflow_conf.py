from dataclasses import dataclass


@dataclass
class MlflowConf:
    tracking_uri: str = "http://128.0.1.1:8080"
    exp_name: str = "VGMLOps23"
