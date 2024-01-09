from conf.config import ModelConf
from torch import nn


def get_conv(in_channels: int, out_channels: int):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
    )


class MyModel(nn.Module):
    def __init__(self, cfg: ModelConf):
        super().__init__()

        self.device = cfg.device
        self.net = nn.Sequential(
            get_conv(3, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            get_conv(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            get_conv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            get_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        return self.net(batch).view(-1)


def get_model(cfg: ModelConf):
    model = MyModel(cfg)
    return model.to(cfg.device)
