from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision
from conf.config import Config
from conf.train.train_conf import TrainConf
from model import MyModel, get_model
from safetensors.torch import save_model
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model: MyModel, cfg: TrainConf):
        self.cfg = cfg
        self.model = model

    def setup(self):
        self._setup_loaders()
        self._setup_optimizer()

    def _setup_loaders(self):
        data_path = Path(self.cfg.data_path)
        train_path = data_path / self.cfg.train_folder
        val_path = data_path / self.cfg.val_folder

        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transform)
        val_dataset = torchvision.datasets.ImageFolder(val_path, transform=transform)

        full_dataset = ConcatDataset([train_dataset, val_dataset])

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True
        )
        self.full_loader = DataLoader(
            full_dataset, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True
        )

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def train_epoch(self, epoch: int):
        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        loader = self.full_loader if self.cfg.full_train else self.train_loader
        for images, labels in tqdm(loader, desc=f"Training epoch #{epoch}"):
            images = images.to(self.model.device)
            labels = labels.to(self.model.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            self.optimizer.step()

    def validation_epoch(self, epoch: int):
        if not self.cfg.full_train:
            accuracies = []
            self.model.eval()
            with torch.no_grad():
                for images, labels in tqdm(
                    self.val_loader, desc=f"Validation epoch #{epoch}"
                ):
                    images = images.to(self.model.device)
                    labels = labels.to(self.model.device)
                    outputs = self.model(images)
                    predictions = (outputs > 0).long()
                    accuracy = (predictions == labels).sum() / labels.size(0)
                    accuracies.append(accuracy.item())
            print(f"Val accuracy: {np.mean(accuracies):.3f}")

    def train(self):
        for epoch in range(self.cfg.num_epochs):
            self.train_epoch(epoch)
            self.validation_epoch(epoch)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config):
    model = get_model(cfg.model)
    trainer = Trainer(model, cfg.train)
    trainer.setup()
    trainer.train()
    save_model(model, cfg.save_path)


if __name__ == "__main__":
    main()
