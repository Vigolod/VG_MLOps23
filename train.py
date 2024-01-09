from pathlib import Path

import hydra
import mlflow
import torch
import torchvision
from conf.config import Config, TrainConf
from dvc.repo import Repo
from mlflow.utils.git_utils import get_git_commit
from model import MyModel, get_model
from omegaconf import OmegaConf
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
            train_dataset,
            batch_size=self.cfg.params.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.params.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        self.full_loader = DataLoader(
            full_dataset,
            batch_size=self.cfg.params.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.params.learning_rate,
            weight_decay=self.cfg.params.weight_decay,
        )

    def train_epoch(self, epoch: int):
        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        loader = self.full_loader if self.cfg.full_train else self.train_loader
        accum_loss = 0
        for i, (images, labels) in tqdm(
            enumerate(loader), desc=f"Training epoch #{epoch}", total=len(loader)
        ):
            images = images.to(self.model.device)
            labels = labels.to(self.model.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels.float())
            accum_loss += loss.item()
            if (i + 1) % self.cfg.logging_steps == 0:
                mlflow.log_metric("train_bce_loss", accum_loss / self.cfg.logging_steps)
                accum_loss = 0
            loss.backward()
            self.optimizer.step()

    def validation_epoch(self, epoch: int):
        if not self.cfg.full_train:
            criterion = nn.BCEWithLogitsLoss()
            self.model.eval()
            accum_loss = 0
            accum_accuracy = 0
            with torch.no_grad():
                for i, (images, labels) in tqdm(
                    enumerate(self.val_loader),
                    desc=f"Validation epoch #{epoch}",
                    total=len(self.val_loader),
                ):
                    images = images.to(self.model.device)
                    labels = labels.to(self.model.device)
                    outputs = self.model(images)
                    predictions = (outputs > 0).long()
                    accuracy = (predictions == labels).sum() / labels.size(0)
                    loss = criterion(outputs, labels.float())
                    accum_accuracy += accuracy.item()
                    accum_loss += loss.item()
                    if (i + 1) % self.cfg.logging_steps == 0:
                        metrics = {
                            "eval_accuracy": accum_accuracy / self.cfg.logging_steps,
                            "eval_bce_loss": accum_loss / self.cfg.logging_steps,
                        }
                        mlflow.log_metrics(metrics)
                        accum_accuracy = accum_loss = 0

    def train(self):
        for epoch in range(self.cfg.params.num_epochs):
            self.train_epoch(epoch)
            self.validation_epoch(epoch)


def download_data():
    repo = Repo(".")
    repo.pull()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config):
    download_data()

    mlflow.set_tracking_uri(uri=cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.exp_name)

    model = get_model(cfg.model)
    trainer = Trainer(model, cfg.train)
    trainer.setup()

    with mlflow.start_run():
        commit_id = get_git_commit(Path.cwd())
        mlflow.set_tag("Git commit", commit_id)
        params = OmegaConf.to_container(cfg.model)
        params.update(OmegaConf.to_container(cfg.train.params))
        mlflow.log_params(params)
        trainer.train()

    save_model(model, cfg.model_save_path)


if __name__ == "__main__":
    main()
