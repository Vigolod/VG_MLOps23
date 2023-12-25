import os
from pathlib import Path

import hydra
import pandas as pd
import torch
from conf.config import Config
from model import MyModel, get_model
from PIL import Image
from safetensors.torch import load_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class TestDataset(Dataset):
    def __init__(self, path: str, transform):
        path = Path(path)
        self.cat_path = path / "cat"
        self.dog_path = path / "dog"
        self.length = len(os.listdir(self.cat_path)) + len(os.listdir(self.dog_path))
        self.transform = transform
        self.offset = 1

    def __getitem__(self, index):
        img_found = False
        while not img_found:
            idx = index + self.offset
            cat_file = self.cat_path / f"cat.{idx}.jpg"
            dog_file = self.dog_path / f"dog.{idx}.jpg"
            if os.path.isfile(cat_file):
                img = Image.open(cat_file)
                label = 0
                img_found = True
            elif os.path.isfile(dog_file):
                img = Image.open(dog_file)
                label = 1
                img_found = True
            else:
                self.offset += 1
        return idx, self.transform(img), label

    def __len__(self):
        return self.length


def inference(model: MyModel, loader: DataLoader):
    model.eval()
    predictions = []
    index = []
    with torch.inference_mode():
        for ids, images, _ in tqdm(loader, desc="Predicting..."):
            images = images.to(model.device)
            output = model(images)
            batch_pred = (output > 0).long().detach().cpu().numpy()
            predictions.extend(batch_pred)
            ids = ids.numpy()
            index.extend(ids)
    return index, predictions


def save_predictions(index: list[int], predictions: list[int], save_path: str):
    predictions: pd.Series = pd.Series(data=predictions, index=index, name="label")
    id_to_label = {0: "cat", 1: "dog"}
    predictions = predictions.map(id_to_label)
    predictions.to_csv(save_path, index_label="image_id")


def get_loader(cfg: Config):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = TestDataset(cfg.infer_data_path, transform)
    loader = DataLoader(
        dataset, batch_size=cfg.infer_batch_size, shuffle=False, pin_memory=True
    )
    return loader


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config):
    model = get_model(cfg.model)
    load_model(model, cfg.model_save_path)
    loader = get_loader(cfg)
    index, predictions = inference(model, loader)
    save_predictions(index, predictions, cfg.predict_save_path)


if __name__ == "__main__":
    main()
