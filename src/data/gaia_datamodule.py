import os
import glob
import json
from typing import Any, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
import random

from omegaconf import DictConfig

# Standard for many base Mamba models
Image.MAX_IMAGE_PIXELS = None

import torchvision.transforms.functional as F


SPHERE_ANCORS = {
        'Atmosphere': ['dust_storm', 'cyclon', 'typhoon', 'hurricane', 'tornado', 'smoke_plume', 'aerosol', 'cloud',
                       'ash', 'haze', 'meteorology', 'weather_pattern'],
        'Hydrosphere': ['water', 'ocean', 'sea', 'river', 'lake', 'flood', 'delta', 'estuar', 'phytoplankton', 'algae',
                        'marine', 'coast', 'reef', 'reservoir', 'currents', 'hydrology', 'tidal'],
        'Biosphere': ['agri', 'farm', 'crop', 'forest', 'veget', 'plant', 'fire', 'burn', 'wildfire', 'ndvi', 'paddy',
                      'tree', 'leaf', 'flora', 'harvest', 'deforest', 'ecology', 'habitat', 'irrigat', 'cultiv',
                      'plantation', 'orchard'],
        'Cryosphere': ['ice', 'snow', 'glacier', 'polar', 'arctic', 'freeze', 'frost', 'permafrost', 'antarct',
                       'iceberg', 'shelf', 'meltwater'],
        'Geosphere': ['geology', 'mining', 'volcan', 'earthq', 'soil', 'mountain', 'topography', 'urban', 'city',
                      'plateau', 'desert', 'land_use', 'land_management', 'geography']
    }

def classify_to_sphere(image_tags):
    tag_blob = " ".join([str(t).lower() for t in image_tags])

    scores = {sphere: 0 for sphere in SPHERE_ANCORS}

    for sphere, stems in SPHERE_ANCORS.items():
        for stem in stems:
            if stem in tag_blob:
                scores[sphere] += 2

                # --- STRATEGIC WEIGHTING ---
    if scores['Biosphere'] > 0: scores['Biosphere'] += 5  # Maximum protection for your top interest
    if scores['Hydrosphere'] > 0: scores['Hydrosphere'] += 2
    if scores['Atmosphere'] > 0: scores['Atmosphere'] += 1

    # Geosphere Tax: Only wins if it's the ONLY clear signal
    if scores['Geosphere'] > 0: scores['Geosphere'] -= 2

    top_sphere = max(scores, key=scores.get)

    if scores[top_sphere] <= 0:
        if any(c in tag_blob for c in ['cold', 'winter', 'degree']):
            return 'Cryosphere'
        return 'Geosphere'
    return top_sphere


class ResizeAndPad:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        target_w, target_h = self.target_size

        # 1. Calculate the scaling factor to make the longest edge fit
        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        # 2. Resize to the new dimensions
        img = F.resize(img, (new_h, new_w), interpolation=Image.Resampling.LANCZOS)

        # 3. Calculate padding to get to exactly 224x224
        pad_w = target_w - new_w
        pad_h = target_h - new_h

        # padding is (left, top, right, bottom)
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))

        return F.pad(img, padding, fill=0, padding_mode='constant')


class GAIADataset(Dataset):
    """Custom Dataset for GAIA Information Retrieval.
    
    Each record is ``(image_path, captions)`` where ```captions`` is the list of
    synthetic captions GAIA provides per image. During training, a caption is
    sampled at random each epoch (free text-side augmentation); during
    validation/testing the first caption is used deterministically
    """

    def __init__(
        self,
        records: List[Tuple[str, List[str]]],
        tokenizer: Any, 
        transform: Optional[Any] = None, 
        max_length: int = 128,
        is_training: bool = True
    ):
        self.records = records
        self.transform = transform
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.is_training = is_training

    def __len__(self):
        return len(self.records)

    def set_train(self, mode: bool):
        self.is_training = mode

    def _pick_caption(self, captions: List[str]) -> str:
        if not captions:
            return ""
        if self.is_training:
            return random.choice(captions)
        return captions[0]

    def __getitem__(self, idx):
        img_path, captions = self.records[idx]

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        caption = self._pick_caption(captions)

        # Tokenize. attention_mask is returned so the model can pool the last
        # non-pad toke (the tokenizer right-pads to max_length)
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)
        return image, input_ids, attention_mask, caption


class GAIADataModule(LightningDataModule):
    """ GAIA datamodule reading img2dataset's ``files`` output layout:
    ``data_dir/{train,val,test}/<shard>/<sample>.{png,json,txt}``, donwloaded directly
    from GAIA's official json split files. Split membership
    is therefore inherent to which folder a sample lives in (no id-matching needed).
    Each sample's json sidecar carries its ``id``, ``captions``, ``tag`` and downloaded ``status``.
    """
    def __init__(
            self,
            tokenizer: Any,
            data_dir: str = "data/GAIA",
            spheres: Optional[List[str]] = None,
            batch_size: int = 128,
            num_workers: int = 4,
            pin_memory: bool = False,
            max_length: int = 128,
    ) -> None:
        super().__init__()

        if isinstance(tokenizer, (dict, DictConfig)):
            from hydra.utils import instantiate
            self.tokenizer = instantiate(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.save_hyperparameters(logger=False, ignore=['tokenizer'])

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757]),
            transforms.RandomErasing(p=0.2),
        ])

        self.val_test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _scan_split(self, split: str) -> List[Tuple[str, List[str]]]:
        """Walk data_dir/<split>/<shard>/*.png -> [(full_path, caption)]."""
        splits_dir = os.path.join(self.hparams.data_dir, split)
        if not os.path.isdir(splits_dir):
            raise FileNotFoundError(f"Split directory {root} does not exist.")

        records: List[Tuple[str, List[str]]] = []
        n_failed, n_off_sphere = 0, 0
        for img_path in sorted(glob.glob(os.path.join(splits_dir, "*", "*.png"))):
            json_path = os.path.splitext(img_path)[0] + ".json"
            if not os.path.isfile(json_path):
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            if meta.get("status") != "success":
                n_failed += 1
                continue

            captions = meta.get("captions") or []
            if not captions:
                continue

            if self.hparams.spheres:
                sphere = classify_to_sphere(meta.get("tag") or [])
                if sphere not in self.hparams.spheres:
                    n_off_sphere += 1
                    continue
            
            records.append((img_path, captions))

        if n_failed:
            print(f"GAIA {split}: skipped {n_failed} non-success downloads.")
        if n_off_sphere:
            print(f"GAIA {split}: skipped {n_off_sphere} images outside spheres={self.hparams.spheres}.")
        return records
                    

    # ------------------------------------------------------------------ #
    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None:
            return

        train_records = self._scan_split("train")
        val_records = self._scan_split("val")
        test_records = self._scan_split("test")

        for name, records in (("train", train_records), ("val", val_records), ("test", test_records)):
            print(f"GAIA {name}: {len(records)} image-text pairs.")

        self.data_train = GAIADataset(
            train_records, self.tokenizer, self.train_transforms,
            self.hparams.max_length, is_training=True,
        )
        self.data_val = GAIADataset(
            val_records, self.tokenizer, self.val_test_transforms,
            self.hparams.max_length, is_training=False,
        )
        self.data_test = GAIADataset(
            test_records, self.tokenizer, self.val_test_transforms,
            self.hparams.max_length, is_training=False,
        )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=2,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )