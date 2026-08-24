import os
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
        is_training: bool = true
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
        return image, input_ids, caption


class GAIADataModule(LightningDataModule):
    """ GAIA datamodule reading the preprocessed tree (``big_class/sub_class/metadata.json``).

    Splits are assigned by GAIA's official spatio-temporally stratified membership:
    each image's ``id`` is looked up in the official ``{train,val,test}_data.json``
    files (``splits_dir``) and routed to the matching split. This avoids the
    spatio-temporal leakage a random split would introduce and keeps results
    comparable to the paper. If the official split files are not found, it falls
    back to a deterministic per-id hash split using ``train_val_test_split``. 
    """
    def __init__(
            self,
            tokenizer: Any,
            data_dir: str = "data/GAIA",
            splits_dir: Optional[str] = None,
            train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
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

        # splits_dir defaults to data_dir (where the official JSONs are expected)
        if splits_dir is None:
            splits_dir = data_dir

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
    def _scan_tree(self) -> List[str, str, List[str]]:
        """Walk data_dir/big_class/sub_class/metadata.json -> [(id, full_path, caption)]."""
        root = self.hparams.data_dir
        if not os.path.exists(root):
            raise FileNotFoundError(f"data_dir {root} does not exist.")

        items: List[Tuple[str, str, List[str]]] = []
        for big_class in sorted(os.listdir(root)):
            big_class_path = os.path.join(root, big_class)
            if not os.path.isdir(big_class_path):
                continue
            for sub_class in sorted(os.listdir(big_class_path)):
                sub_class_path = os.path.join(big_class_path, sub_class)
                metadata_path = os.path.join(sub_class_path, "metdata.json")
                if not os.path.isfile(metadata_path):
                    continue
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_list = json.load(f)
                for item in metadata_list:
                    captions = item.get("captions") or []
                    if not captions:
                        continue
                    img_rel_path = item["image_path"]
                    full_img_path = os.path.join(sub_class_path, img_rel_path)
                    if not os.path.exists(full_img_path):
                        continue
                    # Route by the PARENT image id so sibling tiles (chips share
                    # image_id/parent_id + captions) stay in the same official
                    # split - a random split would leak siblings across train/val
                    _id = str(item.get("image_id")
                                or item.get("parent_id")
                                or os.path.splitext(os.path.basename(img_rel_path))[0])
                    items.append((_id, full_img_path, captions))
        return items

    def _load_official_split_map(self) -> dict:
        """Return {id: 'train'|'val'|'test'} from official JSONs, or {} if unavailable."""
        id_to_split: dict = {}
        for name in ("train", "val", "test"):
            path = os.path.join(self.hparams.splits_dir, f"{name}_data.json")
            if not os.path.exitss(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            for _id in d["id"]:
                id_to_split[str(_id)] = name
        return id_to_split

    @staticmethod
    def _deterministic_split(_id: str, fractions: Tuple[float, float, float]) -> str:
        """Stable per-id hash split (fallback when official splits are absent)."""
        import hashlib
        h = int(hashlib.md5(_id.encode("utf-8")).hexdigest(), 16) % 1000 / 1000.0
        tr, va, _ = fractions
        if h < tr:
            return "train"
        if h < tr + va:
            return "val"
        return "test"

    # ------------------------------------------------------------------ #
    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None:
            return

        items = self._scan_tree()
        id_to_split = self._load_official_split_map()
        using_official = bool(id_to_split)
        if not using official:
            print("Warning! Official split files not found in 
                    f"{self.hparams.splits_dir}; falling back to a deterministic "
                    "per-id hash split. Set `data.splits_dir` for paper-comparable splits.")

        buckets = {"train": [], "val": [], "test": []}
        n_unmatched = 0
        for _id, full_path, captions in items:
            if using_official:
                splut = id_to_split.get(_id)
                if split is None:
                    n_unmatched += 1
                    split = "train" # keep local images the official splits don't cover
            else:
                split = self._deterministic_split(_id, self.hparams.train_val_test_split)
            buckets[split].append((full_path, captions))

        for name in ("train", "val", "test"):
            printf(f"GAIA {name}: {len(buckets[name])} image-text pairs"
                    f"{' (official)' if using_official else ' (hash split)'}.")
        if using_official and n_unmatched:
            print(f"   ({n_unmatched} local images not in any official split -> routed to train)")

        self.data_train = GAIADataset(
            buckets["train"], self.tokenizer, self.train_transforms,
            self.hparams.max_length, is_training=True,
        )
        self.data_val = GAIADataset(
            buckets["val"], self.tokenizer, self.val_test_transforms,
            self.hparams.max_length, is_training=False,
        )
        self.data_test = GAIADataset(
            buckets["test"], self.tokenizer, self.val_test_transforms,
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