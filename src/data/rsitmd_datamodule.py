import os
import json
from typing import Any, Dict, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from PIL import Image
from transformers import AutoTokenizer
import random

from omegaconf import DictConfig
from hydra.utils import instantiate

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
        # Using a tuple (new_h, new_w) avoids the size/max_size conflict
        img = F.resize(img, (new_h, new_w), interpolation=Image.Resampling.LANCZOS)

        # 3. Calculate padding to get to exactly 224x224
        pad_w = target_w - new_w
        pad_h = target_h - new_h

        # padding is (left, top, right, bottom)
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))

        return F.pad(img, padding, fill=0, padding_mode='constant')


class RSITMDDataset(Dataset):
    """Custom Dataset for RSITMD Information Retrieval."""

    def __init__(self, root_dir: str, tokenizer: Any, transform: Optional[Any] = None, max_length: int = 77,
                 split: str = "train", test_ratio: float = 0.20):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_pairs = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory {root_dir} does not exist.")

        metadata_path = os.path.join(root_dir, "metadata.json")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        # Load the JSON file
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)

        # RSITMD structure contains a top-level "images" key
        images_list = metadata_dict.get("images", [])

        # for item in images_list:
        #     # Optional: Filter by split ('train', 'val', 'test') if needed
        #     if item.get("split") != split:
        #         continue
        #
        #     filename = item["filename"]
        #
        #     # Adjust this path logic depending on where your .tif files are stored
        #     # (e.g., if they are in an "images/" subfolder, use os.path.join(root_dir, "images", filename))
        #     full_img_path = os.path.join(root_dir, "images", filename)
        #
        #     if os.path.exists(full_img_path):
        #         # Loop through all available captions for this specific image
        #         for sentence_obj in item.get("sentences", []):
        #             caption = sentence_obj["raw"]
        #             self.data_pairs.append((full_img_path, caption))
        #     else:
        #         print(f"Warning: Image path not found: {full_img_path}")

        # Gather ALL valid image-caption pairs across the dataset first
        all_collected_pairs = []
        for item in images_list:
            filename = item["filename"]
            full_img_path = os.path.join(root_dir, "images", filename)

            if os.path.exists(full_img_path):
                for sentence_obj in item.get("sentences", []):
                    caption = sentence_obj["raw"]
                    all_collected_pairs.append((full_img_path, caption))

        # Deterministically shuffle the pairs using a fixed seed
        # This prevents train/test leakage across training epochs!
        random.Random(42).shuffle(all_collected_pairs)

        # Calculate split index
        total_samples = len(all_collected_pairs)
        test_split_idx = int(total_samples * test_ratio)

        # Assign the slices based on your target phase
        if split == "test" or split == "val":
            # The first 20% goes to evaluation
            self.data_pairs = all_collected_pairs[:test_split_idx]
        else:
            # The remaining 80% goes to training
            self.data_pairs = all_collected_pairs[test_split_idx:]

        print(f"📦 RSITMD Custom Split [{split.upper()}]: Allocated {len(self.data_pairs)} samples.")

    def __len__(self):
        return len(self.data_pairs)

    def set_train(self, mode: bool):
        self.is_training = mode

    def __getitem__(self, idx):
        img_path, caption = self.data_pairs[idx]

        try:
            # Corrected: Open the image without a context manager so it stays accessible
            image = Image.open(img_path).convert("RGB")
            # Optional: Force loading into memory immediately to catch corrupt files HERE
            image.load()

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Safeguard against infinite loops if the whole dataset is broken
            return self.__getitem__((idx + 1) % len(self.data_pairs))

        # Transformations happen on an open, valid image object
        if self.transform:
            image = self.transform(image)

        # 2. Process Text (Tokenization)
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Squeeze out the batch dimension [1, seq_len] -> [seq_len]
        input_ids = tokens.input_ids.squeeze(0)

        # If your model needs attention masks, grab it here too:
        attention_mask = tokens.attention_mask.squeeze(0)

        return image, input_ids, caption


class RSITMDDataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer: Any,
            data_dir: str = "data/RSITMD",
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = False,
            max_length: int = 24,
            test_ratio: float = 0.20,
    ) -> None:
        super().__init__()

        if isinstance(tokenizer, (dict, DictConfig)):
            from hydra.utils import instantiate
            self.tokenizer = instantiate(tokenizer)
        else:
            self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Removed 'train_val_test_split' from hparams since RSITMD specifies splits internally
        self.save_hyperparameters(logger=False, ignore=['tokenizer'])
        self.max_length = max_length
        self.test_ratio = test_ratio

        # --- TRAINING TRANSFORMS ---
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # Note: You can keep these CLIP normalization values,
            # but standard ImageNet stats ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # are also common if your vision backbone isn't CLIP.
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757]),
            transforms.RandomErasing(p=0.2),
        ])

        # --- VAL/TEST TRANSFORMS ---
        self.val_test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate datasets by mapping JSON 'test' split directly to validation."""

        # --- TRAINING & VALIDATION STAGE ---
        if stage in ("fit", "validate") or stage is None:
            # 1. Grab everything tagged "train"
            self.data_train = RSITMDDataset(
                root_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                transform=self.train_transforms,
                max_length=self.hparams.max_length,
                split="train",
                test_ratio=self.hparams.test_ratio  # Pass to dataset
            )

            # 2. Map everything tagged "test" to be your validation set
            self.data_val = RSITMDDataset(
                root_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                transform=self.val_test_transforms,
                max_length=self.hparams.max_length,
                split="test",  # Re-routing the JSON "test" rows to validation
                test_ratio=self.hparams.test_ratio  # Pass to dataset
            )

        # --- TESTING STAGE ---
        if stage == "test" or stage is None:
            # If you still run trainer.test() later, it will use the same pool
            self.data_test = RSITMDDataset(
                root_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                transform=self.val_test_transforms,
                max_length=self.hparams.max_length,
                split="test"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,  # Boosted from 4 to mirror hparams configuration safely
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
            drop_last=False,  # Changed to False: You typically don't want to drop valuation steps
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )