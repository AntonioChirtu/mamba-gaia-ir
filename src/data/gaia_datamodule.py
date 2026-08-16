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


class GAIADataset(Dataset):
    """Custom Dataset for GAIA Information Retrieval."""

    def __init__(self, root_dir: str, tokenizer: Any, transform: Optional[Any] = None, max_length=77):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer  # Integrated tokenizer
        self.data_pairs = []
        self.is_training = True

        self.max_length = max_length

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory {root_dir} does not exist.")

        for big_class in sorted(os.listdir(root_dir)):
            big_class_path = os.path.join(root_dir, big_class)
            if not os.path.isdir(big_class_path): continue

            for sub_class in sorted(os.listdir(big_class_path)):
                sub_class_path = os.path.join(big_class_path, sub_class)
                metadata_path = os.path.join(sub_class_path, "metadata.json")

                if os.path.isfile(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata_list = json.load(f)
                    for item in metadata_list:
                        caption = item["captions"][0]
                        img_rel_path = item["image_path"]
                        full_img_path = os.path.join(sub_class_path, img_rel_path)
                        if os.path.exists(full_img_path):
                            self.data_pairs.append((full_img_path, caption))

    def __len__(self):
        return len(self.data_pairs)

    def set_train(self, mode: bool):
        self.is_training = mode

    def __getitem__(self, idx):
        img_path, caption = self.data_pairs[idx]

        try:
            # 1. Open lazily
            with Image.open(img_path) as img:
                image = img.convert("RGB")
            #     w, h = img.size
            #     th, tw = 512, 512
            #
            #     # Check if image is actually big enough for the crop
            #     if w < tw or h < th:
            #         # If it's too small, just resize the whole thing
            #         image = img.resize((tw, th), resample=Image.Resampling.LANCZOS).convert("RGB")
            #     else:
            #         if self.is_training:
            #             i = random.randint(0, h - th)
            #             j = random.randint(0, w - tw)
            #         else:
            #             i = (h - th) // 2
            #             j = (w - tw) // 2
            #
            #         # CROP AND CONVERT inside the 'with' block
            #         # .convert("RGB") forces Pillow to actually read the pixels NOW
            #         image = img.crop((j, i, j + tw, i + th)).convert("RGB")
            #
            # # Now 'image' is a fully loaded PIL object in RAM,
            # # and it's safe that the file is closed.

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        # 2. Process Text (Tokenization)
        # We return the tokens as a tensor so Lightning can move them to the GPU
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,  # Standard for retrieval models like CLIP
            return_tensors="pt"
        )

        # We squeeze(0) because return_tensors="pt" adds a batch dimension [1, seq_len]
        # and the DataLoader will add its own batch dimension.
        return image, tokens.input_ids.squeeze(0), caption


class GAIADataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer: Any,  # Pass your model's tokenizer here
            data_dir: str = "data/GAIA",
            train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = False,
            max_length: int = 77,
    ) -> None:
        super().__init__()

        if isinstance(tokenizer, (dict, DictConfig)):
            from hydra.utils import instantiate
            # This looks at the '_target_' in the config and builds the AutoTokenizer
            self.tokenizer = instantiate(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.save_hyperparameters(logger=False, ignore=['tokenizer'])

        self.max_length = max_length

        # self.train_transforms = transforms.Compose([
        #     transforms.RandomCrop((224, 224), pad_if_needed=True),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        # ])
        #
        # self.val_test_transforms = transforms.Compose([
        #     transforms.CenterCrop((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        # ])

        # --- UPDATED TRAINING TRANSFORMS ---
        self.train_transforms = transforms.Compose([
            # scale=(0.08, 1.0) means it will take anywhere from 8% to 100% of the image area
            # ratio=(0.75, 1.33) applies slight aspect ratio stretching for robustness
            # transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
            # ResizeAndPad((224, 224)),
            transforms.RandomHorizontalFlip(),  # Highly recommended to add this for free augmentation
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757]),
            transforms.RandomErasing(p=0.2),
        ])

        # --- UPDATED VAL/TEST TRANSFORMS ---
        self.val_test_transforms = transforms.Compose([
            # Crucial: Resize the whole image so the shortest edge is 256
            # Then CenterCrop the middle 224x224.
            # If you skip Resize, a 4000px image will just yield a tiny zoomed-in center dot.
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # 1. Create the Training version (Random Crop)
        self.data_train_full = GAIADataset(
            root_dir=self.hparams.data_dir,
            tokenizer=self.tokenizer,
            transform=self.train_transforms,
            max_length=self.hparams.max_length
        )
        self.data_train_full.set_train(True)

        # 2. Create the Eval version (Center Crop)
        self.data_val_full = GAIADataset(
            root_dir=self.hparams.data_dir,
            tokenizer=self.tokenizer,
            transform=self.val_test_transforms,
            max_length=self.hparams.max_length
        )
        self.data_val_full.set_train(False)

        # 3. Use the same seed to split them so the indices match!
        dataset_size = len(self.data_train_full)
        train_len = int(self.hparams.train_val_test_split[0] * dataset_size)
        val_len = int(self.hparams.train_val_test_split[1] * dataset_size)
        test_len = dataset_size - train_len - val_len

        # Split the Training object for the train set
        self.data_train, _, _ = random_split(
            self.data_train_full, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

        # Split the Eval object for val and test
        _, self.data_val, self.data_test = random_split(
            self.data_val_full, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
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
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=2,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )