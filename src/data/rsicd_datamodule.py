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
import pandas as pd
import ast

from omegaconf import DictConfig
from hydra.utils import instantiate

# Standard for many base Mamba models
Image.MAX_IMAGE_PIXELS = None

import torchvision.transforms.functional as F

class RSICDDataset(Dataset):
    """Custom Dataset for RSICD Information Retrieval."""

    def __init__(self, root_dir: str, tokenizer: Any, transform: Optional[Any] = None, max_length: int = 77,
                 split: str = "train"):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_pairs = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory {root_dir} does not exist.")

        csv_files_to_load = []

        if split == "train":
            csv_files_to_load.append(os.path.join(root_dir, "train.csv"))
        elif split in ["val", "test", "combined_val"]:
            # If we want the combined validation set, we load BOTH csv targets
            # csv_files_to_load.append(os.path.join(root_dir, "train.csv"))
            csv_files_to_load.append(os.path.join(root_dir, "val.csv"))
            csv_files_to_load.append(os.path.join(root_dir, "test.csv"))
        else:
            raise ValueError(f"Unknown split type: {split}")
        
        dfs = []
        for csv_path in csv_files_to_load:
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"Required CSV split file not found at {csv_path}")
            dfs.append(pd.read_csv(csv_path))
        
        # Concatenate columns seamlessly 
        combined_df = pd.concat(dfs, ignore_index=True)


        # 3. Iterate through rows and collect valid file paths + text descriptions
        # Using .itertuples() is significantly faster than standard pandas .iterrows()
        for row in combined_df.itertuples(index=False):
            # Fallback strings if columns are empty or malformed
            filename = getattr(row, "filename", None)
            captions = getattr(row, "captions", None)
            
            if filename and pd.notna(captions):
                full_img_path = os.path.join(root_dir, str(filename))
                
                # Guard verification: Ensure the local image file actually exists on your storage drive
                if os.path.exists(full_img_path):
                    # self.data_pairs.append((full_img_path, str(caption)))
                    try:
                        caption_list = ast.literal_eval(captions)
                        if not isinstance(caption_list, list):
                            caption_list = [captions]
                    except (ValueError, SyntaxError):
                        # Fallback if a row happens to be a single plain string instead of a list string
                        caption_list = [captions]

                    for caption in caption_list:
                        if pd.notna(caption) and str(caption).strip():
                            self.data_pairs.append((full_img_path, str(caption)))

        # 4. Optional: Shuffle the pairs deterministically 
        # (Great practice for validation tracking stability across steps)
        random.Random(42).shuffle(self.data_pairs)

        print(f"📦 Custom Split [{split.upper()}]: Processed {len(csv_files_to_load)} CSV(s). "
            f"Allocated {len(self.data_pairs)} valid image-caption samples.")


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


class RSICDDataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer: Any,
            data_dir: str = "data/RSICD",
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = False,
            max_length: int = 24,
    ) -> None:
        super().__init__()

        if isinstance(tokenizer, (dict, DictConfig)):
            from hydra.utils import instantiate
            self.tokenizer = instantiate(tokenizer)
        else:
            self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Removed 'train_val_test_split' from hparams since RSICD specifies splits internally
        self.save_hyperparameters(logger=False, ignore=['tokenizer'])
        self.max_length = max_length

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
            self.data_train = RSICDDataset(
                root_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                transform=self.train_transforms,
                max_length=self.hparams.max_length,
                split="train",
            )

            # 2. Map everything tagged "test" to be your validation set
            self.data_val = RSICDDataset(
                root_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                transform=self.val_test_transforms,
                max_length=self.hparams.max_length,
                split="test",  # Re-routing the JSON "test" rows to validation
            )

        # --- TESTING STAGE ---
        if stage == "test" or stage is None:
            # If you still run trainer.test() later, it will use the same pool
            self.data_test = RSICDDataset(
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