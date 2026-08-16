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
import gc

# Standard for many base Mamba models
tokenizer_global = AutoTokenizer.from_pretrained("eleutherai/gpt-neox-20b")
# Image.MAX_IMAGE_PIXELS = None


class GAIADataset(Dataset):
    """Custom Dataset for GAIA Information Retrieval."""

    def __init__(self, data_pairs: List[Tuple[str, str]], tokenizer: Any, transform: Optional[Any] = None):
        # self.root_dir = root_dir
        self.data_pairs = data_pairs
        self.transform = transform
        self.tokenizer = tokenizer_global  # Integrated tokenizer
        self.is_training = True

    def __len__(self):
        return len(self.data_pairs)

    def set_train(self, mode: bool):
        self.is_training = mode

    def __getitem__(self, idx):
        img_path, caption = self.data_pairs[idx]

        try:
            # 1. Open lazily
            with Image.open(img_path) as img:
                w, h = img.size
                th, tw = 512, 512

                # Check if image is actually big enough for the crop
                if w < tw or h < th:
                    # If it's too small, just resize the whole thing
                    image = img.resize((tw, th), resample=Image.Resampling.LANCZOS).convert("RGB")
                else:
                    if self.is_training:
                        i = random.randint(0, h - th)
                        j = random.randint(0, w - tw)
                    else:
                        i = (h - th) // 2
                        j = (w - tw) // 2

                    # CROP AND CONVERT inside the 'with' block
                    # .convert("RGB") forces Pillow to actually read the pixels NOW
                    image = img.crop((j, i, j + tw, i + th)).convert("RGB")

            # Now 'image' is a fully loaded PIL object in RAM,
            # and it's safe that the file is closed.

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
            max_length=77,  # Standard for retrieval models like CLIP
            return_tensors="pt"
        )

        img.close()

        # We squeeze(0) because return_tensors="pt" adds a batch dimension [1, seq_len]
        # and the DataLoader will add its own batch dimension.
        input_ids = tokens.input_ids.squeeze(0).clone()
        del tokens
        return image, input_ids, caption


class GAIADataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer: Any,  # Pass your model's tokenizer here
            data_dir: str = "data/GAIA",
            train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['tokenizer'])
        self.tokenizer = tokenizer

        self.train_transforms = transforms.Compose([
            transforms.RandomCrop((224, 224), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        ])

        self.val_test_transforms = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # 1. Gather all data pairs first
        all_pairs = []
        # We need a dictionary to group chips by their parent
        # Parent is 'massive_image_1' if the chip is 'massive_image_1_x0_y0.jpg'
        parent_to_chips = {}
        # print(f"Scanning directory: {self.hparams.data_dir}")  # DEBUG

        for big_class in sorted(os.listdir(self.hparams.data_dir)):
            # print(f"Checking class: {big_class}")  # DEBUG
            big_class_path = os.path.join(self.hparams.data_dir, big_class)
            if not os.path.isdir(big_class_path): continue

            for sub_class in sorted(os.listdir(big_class_path)):
                sub_class_path = os.path.join(big_class_path, sub_class)
                metadata_path = os.path.join(sub_class_path, "metadata.json")

                # print(f"Found metadata at: {metadata_path}")  # DEBUG

                # if os.path.isfile(metadata_path):
                #     with open(metadata_path, 'r', encoding='utf-8') as f:
                #         metadata_list = json.load(f)
                #
                #     for item in metadata_list:
                #         img_rel_path = item["image_path"]
                #         # Ensure we aren't accidentally joining an absolute path
                #         full_img_path = os.path.join(sub_class_path, img_rel_path)
                #
                #         if os.path.exists(full_img_path):
                #             filename = os.path.basename(full_img_path)
                #
                #             # Robust parent identification
                #             if '_x' in filename:
                #                 parent_name = filename.split('_x')[0]
                #             else:
                #                 parent_name = os.path.splitext(filename)[0]
                #
                #             if parent_name not in parent_to_chips:
                #                 parent_to_chips[parent_name] = []
                #
                #             for caption in item["captions"]:
                #                 parent_to_chips[parent_name].append((full_img_path, caption))
                #         else:
                #             # THIS WILL TELL US THE TRUTH
                #             # Only print once to avoid flooding the console
                #             if item == metadata_list[0]:
                #                 print(f"FILE NOT FOUND: {full_img_path}")

                if os.path.isfile(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata_list = json.load(f)
                    for item in metadata_list:
                        caption = item["captions"][0]
                        img_rel_path = item["image_path"]
                        full_img_path = os.path.join(sub_class_path, img_rel_path)

                        if os.path.exists(full_img_path):
                            # Identify the parent.
                            # If file is 'city_x0_y500.jpg', parent is 'city'
                            # We split by the first '_' or use the filename stem
                            filename = os.path.basename(full_img_path)
                            parent_name = filename.split('_x')[0]  # Assumes our naming convention

                            if parent_name not in parent_to_chips:
                                parent_to_chips[parent_name] = []
                            for caption in item["captions"]:
                                parent_to_chips[parent_name].append((full_img_path, caption))

        # 2. Split the PARENTS
        parents = sorted(list(parent_to_chips.keys()))
        random.seed(42)
        random.shuffle(parents)

        n_train = int(len(parents) * self.hparams.train_val_test_split[0])
        n_val = int(len(parents) * self.hparams.train_val_test_split[1])

        train_parents = parents[:n_train]
        val_parents = parents[n_train:n_train + n_val]
        test_parents = parents[n_train + n_val:]

        # 3. Reconstruct pair lists from the split parents
        train_pairs = [pair for p in train_parents for pair in parent_to_chips[p]]
        val_pairs = [pair for p in val_parents for pair in parent_to_chips[p]]
        test_pairs = [pair for p in test_parents for pair in parent_to_chips[p]]

        # 4. Create actual Dataset objects
        self.data_train = GAIADataset(train_pairs, self.tokenizer, self.train_transforms)

        self.data_val = GAIADataset(val_pairs, self.tokenizer, self.val_test_transforms)
        self.data_val.set_train(False)

        self.data_test = GAIADataset(test_pairs, self.tokenizer, self.val_test_transforms)
        self.data_test.set_train(False)

        print(f"Train chips: {len(train_pairs)} (from {len(train_parents)} original images)")
        print(f"Val chips: {len(val_pairs)} (from {len(val_parents)} original images)")

        if hasattr(self, 'train_pairs'): del self.train_pairs
        if hasattr(self, 'val_pairs'): del self.val_pairs

        gc.collect()


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # prefetch_factor=1,
            shuffle=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )