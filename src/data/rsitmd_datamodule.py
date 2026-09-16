import json
import os
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.data.eval_collate import eval_collate_fn

# Standard for many base Mamba models
Image.MAX_IMAGE_PIXELS = None


class RSITMDDataset(Dataset):
    """Custom Dataset for RSITMD Information Retrieval.

    Train mode (`split="train"`) returns one (image, caaption) pair at a
    time. Eval mode groups all captions for the same image together."""

    def __init__(
        self,
        root_dir: str,
        tokenizer: Any,
        transform: Any | None = None,
        max_length: int = 77,
        split: str = "train",
        max_captions: int = 5,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_captions = max_captions
        self.is_eval = split in ("val", "test")
        self.data_pairs = []

        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split type: {split}")

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory {root_dir} does not exist.")

        metadata_path = os.path.join(root_dir, "metadata.json")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

        images_list = metadata_dict.get("images", [])

        for item in images_list:
            if item.get("split") != split:
                continue

            filename = item["filename"]
            full_img_path = os.path.join(root_dir, "images", filename)

            if os.path.exists(full_img_path):
                for sentence_obj in item.get("sentences", []):
                    caption = sentence_obj["raw"]
                    self.data_pairs.append((full_img_path, caption))

        if not self.data_pairs:
            raise ValueError(
                f"No samples found for split={split!r}. Check the 'split' values in {metadata_path}."
            )

        if self.is_eval:
            grouped: Dict[str, List[str]] = dict()
            for path, caption in self.data_pairs:
                grouped.setdefault(path, []).append(caption)
            self.eval_records: List[Tuple[str, List[str]]] = list(grouped.items())

        print(
            f"📦 RSITMD Custom Split [{split.upper()}]: "
            f"Allocated {len(self.eval_records) if self.is_eval else len(self.data_pairs)} samples."
        )

    def __len__(self):
        return len(self.eval_records) if self.is_eval else len(self.data_pairs)

    def set_train(self, mode: bool):
        self.is_training = mode

    def __getitem__(self, idx):
        if self.is_eval:
            return self._get_eval_item(idx)
        return self._get_train_item(idx)

    def _get_train_item(self, idx):
        img_path, caption = self.data_pairs[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image.load()

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self._get_train_item((idx + 1) % len(self.data_pairs))

        if self.transform:
            image = self.transform(image)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)

        return image, input_ids, attention_mask, caption

    def _get_eval_item(self, idx):
        img_path, captions = self.eval_records[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image.load()
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self._get_eval_item((idx + 1) % len(self.eval_records))

        if self.transform:
            image = self.transform(image)

        eval_captions = list(captions[: self.max_captions])
        if not eval_captions:
            eval_captions = [""]
        if len(eval_captions) < self.max_captions:
            eval_captions = eval_captions + [eval_captions[-1]] * (
                self.max_captions - len(eval_captions)
            )

        tokens = self.tokenizer(
            eval_captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask

        return image, input_ids, attention_mask, eval_captions, idx


class RSITMDDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: Any,
        data_dir: str = "data/RSITMD",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        max_length: int = 24,
        max_captions: int = 5,
    ) -> None:
        super().__init__()

        if isinstance(tokenizer, (dict, DictConfig)):
            self.tokenizer = instantiate(tokenizer)
        else:
            self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Removed 'train_val_test_split' from hparams since RSITMD specifies splits internally
        self.save_hyperparameters(logger=False, ignore=["tokenizer"])
        self.max_length = max_length
        self.max_captions = max_captions

        # --- TRAINING TRANSFORMS ---
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                # Note: You can keep these CLIP normalization values,
                # but standard ImageNet stats ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # are also common if your vision backbone isn't CLIP.
                transforms.Normalize(
                    mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757]
                ),
                transforms.RandomErasing(p=0.2),
            ]
        )

        # --- VAL/TEST TRANSFORMS ---
        self.val_test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757]
                ),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate datasets using the metadata's own train/val/test split."""

        # --- TRAINING & VALIDATION STAGE ---
        if stage in ("fit", "validate") or stage is None:
            # 1. Grab everything tagged "train"
            self.data_train = RSITMDDataset(
                root_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                transform=self.train_transforms,
                max_length=self.hparams.max_length,
                split="train",
            )

            # 2. Map everything tagged eval_split to be your validation set
            self.data_val = RSITMDDataset(
                root_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                transform=self.val_test_transforms,
                max_length=self.hparams.max_length,
                split="val",
                max_captions=self.max_captions,
            )

        # --- TESTING STAGE ---
        if stage == "test" or stage is None:
            # If you still run trainer.test() later, it will use the same pool
            self.data_test = RSITMDDataset(
                root_dir=self.hparams.data_dir,
                tokenizer=self.tokenizer,
                transform=self.val_test_transforms,
                max_length=self.hparams.max_length,
                split="test",
                max_captions=self.max_captions,
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
            drop_last=False,
            collate_fn=eval_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=eval_collate_fn,
        )
