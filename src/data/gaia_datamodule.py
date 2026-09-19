import glob
import json
import os
import random
from typing import Any, List, Optional  # noqa: UP035

from lightning import LightningDataModule
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.data.eval_collate import eval_collate_fn

# Standard for many base Mamba models
Image.MAX_IMAGE_PIXELS = None


class GAIADataset(Dataset):
    """Custom Dataset for GAIA Information Retrieval.
    
    Each record is a dictionary containing the image path, captions,
    stable image/group IDs, stable text IDs, and source metadata,
    where ``captions`` is the list of
    synthetic captions GAIA provides per image. During training, a caption is
    sampled at random each epoch (free text-side augmentation). During
    validation/testing all ``max_captions`` captions are returned so recall
    can be computed against thefull multi-relevant target set (standard
    protocol for 5-captions-per-image benchmarks: Flickr30k, COCO, RSICD/RSITMD)
    instead of an arbitrary single caption.
    """

    def __init__(
        self,
        records: List[dict[str, Any]],
        tokenizer: Any, 
        transform: Optional[Any] = None, 
        max_length: int = 128,
        is_training: bool = True,
        max_captions: int = 5,
    ):
        self.records = records
        self.transform = transform
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.is_training = is_training
        self.max_captions = max_captions

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

    def get_retrieval_metadata(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]

        num_valid_captions = min(
            len(record["captions"]),
            self.max_captions,
        )

        return {
            "dataset_index": idx,
            "sample_id": record["sample_id"],
            "group_id": record["group_id"],
            "text_ids": record["text_ids"][:num_valid_captions],
            "num_valid_captions": num_valid_captions,
            "location": record.get("location"),
            "image_alt": record.get("image_alt"),
        }

    def __getitem__(self, idx):
        record = self.records[idx]

        img_path = record["image_path"]
        captions = record["captions"]

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        if self.is_training:
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
        
        # Eval: return all captions so recall can be computed against the
        # full multi-relevant target set (not just an arbitrary single one)
        eval_captions = list(captions[: self.max_captions])
        if not eval_captions:
            eval_captions = [""]
        if len(eval_captions) < self.max_captions:
            eval_captions = eval_captions + [eval_captions[-1]] * (
                self.max_captions - len(eval_captions)
            )
        
        tokens = self.tokenizer(
                eval_captions,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
        )

        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        return image, idx, input_ids, attention_mask, eval_captions



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
            max_captions: int = 5,
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
    def _scan_split(self, split: str) -> List[dict[str, Any]]:
        """Scan a split and return one metadata dictionary per image."""
        splits_dir = os.path.join(self.hparams.data_dir, split)
        if not os.path.isdir(splits_dir):
            raise FileNotFoundError(f"Split directory {splits_dir} does not exist.")

        records: List[dict[str, Any]] = []
        seen_sample_ids: set[str] = set()

        n_failed, n_off_sphere = 0, 0
        n_missing_id = 0

        pattern = os.path.join(splits_dir, "*", "*.png")

        for img_path in sorted(glob.glob(pattern)):
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

            # Ensure the expected representation.
            if not isinstance(captions, list):
                raise TypeError(
                    f"'captions' must be a list in {json_path}, "
                    f"got {type(captions).__name__}"
                )

            # Remove empty captions while preserving original caption indexes.
            indexed_captions = [
                (caption_index, caption)
                for caption_index, caption in enumerate(captions)
                if isinstance(caption, str) and caption.strip()
            ]

            if not indexed_captions:
                continue
            
            source_image_id = meta.get("id")

            if source_image_id is None:
                n_missing_id += 1
                continue

            source_image_id = str(source_image_id)
            sample_id = f"gaia:image:{source_image_id}"

            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"Duplicate GAIA image ID {source_image_id!r} "
                    f"found while scanning {split}. File: {json_path}"
                )

            seen_sample_ids.add(sample_id)

            # Captions and text_ids remain positionally aligned.
            clean_captions = [
                caption
                for _, caption in indexed_captions
            ]

            text_ids = [
                f"gaia:caption:{source_image_id}:{source_caption_index}"
                for source_caption_index, _ in indexed_captions
            ]

            records.append(
                {
                    "image_path": img_path,
                    "captions": clean_captions,
                    "text_ids": text_ids,
                    "sample_id": sample_id,
                    "group_id": sample_id,
                    "image_alt": meta.get("image_alt"),
                    "location": meta.get("location"),
                }
            )

        if n_failed:
            print(f"GAIA {split}: skipped {n_failed} non-success downloads.")
        if n_off_sphere:
            print(f"GAIA {split}: skipped {n_off_sphere} images outside spheres={self.hparams.spheres}.")
        if n_missing_id:
            print(
                f"GAIA {split}: skipped {n_missing_id} images "
                "without a stable id."
            )
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
            max_captions=self.hparams.max_captions,
        )
        self.data_test = GAIADataset(
            test_records, self.tokenizer, self.val_test_transforms,
            self.hparams.max_length, is_training=False,
            max_captions=self.hparams.max_captions,
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
            collate_fn=eval_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=2,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=eval_collate_fn,
        )