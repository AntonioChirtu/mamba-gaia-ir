"""Multimodal model combining ViT for images and Mamba for text processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Optimizer
from typing import Dict, Any, Tuple


class VitMambaMultimodalLitModule(L.LightningModule):
    """
    Multimodal Lightning module combining Vision Transformer for images
    and Mamba for text processing with contrastive learning.
    """

    def __init__(
            self,
            vision_encoder: nn.Module,
            text_encoder: nn.Module,
            projections: Dict[str, nn.Module],
            contrastive: Dict[str, Any],
            optimizer: Dict[str, Any],
            scheduler: Dict[str, Any],
            compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Encoders
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # Fusion and projections
        self.fusion = fusion
        self.vision_proj = projections["vision_proj"]
        self.text_proj = projections["text_proj"]

        # Contrastive learning parameters
        self.temperature = contrastive["temperature"]
        self.loss_type = contrastive.get("loss_type", "InfoNCE")

        # Optimizer config
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # Compile model if requested
        if compile:
            self.vision_encoder = torch.compile(self.vision_encoder)
            self.text_encoder = torch.compile(self.text_encoder)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the multimodal model."""
        # Extract image and text data
        images = batch.get("images")
        texts = batch.get("texts")

        outputs = {}

        # Process images through ViT
        if images is not None:
            vision_features = self.vision_encoder(images)
            vision_embeddings = self.vision_proj(vision_features)
            outputs["vision_embeddings"] = vision_embeddings
            outputs["vision_features"] = vision_features

        # Process text through Mamba
        if texts is not None:
            text_features = self.text_encoder(texts)
            text_embeddings = self.text_proj(text_features)
            outputs["text_embeddings"] = text_embeddings
            outputs["text_features"] = text_features

        # Multimodal fusion
        if images is not None and texts is not None:
            fused_features = self.fusion(vision_features, text_features)
            outputs["fused_features"] = fused_features

        return outputs

    def compute_contrastive_loss(
            self,
            vision_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss."""
        batch_size = vision_embeddings.size(0)

        # Normalize embeddings
        vision_embeddings = F.normalize(vision_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(vision_embeddings, text_embeddings.T) / self.temperature

        # Create labels (positive pairs are on the diagonal)
        labels = torch.arange(batch_size, device=self.device)

        # Compute symmetric loss
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)

        return (loss_v2t + loss_t2v) / 2

    def training_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step with contrastive loss."""
        outputs = self.forward(batch)

        # Compute contrastive loss if both modalities are present
        if "vision_embeddings" in outputs and "text_embeddings" in outputs:
            loss = self.compute_contrastive_loss(
                outputs["vision_embeddings"],
                outputs["text_embeddings"]
            )
        else:
            # Fallback to reconstruction or other task-specific loss
            loss = torch.tensor(0.0, device=self.device)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        outputs = self.forward(batch)

        if "vision_embeddings" in outputs and "text_embeddings" in outputs:
            loss = self.compute_contrastive_loss(
                outputs["vision_embeddings"],
                outputs["text_embeddings"]
            )
        else:
            loss = torch.tensor(0.0, device=self.device)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step."""
        outputs = self.forward(batch)

        if "vision_embeddings" in outputs and "text_embeddings" in outputs:
            loss = self.compute_contrastive_loss(
                outputs["vision_embeddings"],
                outputs["text_embeddings"]
            )
        else:
            loss = torch.tensor(0.0, device=self.device)

        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config["lr"],
            weight_decay=self.optimizer_config["weight_decay"],
        )

        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.scheduler_config["T_0"],
            T_mult=self.scheduler_config["T_mult"],
            eta_min=self.scheduler_config["eta_min"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
