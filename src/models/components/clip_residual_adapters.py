from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.full_clip import CLIPPretrainedDualEncoder


class ResidualAdapter(nn.Module):
    """Small, identity-initialized update to a normalized CLIP embedding."""

    def __init__(
        self,
        dim: int,
        bottleneck: int = 64,
        residual_scale: float = 0.2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.residual_scale = residual_scale

        # Zero residual at initialization; keep down randomly initialized.
        nn.init.zeros_(self.up.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(
            self.dropout(self.activation(self.down(features)))
        )
        return F.normalize(
            features + self.residual_scale * residual,
            dim=-1,
        )


class CLIPResidualAdapterDualEncoder(CLIPPretrainedDualEncoder):
    """Frozen matched CLIP towers with trainable embedding-space adapters."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        adapter_bottleneck: int = 64,
        adapter_scale: float = 0.2,
        adapter_dropout: float = 0.1,
        # Accept the existing CLIP configuration fields, but enforce freezing.
        unfreeze_last_n_visual_blocks: int = 0,
        unfreeze_last_n_text_blocks: int = 0,
        fully_unfreeze_visual: bool = False,
        fully_unfreeze_text: bool = False,
        train_projection_heads: bool = False,
        train_logit_scale: bool = False,
    ) -> None:
        if any(
            [
                unfreeze_last_n_visual_blocks != 0,
                unfreeze_last_n_text_blocks != 0,
                fully_unfreeze_visual,
                fully_unfreeze_text,
                train_projection_heads,
                train_logit_scale,
            ]
        ):
            raise ValueError(
                "Trial 3 requires frozen CLIP towers, projections "
                "and logit scale."
            )

        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            unfreeze_last_n_visual_blocks=0,
            unfreeze_last_n_text_blocks=0,
            fully_unfreeze_visual=False,
            fully_unfreeze_text=False,
            train_projection_heads=False,
            train_logit_scale=False,
        )

        self.image_adapter = ResidualAdapter(
            dim=self.d_model,
            bottleneck=adapter_bottleneck,
            residual_scale=adapter_scale,
            dropout=adapter_dropout,
        )
        self.text_adapter = ResidualAdapter(
            dim=self.d_model,
            bottleneck=adapter_bottleneck,
            residual_scale=adapter_scale,
            dropout=adapter_dropout,
        )

        self.clip_model.requires_grad_(False)
        self.clip_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        # Lightning calls model.train(); keep the frozen backbone in eval mode.
        # The adapters still follow the requested training mode.
        self.clip_model.eval()
        return self

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            features = F.normalize(features.float(), dim=-1)

        # Must remain OUTSIDE no_grad so adapter weights receive gradients.
        return self.image_adapter(features)

    def encode_texts(self, raw_texts: list[str]) -> torch.Tensor:
        device = self.clip_model.text_projection.device
        tokens = self.tokenizer(raw_texts).to(device)

        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
            features = F.normalize(features.float(), dim=-1)

        return self.text_adapter(features)