# src/models/components/locked_openclip_image_encoder.py

from __future__ import annotations

import open_clip
import torch
import torch.nn.functional as F
from torch import nn


class LockedOpenCLIPImageEncoder(nn.Module):
    """
    Frozen OpenCLIP image tower for Locked-image Tuning (LiT).

    The complete pretrained visual path is retained and frozen:
      - patch embedding
      - transformer
      - normalization
      - pretrained visual projection

    The returned features are already in CLIP's shared embedding space.
    No additional trainable image projection should be added after this module.
    """

    encodes_raw_images = True
    is_locked = True

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        normalize: bool = True,
    ) -> None:
        super().__init__()

        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )

        # Keep only the vision tower. In OpenCLIP, the visual module includes
        # its pretrained projection into the shared CLIP embedding space.
        self.visual = clip_model.visual

        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize

        # OpenCLIP visual towers normally expose output_dim.
        output_dim = getattr(self.visual, "output_dim", None)

        if output_dim is None:
            # Standard CLIP models expose the shared output dimension through
            # text_projection.
            text_projection = clip_model.text_projection
            output_dim = int(text_projection.shape[-1])

        # Preserve the interface expected by the existing Mamba3LitModule.
        self.d_model = int(output_dim)
        self.output_dim = int(output_dim)

        self._lock()

    def _lock(self) -> None:
        for parameter in self.visual.parameters():
            parameter.requires_grad_(False)

        self.visual.eval()

    def train(self, mode: bool = True):
        """
        Let the wrapper change mode while forcing the vision tower to remain
        in eval mode.

        Lightning recursively calls model.train(), which would otherwise
        reactivate dropout inside the frozen image tower.
        """
        super().train(mode)
        self.visual.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images:
                CLIP-preprocessed tensor with shape [batch, 3, height, width].

        Returns:
            Frozen image embeddings with shape [batch, output_dim].
        """
        self.visual.eval()

        # No graph is needed through a completely locked image tower.
        with torch.no_grad():
            image_features = self.visual(images)

        if isinstance(image_features, tuple):
            image_features = image_features[0]

        if self.normalize:
            image_features = F.normalize(
                image_features.float(),
                dim=-1,
            )

        return image_features

    def assert_locked(self) -> None:
        trainable = [
            name
            for name, parameter in self.visual.named_parameters()
            if parameter.requires_grad
        ]

        if trainable:
            raise RuntimeError(
                "The LiT image tower is not fully locked. "
                f"Trainable parameters: {trainable[:10]}"
            )

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.visual.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.visual.parameters()
            if parameter.requires_grad
        )