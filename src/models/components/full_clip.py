from __future__ import annotations

from typing import Any

import open_clip
import torch
import torch.nn.functional as F
from torch import nn


def _set_trainable(component: Any, trainable: bool = True) -> None:
    """Handle both nn.Modules and standalone nn.Parameters."""
    if component is None:
        return

    if isinstance(component, nn.Parameter):
        component.requires_grad = trainable
        return

    for parameter in component.parameters():
        parameter.requires_grad = trainable


def _unfreeze_last_blocks(blocks: nn.ModuleList, count: int) -> None:
    if count <= 0:
        return

    for block in list(blocks)[-count:]:
        _set_trainable(block)


class CLIPPretrainedDualEncoder(nn.Module):
    """Matched pretrained CLIP image and text towers.

    This should be used as a complete dual encoder, rather than being placed
    inside either `image_net` or `text_net`.
    """

    encodes_raw_text = True
    encodes_images = True

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        unfreeze_last_n_visual_blocks: int = 0,
        unfreeze_last_n_text_blocks: int = 0,
        fully_unfreeze_visual: bool = False,
        fully_unfreeze_text: bool = False,
        train_projection_heads: bool = True,
        train_logit_scale: bool = True,
    ) -> None:
        super().__init__()

        (
            self.clip_model,
            self.preprocess_train,
            self.preprocess_val,
        ) = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.d_model = self.clip_model.text_projection.shape[1]

        self._apply_freeze_policy(
            unfreeze_last_n_visual_blocks=unfreeze_last_n_visual_blocks,
            unfreeze_last_n_text_blocks=unfreeze_last_n_text_blocks,
            fully_unfreeze_visual=fully_unfreeze_visual,
            fully_unfreeze_text=fully_unfreeze_text,
            train_projection_heads=train_projection_heads,
            train_logit_scale=train_logit_scale,
        )

    def _apply_freeze_policy(
        self,
        *,
        unfreeze_last_n_visual_blocks: int,
        unfreeze_last_n_text_blocks: int,
        fully_unfreeze_visual: bool,
        fully_unfreeze_text: bool,
        train_projection_heads: bool,
        train_logit_scale: bool,
    ) -> None:
        # Start with the entire matched CLIP model frozen.
        _set_trainable(self.clip_model, False)

        # ----- Visual tower -----
        visual = self.clip_model.visual

        if fully_unfreeze_visual:
            _set_trainable(visual)
        else:
            visual_transformer = getattr(visual, "transformer", None)
            visual_blocks = getattr(visual_transformer, "resblocks", None)

            if visual_blocks is not None:
                _unfreeze_last_blocks(
                    visual_blocks,
                    unfreeze_last_n_visual_blocks,
                )

        # ----- Text tower -----
        if fully_unfreeze_text:
            _set_trainable(self.clip_model.token_embedding)

            self.clip_model.positional_embedding.requires_grad = True

            _set_trainable(self.clip_model.transformer)
            _set_trainable(self.clip_model.ln_final)
            _set_trainable(self.clip_model.text_projection)
        else:
            _unfreeze_last_blocks(
                self.clip_model.transformer.resblocks,
                unfreeze_last_n_text_blocks,
            )

        # Projection-head training is independent of backbone training.
        if train_projection_heads:
            # Text-side final normalization and projection.
            _set_trainable(self.clip_model.ln_final)
            _set_trainable(self.clip_model.text_projection)

            # OpenCLIP ViT models generally expose ln_post and proj.
            _set_trainable(getattr(visual, "ln_post", None))
            _set_trainable(getattr(visual, "proj", None))

        if train_logit_scale:
            _set_trainable(self.clip_model.logit_scale)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(images)
        return F.normalize(features, dim=-1)

    def encode_texts(self, raw_texts: list[str]) -> torch.Tensor:
        device = self.clip_model.text_projection.device
        tokens = self.tokenizer(raw_texts).to(device)

        features = self.clip_model.encode_text(tokens)
        return F.normalize(features, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        raw_texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_images(images)
        text_features = self.encode_texts(raw_texts)
        return image_features, text_features

    @property
    def logit_scale(self) -> torch.Tensor:
        # OpenCLIP stores the logarithm of the scale.
        return self.clip_model.logit_scale.exp()