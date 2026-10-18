import torch
import torch.nn.functional as F

from src.models.components.clip_residual_adapters import (
    CLIPResidualAdapterDualEncoder,
)


class CLIPLastVisualBlockAdapterDualEncoder(CLIPResidualAdapterDualEncoder):
    """Trial 3 plus training of the final visual transformer block."""

    def __init__(self, visual_block_lr: float = 1e-5, **kwargs):
        super().__init__(**kwargs)

        if visual_block_lr <= 0:
            raise ValueError("visual_block_lr must be positive.")

        self.visual_block_lr = float(visual_block_lr)
        self.enable_last_visual_block_training()

    @property
    def last_visual_block(self):
        transformer = getattr(
            self.clip_model.visual, "transformer", None
        )
        blocks = getattr(transformer, "resblocks", None)

        if blocks is None or len(blocks) == 0:
            raise ValueError(
                "Expected OpenCLIP ViT visual transformer blocks."
            )

        return blocks[-1]

    def enable_last_visual_block_training(self):
        # Explicitly preserve the intended freezing policy.
        self.clip_model.requires_grad_(False)
        self.image_adapter.requires_grad_(True)
        self.text_adapter.requires_grad_(True)
        self.last_visual_block.requires_grad_(True)

        # Eval mode does not disable gradients. Keep backbone stochastic
        # layers disabled, matching Trial 3's deterministic backbone.
        self.clip_model.eval()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: no torch.no_grad() around image encoding.
        # Gradients must reach the final visual block.
        features = self.clip_model.encode_image(images)
        features = F.normalize(features.float(), dim=-1)

        return self.image_adapter(features)

    def optimizer_param_groups(self):
        return [
            {
                "name": "adapters",
                "params": (
                    list(self.image_adapter.parameters())
                    + list(self.text_adapter.parameters())
                ),
                # Inherits model.optimizer.lr.
            },
            {
                "name": "last_visual_block",
                "params": list(self.last_visual_block.parameters()),
                "lr": self.visual_block_lr,
            },
        ]