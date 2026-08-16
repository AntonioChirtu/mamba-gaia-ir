"""
Optimized ViT model for faster convergence.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

import timm


class ViTOptimized(nn.Module):
    """
    ViT model optimized for faster convergence with contrastive learning.
    """

    def __init__(
            self,
            backbone: str = "vit_b16",
            pretrained: bool = True,
            d_model: int = 256,
            freeze_backbone: bool = False,
            use_layer_norm: bool = True,
            dropout: float = 0.1
    ):
        super().__init__()

        self.backbone_name = backbone
        self.d_model = d_model

        # Load pretrained ViT
        if backbone == "vit_b16":
            self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            feature_dim = 768
        elif backbone == "vit_b32":
            self.vit = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
            feature_dim = 768
        elif backbone == "vit_l16":
            self.vit = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
            feature_dim = 1024
        elif backbone == "vit_l14":
            self.vit = timm.create_model('vit_large_patch14_clip_224', pretrained=True)
            feature_dim = 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove classification head
        if hasattr(self.vit, 'heads'):
            self.vit.heads = nn.Identity()
        elif hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()

        # 1. Handle Attribute Mapping (timm vs torchvision)
        if hasattr(self.vit, 'patch_embed'):  # timm
            print("Using timm ViT")
            self.patch_embed = self.vit.patch_embed
            self.blocks = self.vit.blocks
            self.pos_embed = self.vit.pos_embed
            self.cls_token = self.vit.cls_token
            self.dropout_layer = self.vit.pos_drop
            self.final_norm = self.vit.norm
        else:  # torchvision
            print("Using torchvision ViT")
            self.patch_embed = self.vit.conv_proj
            self.blocks = self.vit.encoder.layers
            self.pos_embed = self.vit.encoder.pos_embedding
            self.cls_token = self.vit.class_token
            self.dropout_layer = self.vit.encoder.dropout
            self.final_norm = self.vit.encoder.ln

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            # Unfreeze last few layers for faster adaptation
            for param in self.blocks[-4:].parameters():
                param.requires_grad = True
            for param in self.blocks[:-4].parameters():
                param.requires_grad = False

        # Enhanced projection head for better contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        )

        # Initialize projection head properly
        self._init_projection_head()

    def _init_projection_head(self):
        """Initialize projection head for better convergence."""
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] images (can be any resolution multiple of patch_size)
        Returns:
            [B, d_model] optimized features
        """
        B, C, H, W = x.shape

        # # 1. Extract patch embeddings using the native conv_proj
        # x = self.vit.conv_proj(x)  # Shape: [B, feature_dim, grid_H, grid_W]
        # grid_h, grid_w = x.shape[2], x.shape[3]
        #
        # # Flatten and transpose to token sequence
        # x = x.flatten(2).transpose(1, 2)  # [B, N, feature_dim]

        # 1. Patch Embedding
        # Note: timm's patch_embed handles flatten/transpose automatically
        if self.backbone_name == "vit_l14":  # timm
            x = self.patch_embed(x)
        else:  # torchvision
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)

        p_size = getattr(self.patch_embed, 'patch_size', getattr(self.patch_embed, 'kernel_size', None))
        grid_h, grid_w = H // p_size[0], W // p_size[1]

        # 2. Prepend Class Token
        class_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([class_token, x], dim=1)  # [B, N+1, feature_dim]

        # 3. Dynamic Positional Embedding Interpolation
        pos_embed = self.pos_embed  # Base parameter, usually [1, 197, feature_dim]

        if x.shape[1] != pos_embed.shape[1]:
            # Separate class token pos and grid pos
            cls_pos = pos_embed[:, :1, :]
            grid_pos = pos_embed[:, 1:, :]

            # Calculate original grid dimensions (assumes square base resolution, e.g., 14x14)
            orig_grid_size = int(grid_pos.shape[1] ** 0.5)

            # Reshape grid pos to spatial format: [1, feature_dim, orig_H, orig_W]
            grid_pos = grid_pos.reshape(1, orig_grid_size, orig_grid_size, -1).permute(0, 3, 1, 2)

            # Interpolate to current grid size
            grid_pos = torch.nn.functional.interpolate(
                grid_pos,
                size=(grid_h, grid_w),
                mode='bicubic',
                align_corners=False
            )

            # Flatten back to sequence format: [1, N, feature_dim]
            grid_pos = grid_pos.permute(0, 2, 3, 1).flatten(1, 2)

            # Recombine
            interpolated_pos_embed = torch.cat([cls_pos, grid_pos], dim=1)
            x = x + interpolated_pos_embed
        else:
            # Native resolution, just add
            x = x + pos_embed

        # 4. Pass through ViT Encoder blocks
        x = self.dropout_layer(x)
        x = self.blocks(x)
        x = self.final_norm(x) # <--- CRITICAL FIX: Normalize before projecting!

        # 5. Extract class token and project
        cls_features = x[:, 0, :]  # [B, feature_dim]
        projected_features = self.projection_head(cls_features)  # [B, d_model]

        return projected_features


class ViTOnlySimpleOptimized(ViTOptimized):
    """
    Simplified optimized ViT that matches the original interface.
    """

    def __init__(self, **kwargs):
        # Explicitly run the ViTOptimized.__init__
        super().__init__(**kwargs)

    def forward(self, x):
        # Ensure the child uses the parent's forward logic
        return super().forward(x)
