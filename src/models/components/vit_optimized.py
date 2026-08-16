"""
Optimized ViT model for faster convergence.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


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
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classification head
        if hasattr(self.vit, 'heads'):
            self.vit.heads = nn.Identity()
        elif hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            # Unfreeze last few layers for faster adaptation
            for param in self.vit.encoder.layers[-4:].parameters():
                param.requires_grad = True
            for param in self.vit.encoder.layers[:-4].parameters():
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
            x: [B, 3, 224, 224] images
        Returns:
            [B, d_model] optimized features
        """
        # Extract features using ViT
        with torch.set_grad_enabled(not all(p.requires_grad == False for p in self.vit.parameters())):
            features = self.vit(x)  # [B, 197, 768] or [B, 768]
        
        # Handle different output formats
        if features.dim() == 3:
            # Standard ViT output: [B, 197, 768]
            cls_features = features[:, 0, :]  # [B, 768]
        elif features.dim() == 2:
            # Some ViT variants might output [B, 768] directly
            cls_features = features  # [B, 768]
        else:
            raise ValueError(f"Unexpected ViT output shape: {features.shape}")
        
        # Enhanced projection
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
