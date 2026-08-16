"""
Pretrained vision components for Mamba2 model.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class PretrainedVisionEncoder(nn.Module):
    """
    Vision encoder using pretrained backbone (ResNet/ViT) with Mamba2 processing.
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        feature_dim: int = 2048,  # ResNet50 feature dim
        d_model: int = 128,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone
        self.d_model = d_model
        
        # Load pretrained backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove final FC
            feature_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 2048
        elif backbone == "vit_b16":
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            self.backbone.heads = nn.Identity()  # Remove classification head
            feature_dim = 768
        elif backbone == "vit_b32":
            self.backbone = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
            self.backbone.heads = nn.Identity()  # Remove classification head
            feature_dim = 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Project backbone features to d_model
        self.proj = nn.Linear(feature_dim, d_model)
        
        # Learnable positional embeddings for sequence
        self.pos_embed = nn.Parameter(torch.randn(1, 196, d_model))  # 14x14 grid
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] images
        Returns:
            [B, 196, d_model] sequence
        """
        # Extract features using pretrained backbone
        with torch.set_grad_enabled(not all(p.requires_grad == False for p in self.backbone.parameters())):
            features = self.backbone(x)  # [B, feature_dim, 1, 1] or [B, feature_dim]
        
        if len(features.shape) == 4:  # ResNet output
            features = features.flatten(1)  # [B, feature_dim]
        
        # Project to d_model
        features = self.proj(features)  # [B, d_model]
        
        # Expand to sequence and add positional embeddings
        batch_size = features.shape[0]
        sequence = features.unsqueeze(1).expand(batch_size, 196, self.d_model)  # [B, 196, d_model]
        sequence = sequence + self.pos_embed
        
        return sequence


class Mamba2WithPretrainedVision(nn.Module):
    """
    Mamba2 model that uses pretrained vision features instead of patch embedding.
    """
    
    def __init__(
        self,
        vision_encoder: PretrainedVisionEncoder,
        mamba2_model: nn.Module,
        d_model: int = 128
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.mamba2_model = mamba2_model
        self.d_model = d_model
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] images
        Returns:
            [B, 196, d_model] processed features
        """
        # Extract vision features
        vision_features = self.vision_encoder(x)  # [B, 196, d_model]
        
        # Process with Mamba2
        output = self.mamba2_model(vision_features)
        
        return output
