"""
ViT-only model for image processing (no Mamba).
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

# Try to import timm for more model options
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("timm not available. Install with: pip install timm")


class ViTOnly(nn.Module):
    """
    Pure ViT model for image processing without Mamba.
    """
    
    def __init__(
        self,
        backbone: str = "vit_b16",
        pretrained: bool = True,
        d_model: int = 128,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone
        self.d_model = d_model
        
        # Load pretrained ViT
        vit_feature_dim = self._load_vit_model(backbone, pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Project ViT features to d_model
        self.proj = nn.Linear(vit_feature_dim, d_model)
        
        # Global average pooling (ViT outputs [B, 197, 768] for 224x224 images)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def _load_vit_model(self, backbone: str, pretrained: bool) -> int:
        """Load ViT model and return feature dimension."""
        
        # torchvision models
        if backbone == "vit_b16":
            self.vit = models.vit_b_16(pretrained=pretrained)
            return 768
        elif backbone == "vit_b32":
            self.vit = models.vit_b_32(pretrained=pretrained)
            return 768
        elif backbone == "vit_l16":
            self.vit = models.vit_l_16(pretrained=pretrained)
            return 1024
        elif backbone == "vit_l32":
            self.vit = models.vit_l_32(pretrained=pretrained)
            return 1024
        elif backbone == "vit_h14":
            self.vit = models.vit_h_14(pretrained=pretrained)
            return 1280
        
        # timm models (if available)
        elif TIMM_AVAILABLE:
            try:
                # Get model config to determine feature dim
                model_cfg = timm.get_pretrained_cfg(backbone)
                if model_cfg is None:
                    raise ValueError(f"Model {backbone} not found in timm")
                
                # Create model without classification head
                self.vit = timm.create_model(
                    backbone, 
                    pretrained=pretrained, 
                    num_classes=0,  # Remove classification head
                    global_pool=''  # Remove global pooling
                )
                
                # Get feature dimension from model config or model itself
                if hasattr(model_cfg, 'embed_dim'):
                    return model_cfg.embed_dim
                elif hasattr(self.vit, 'embed_dim'):
                    return self.vit.embed_dim
                elif hasattr(self.vit, 'num_features'):
                    return self.vit.num_features
                else:
                    # Fallback: try to infer from first conv layer
                    for module in self.vit.modules():
                        if hasattr(module, 'out_channels'):
                            return module.out_channels
                    raise ValueError(f"Cannot determine feature dimension for {backbone}")
                    
            except Exception as e:
                raise ValueError(f"Failed to load {backbone} from timm: {e}")
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classification head for torchvision models
        if hasattr(self.vit, 'heads'):
            self.vit.heads = nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] images
        Returns:
            [B, 196, d_model] sequence (remove CLS token)
        """
        # Extract features using ViT
        with torch.set_grad_enabled(not all(p.requires_grad == False for p in self.vit.parameters())):
            features = self.vit(x)  # Could be [B, 197, 768] or [B, 768]
        
        # Handle different output formats
        if features.dim() == 3:
            # Standard ViT output: [B, 197, 768]
            # Remove CLS token (first token)
            features = features[:, 1:, :]  # [B, 196, 768]
        elif features.dim() == 2:
            # Some ViT variants output [B, 768] directly
            # Expand to sequence format by repeating the features
            batch_size = features.shape[0]
            features = features.unsqueeze(1).expand(batch_size, 196, -1)  # [B, 196, 768]
        else:
            raise ValueError(f"Unexpected ViT output shape: {features.shape}")
        
        # Project to d_model
        features = self.proj(features)  # [B, 196, d_model]
        
        return features


class ViTOnlySimple(nn.Module):
    """
    Even simpler ViT that directly outputs pooled features.
    """
    
    def __init__(
        self,
        backbone: str = "vit_b16",
        pretrained: bool = True,
        d_model: int = 128,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone
        self.d_model = d_model
        
        # Load pretrained ViT using the same method
        vit_feature_dim = self._load_vit_model(backbone, pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Project ViT CLS token to d_model
        self.proj = nn.Linear(vit_feature_dim, d_model)
        
    def _load_vit_model(self, backbone: str, pretrained: bool) -> int:
        """Load ViT model and return feature dimension."""
        
        # torchvision models
        if backbone == "vit_b16":
            self.vit = models.vit_b_16(pretrained=pretrained)
            return 768
        elif backbone == "vit_b32":
            self.vit = models.vit_b_32(pretrained=pretrained)
            return 768
        elif backbone == "vit_l16":
            self.vit = models.vit_l_16(pretrained=pretrained)
            return 1024
        elif backbone == "vit_l32":
            self.vit = models.vit_l_32(pretrained=pretrained)
            return 1024
        elif backbone == "vit_h14":
            self.vit = models.vit_h_14(pretrained=pretrained)
            return 1280
        
        # timm models (if available)
        elif TIMM_AVAILABLE:
            try:
                # Get model config to determine feature dim
                model_cfg = timm.get_pretrained_cfg(backbone)
                if model_cfg is None:
                    raise ValueError(f"Model {backbone} not found in timm")
                
                # Create model without classification head
                self.vit = timm.create_model(
                    backbone, 
                    pretrained=pretrained, 
                    num_classes=0,  # Remove classification head
                    global_pool=''  # Remove global pooling
                )
                
                # Get feature dimension from model config or model itself
                if hasattr(model_cfg, 'embed_dim'):
                    return model_cfg.embed_dim
                elif hasattr(self.vit, 'embed_dim'):
                    return self.vit.embed_dim
                elif hasattr(self.vit, 'num_features'):
                    return self.vit.num_features
                else:
                    # Fallback: try to infer from first conv layer
                    for module in self.vit.modules():
                        if hasattr(module, 'out_channels'):
                            return module.out_channels
                    raise ValueError(f"Cannot determine feature dimension for {backbone}")
                    
            except Exception as e:
                raise ValueError(f"Failed to load {backbone} from timm: {e}")
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classification head for torchvision models
        if hasattr(self.vit, 'heads'):
            self.vit.heads = nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] images
        Returns:
            [B, d_model] pooled features (using CLS token)
        """
        # Extract features using ViT
        with torch.set_grad_enabled(not all(p.requires_grad == False for p in self.vit.parameters())):
            features = self.vit(x)  # Could be [B, 197, 768] or [B, 768]
        
        print(f"ViT output shape: {features.shape}, dim: {features.dim()}")
        
        # Handle different output formats
        if features.dim() == 3:
            # Standard ViT output: [B, 197, 768]
            cls_features = features[:, 0, :]  # [B, 768]
        elif features.dim() == 2:
            # Some ViT variants might output [B, 768] directly
            cls_features = features  # [B, 768]
        else:
            raise ValueError(f"Unexpected ViT output shape: {features.shape}")
        
        print(f"CLS features shape: {cls_features.shape}")
        print(f"Projection layer input dim: {self.proj.in_features}, output dim: {self.proj.out_features}")
        
        # Handle dimension mismatch
        if cls_features.shape[-1] != self.proj.in_features:
            print(f"Dimension mismatch! Expected {self.proj.in_features}, got {cls_features.shape[-1]}")
            # Create a new projection layer with correct dimensions
            self.proj = nn.Linear(cls_features.shape[-1], self.d_model).to(cls_features.device)
            print(f"Created new projection layer: {cls_features.shape[-1]} -> {self.d_model}")
        
        # Project to d_model
        features = self.proj(cls_features)  # [B, d_model]
        
        return features
