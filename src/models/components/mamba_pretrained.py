"""
Utilities for loading pretrained Mamba weights.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from huggingface_hub import hf_hub_download


def load_pretrained_mamba(
    model: nn.Module,
    model_name: str = "state-spaces/mamba-2.8b",
    strict: bool = False,
    device: Optional[str] = None
) -> nn.Module:
    """
    Load pretrained Mamba weights from HuggingFace Hub.
    
    Args:
        model: Your Mamba2 model instance
        model_name: Name of pretrained model on HF Hub
        strict: Whether to strictly enforce that the keys match
        device: Device to load weights on
        
    Returns:
        Model with loaded weights
    """
    try:
        # Download pretrained weights
        checkpoint_path = hf_hub_download(
            repo_id=model_name,
            filename="pytorch_model.bin"
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load state dict
        model.load_state_dict(checkpoint, strict=strict)
        
        print(f"Successfully loaded pretrained weights from {model_name}")
        return model
        
    except Exception as e:
        print(f"Failed to load pretrained weights: {e}")
        print("Continuing with randomly initialized weights...")
        return model


def load_vision_pretrained_mamba(
    model: nn.Module,
    checkpoint_path: Optional[str] = None,
    strict: bool = False
) -> nn.Module:
    """
    Load Mamba weights that were pretrained on vision tasks.
    
    Args:
        model: Your Mamba2 model instance
        checkpoint_path: Path to local checkpoint file
        strict: Whether to strictly enforce that the keys match
        
    Returns:
        Model with loaded weights
    """
    if checkpoint_path is None:
        print("No checkpoint path provided. Using random initialization.")
        return model
    
    try:
        # Load local checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove any prefix from keys (e.g., 'model.', 'backbone.')
        clean_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key
            if key.startswith('model.'):
                clean_key = key[6:]  # Remove 'model.' prefix
            elif key.startswith('backbone.'):
                clean_key = key[9:]  # Remove 'backbone.' prefix
            clean_state_dict[clean_key] = value
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=strict)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print(f"Successfully loaded weights from {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Continuing with randomly initialized weights...")
        return model


def create_vision_pretrained_mamba(
    d_model: int = 128,
    d_state: int = 32,
    d_conv: int = 4,
    expand: int = 2,
    headdim: int = 8,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """
    Create a Mamba2 model and optionally load vision pretrained weights.
    
    Args:
        d_model: Model dimension
        d_state: State dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        headdim: Head dimension
        checkpoint_path: Path to pretrained checkpoint
        
    Returns:
        Mamba2 model with optional pretrained weights
    """
    from mamba_ssm_local.modules.mamba2 import Mamba2
    
    # Create model
    model = Mamba2(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim
    )
    
    # Load pretrained weights if provided
    if checkpoint_path:
        model = load_vision_pretrained_mamba(model, checkpoint_path)
    
    return model
