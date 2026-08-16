"""Projection heads for contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(
            self,
            input_dim: int = 768,
            hidden_dim: int = 512,
            output_dim: int = 256,
            dropout: float = 0.1,
            use_bn: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_bn = use_bn

        if use_bn:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (B, input_dim)

        Returns:
            Projected features of shape (B, output_dim)
        """
        return self.projection(x)


class DualProjectionHead(nn.Module):
    """Dual projection head with separate vision and text projections."""

    def __init__(
            self,
            vision_dim: int = 768,
            text_dim: int = 768,
            hidden_dim: int = 512,
            output_dim: int = 256,
            dropout: float = 0.1,
            use_bn: bool = True,
    ):
        super().__init__()

        self.vision_proj = ProjectionHead(
            vision_dim, hidden_dim, output_dim, dropout, use_bn
        )
        self.text_proj = ProjectionHead(
            text_dim, hidden_dim, output_dim, dropout, use_bn
        )

    def forward(
            self,
            vision_features: torch.Tensor,
            text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both modalities."""
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        return vision_proj, text_proj


class AdaptiveProjectionHead(nn.Module):
    """Adaptive projection head with learnable temperature."""

    def __init__(
            self,
            input_dim: int = 768,
            hidden_dim: int = 512,
            output_dim: int = 256,
            dropout: float = 0.1,
            temperature_init: float = 0.07,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Learnable temperature parameter
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))

    @property
    def temperature(self) -> torch.Tensor:
        """Get temperature parameter."""
        return torch.exp(self.log_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling."""
        x = self.projection(x)
        # Apply temperature scaling (will be normalized in contrastive loss)
        return x / self.temperature.clamp(min=0.001, max=1.0)
