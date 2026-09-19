import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2 import Mamba2 as Mamba2Block


class Mamba2ResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
        layer_idx: int | None = None,
    ):
        super().__init__()

        # Outer block normalization. This is separate from Mamba2's
        # internal gated RMSNorm.
        self.norm = nn.RMSNorm(d_model)

        self.mixer = Mamba2Block(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            layer_idx=layer_idx,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mixer(self.norm(x)))


class Mamba2TextEncoder(nn.Module):
    """Stacked causal Mamba2 text encoder with a shared encoder interface."""

    encodes_raw_text = False

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [
                Mamba2ResidualBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    layer_idx=i,
                )
                for i in range(n_layers)
            ]
        )

        self.norm_final = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask.bool()

            # This wrapper relies on causal processing plus right-padding.
            # Reject left padding or holes in the mask.
            becomes_valid_again = (~attention_mask[:, :-1]) & attention_mask[:, 1:]
            if becomes_valid_again.any():
                raise ValueError(
                    "Mamba2TextEncoder requires right-padded attention masks "
                    "of the form [1, 1, ..., 0, 0]."
                )

            # Zero input embeddings at padding positions.
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_final(x)

        # Padded positions are not used by last-valid-token pooling, but
        # zeroing them makes the returned sequence safer for other consumers.
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        return x