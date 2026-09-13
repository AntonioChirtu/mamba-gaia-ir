"""
Alternative text encoders for the Mamba-3 text-encoder ablation.

Interface contract (matches src/models/components/mamba_ssm_local/modules/mamba3.py):
- __init__(d_model, ...) stores self.d_model
- forward(x, **kwargs) takes already-embedded [B, L, d_model] and returns [B, L, d_model]
  (embedding and pooling happen outside, in Mamba3LitModule.forward)
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerTextEncoder(nn.Module):
    """Param-matched Transformer baseline for the Mamba-3 text-encoder ablation."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        nhead: int = 8,
        dim_feedforward: int = None,
        dropout: float = 0.2,
        max_len: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        dim_feedforward = dim_feedforward or (4 * d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.pos_encoding(x)
        return self.encoder(x)


class BiLSTMTextEncoder(nn.Module):
    """Classic recurrent baseline for the Mamba-3 text-encoder ablation."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        # bidirectional -> 2 * hidden_size == d_model
        hidden_size = d_model // 2
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.dropout(out)