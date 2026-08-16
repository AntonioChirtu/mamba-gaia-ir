import torch.nn as nn
from src.models.components.mamba_ssm_local.modules.mamba3_layer import Mamba3Layer

class Mamba3(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, headdim, dropout, n_layers, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            Mamba3Layer(d_model=self.d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, dropout=dropout, **kwargs)
            for _ in range(n_layers)
        ])

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)  # Loop here, not inside the layer!
        return x