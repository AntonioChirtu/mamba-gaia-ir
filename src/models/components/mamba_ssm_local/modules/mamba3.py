from functools import partial

from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch import nn

from src.models.components.mamba_ssm_local.modules.block import Block
from src.models.components.mamba_ssm_local.modules.mamba3_layer import Mamba3Layer


class Mamba3(nn.Module):
    def __init__(
        self, d_model, d_state, d_conv, expand, headdim, dropout, n_layers, **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        mixer_cls = partial(
            Mamba3Layer,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            dropout=dropout,
            **kwargs,
        )
        self.layers = nn.ModuleList(
            [
                Block(
                    d_model,
                    mixer_cls=partial(mixer_cls, layer_idx=i),
                    mlp_cls=nn.Identity,
                    norm_cls=RMSNorm,
                )
                for i in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)

    def forward(self, x, **kwargs):
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual, **kwargs)
        residual = (x + residual) if residual is not None else x
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
