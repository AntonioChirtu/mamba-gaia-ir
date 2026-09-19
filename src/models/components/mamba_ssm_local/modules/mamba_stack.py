from functools import partial

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch import nn

from src.models.components.mamba_ssm_local.models.mixer_seq_simple import _init_weights
from src.models.components.mamba_ssm_local.modules.block import Block
from src.models.components.mamba_ssm_local.modules.mamba3_layer import Mamba3Layer
from src.models.components.mamba_ssm_local.modules.mixer_stack import run_mixer_layers

# layer name -> (mixer class, does the layer accept `dropout` itself?)
# Mamba3Layer takes dropout internally; Mamba2 doesn't, so it needs stack-level dropout.
_LAYER_TYPES = {
    "Mamba2": (Mamba2, False),
    "Mamba3": (Mamba3Layer, True),
}

class MambaStack(nn.Module):
    """Generic multi-layer Mamba encoder: N `Block`s (RMSNorm + mixer + residual),
    mirroring Mamba3's structure, parameterized over the SSM layer type."""

    encodes_raw_text = False

    def __init__(
        self,
        d_model,
        n_layers,
        layer="Mamba3",
        dropout=0.0,
        apply_init_weights=True,
        **layer_kwargs,
    ):
        super().__init__()
        self.d_model = d_model

        if layer not in _LAYER_TYPES:
            raise ValueError(f"Invalid layer: {layer!r}, only support {list(_LAYER_TYPES)}")
        mixer_type, dropout_in_layer = _LAYER_TYPES[layer]

        mixer_cls = partial(
            mixer_type,
            **({"dropout": dropout} if dropout_in_layer else {}),
            **layer_kwargs,
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
        self.dropout = None if dropout_in_layer else nn.Dropout(dropout)
        self.norm_f = RMSNorm(d_model)

        if apply_init_weights:
            self.apply(partial(_init_weights, n_layer=n_layers))

    def forward(self, x, **kwargs):
        hidden = run_mixer_layers(self.layers, x, dropout=self.dropout, **kwargs)
        return self.norm_f(hidden.to(dtype=self.norm_f.weight.dtype))