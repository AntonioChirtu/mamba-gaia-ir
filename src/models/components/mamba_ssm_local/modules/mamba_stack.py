import logging
from functools import partial

from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch import nn

from src.models.components.mamba_ssm_local.models.mixer_seq_simple import _init_weights
from src.models.components.mamba_ssm_local.modules.block import Block
from src.models.components.mamba_ssm_local.modules.mamba2 import Mamba2
from src.models.components.mamba_ssm_local.modules.mamba3 import Mamba3
from src.models.components.mamba_ssm_local.modules.mixer_stack import run_mixer_layers
from src.models.components.mamba_ssm_local.modules.mlp import GatedMLP
from src.models.components.mamba_ssm_local.utils.hf import (
    load_config_hf,
    load_state_dict_hf,
)

log = logging.getLogger(__name__)

# layer name -> (mixer class, does the layer accept `dropout` itself?)
# Mamba3 takes dropout internally; Mamba2 doesn't, so it needs stack-level dropout.
_LAYER_TYPES = {
    "Mamba2": (Mamba2, False),
    "Mamba3": (Mamba3, True),
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
        d_intermediate=0,
        apply_init_weights=True,
        unfreeze_last_n_blocks=None,
        **layer_kwargs,
    ):
        super().__init__()
        self.d_model = d_model

        if layer not in _LAYER_TYPES:
            raise ValueError(
                f"Invalid layer: {layer!r}, only support {list(_LAYER_TYPES)}"
            )
        mixer_type, dropout_in_layer = _LAYER_TYPES[layer]

        mixer_cls = partial(
            mixer_type,
            **({"dropout": dropout} if dropout_in_layer else {}),
            **layer_kwargs,
        )
        mlp_cls = (
            partial(GatedMLP, hidden_features=d_intermediate)
            if d_intermediate > 0
            else nn.Identity
        )
        self.layers = nn.ModuleList(
            [
                Block(
                    d_model,
                    mixer_cls=partial(mixer_cls, layer_idx=i),
                    mlp_cls=mlp_cls,
                    norm_cls=RMSNorm,
                )
                for i in range(n_layers)
            ]
        )
        self.dropout = None if dropout_in_layer else nn.Dropout(dropout)
        self.norm_f = RMSNorm(d_model)

        if apply_init_weights:
            self.apply(partial(_init_weights, n_layer=n_layers))

        if unfreeze_last_n_blocks is not None:
            self._apply_freeze_policy(unfreeze_last_n_blocks)

    def _apply_freeze_policy(self, unfreeze_last_n_blocks: int) -> None:
        for param in self.layers.parameters():
            param.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            for block in list(self.layers)[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        for param in self.norm_f.parameters():
            param.requires_grad = True

    def forward(self, x, **kwargs):
        hidden = run_mixer_layers(self.layers, x, dropout=self.dropout, **kwargs)
        return self.norm_f(hidden.to(dtype=self.norm_f.weight.dtype))

    @classmethod
    def from_pretrained(cls, pretrained_model_name, n_layers=None, **overrides):
        """Build a MambaStack matching a HF `state-spaces/*` checkpoint's backbone
        and load its pretrained weights (embedding/lm_head are dropped - this is
        an encoder-only stack). Layers beyond the checkpoint's depth (`n_layers`
        override > checkpoint n_layer) are left at their fresh init; layers past
        a smaller `n_layers` are simply not loaded."""
        config = load_config_hf(pretrained_model_name)
        ssm_cfg = dict(config.get("ssm_cfg", {}))
        ssm_cfg.pop("layer", None)

        kwargs = {
            "d_model": config["d_model"],
            "n_layers": n_layers if n_layers is not None else config["n_layer"],
            "layer": config.get("ssm_cfg", {}).get("layer", "Mamba3"),
            "d_intermediate": config.get("d_intermediate", 0),
            **ssm_cfg,
            **overrides,
        }
        model = cls(**kwargs)

        state_dict = load_state_dict_hf(pretrained_model_name)
        backbone_prefix = "backbone."
        pretrained = {
            k[len(backbone_prefix) :]: v
            for k, v in state_dict.items()
            if k.startswith(backbone_prefix)
        }

        missing, unexpected = model.load_state_dict(pretrained, strict=False)
        loaded = len(pretrained) - len(unexpected)
        log.info(
            "MambaStack.from_pretrained(%s): loaded %d/%d tensors "
            "(missing=%d, unexpected=%d, fresh-init kept for missing)",
            pretrained_model_name,
            loaded,
            len(model.state_dict()),
            len(missing),
            len(unexpected),
        )
        return model
