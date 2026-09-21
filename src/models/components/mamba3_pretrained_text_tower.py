"""
Mamba3-pretrained text tower - a self-contained text_net arm, mirroring
CLIPPretrainedTextEncoder's shape: owns its own tokenizer and token embedding
instead of consuming Mamba3LitModule's shared, from-scratch nn.Embedding.

The state-spaces/mamba3-* checkpoints are pretrained with the
meta-llama/Llama-3.1-8B tokenizer and tied input/output embeddings - feeding
the pretrained backbone token ids from GAIA's GPT-NeoX tokenizer would put it
in a vocabulary it never saw, discarding most of the pretraining signal. This
arm bypasses that shared embedding entirely via the `encodes_raw_text` marker
Mamba3LitModule checks for (same mechanism CLIPPretrainedTextEncoder uses).

Note: meta-llama/Llama-3.1-8B is a gated HF repo - the environment running
this needs an HF token with that license accepted.
"""

import torch
from torch import nn
from transformers import AutoTokenizer

from src.models.components.mamba_ssm_local.modules.mamba_stack import MambaStack
from src.models.components.mamba_ssm_local.utils.hf import (
    load_config_hf,
    load_state_dict_hf,
)


class Mamba3PretrainedTextEncoder(nn.Module):
    encodes_raw_text = True

    def __init__(
        self,
        pretrained_model_name: str = "state-spaces/mamba3-siso-187m",
        tokenizer_name: str = "meta-llama/Llama-3.1-8B",
        n_layers: int | None = None,
        unfreeze_last_n_blocks: int | None = None,
        unfreeze_embedding: bool = False,
        max_length: int = 128,
        **backbone_overrides,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        self.backbone = MambaStack.from_pretrained(
            pretrained_model_name,
            n_layers=n_layers,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            **backbone_overrides,
        )
        self.d_model = self.backbone.d_model

        config = load_config_hf(pretrained_model_name)
        state_dict = load_state_dict_hf(pretrained_model_name)
        embedding_key = "backbone.embedding.weight"
        if embedding_key not in state_dict:
            raise KeyError(
                f"{embedding_key!r} not found in {pretrained_model_name}"
                " checkpoint - can't build a matching pretrained embedding.",
            )

        self.embedding = nn.Embedding(config["vocab_size"], self.d_model)
        self.embedding.weight.data.copy_(state_dict[embedding_key])
        self.embedding.weight.requires_grad = unfreeze_embedding

    def forward(self, raw_texts: list[str]) -> torch.Tensor:
        device = self.embedding.weight.device
        tokens = self.tokenizer(
            raw_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        x = self.embedding(tokens.input_ids)
        out = self.backbone(x)  # [B, L, d_model]

        # Pool the last non-pad token (tokenizer right-pads by default).
        lengths = tokens.attention_mask.long().sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, out.size(-1))
        return out.gather(1, idx).squeeze(1)