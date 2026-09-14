"""CLIP-pretrained transformer text tower - a self-contained text_net arm.

Unlike the other text_net arms (mamba2/mamba3/transformer/bilstm), this
module owns its own tokenizer and token embedding instead of consuming
Mamba3LitModule's shared, from-scratch nn.Embedding. CLIP's pretrained
transformer blocks were trained jointly with CLIP's own embedding space -
feeding them GPT-NeoX embeddings would discard most of the pretraining
signal, so this arm bypasses that shared embedding entirely via the
`encodes_raw_text` marker Mamba3LitModule checks for.
"""

import open_clip
import torch
from torch import nn


class CLIPPretrainedTextEncoder(nn.Module):
    encodes_raw_text = True

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        unfreeze_last_n_blocks: int = 4,
    ):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.clip_model = clip_model
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.d_model = clip_model.text_projection.shape[1]
        self._apply_freeze_policy(unfreeze_last_n_blocks)

    def _apply_freeze_policy(self, unfreeze_last_n_blocks: int) -> None:
        for param in self.clip_model.parameters():
            param.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            blocks = list(self.clip_model.transformer.resblocks)
            for block in blocks[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        for param in self.clip_model.ln_final.parameters():
            param.requires_grad = True
        self.clip_model.text_projection.requires_grad = True

    def forward(self, raw_texts: list[str]) -> torch.Tensor:
        device = self.clip_model.text_projection.device
        tokens = self.tokenizer(raw_texts).to(device)
        return self.clip_model.encode_text(tokens)