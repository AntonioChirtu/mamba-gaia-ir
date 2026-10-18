import torch
from src.models.mamba3_module import Mamba3LitModule


class RSITMDHeads(Mamba3LitModule):
    """Weights-only transfer, frozen encoders, fresh head optimizer."""

    def __init__(self, init_checkpoint: str, **kwargs):
        super().__init__(**kwargs)

        if self.dual_encoder is not None or self.hparams.compile:
            raise ValueError("Use separate, uncompiled encoders for this pilot.")

        # Load only your own trusted local checkpoints.
        state = torch.load(
            init_checkpoint, map_location="cpu", weights_only=False
        )["state_dict"]

        weight = state["text_embed.weight"]
        if weight.shape[1] != self.text_embed.embedding_dim:
            raise ValueError("Checkpoint/text-encoder width mismatch.")

        self.text_embed = torch.nn.Embedding(*weight.shape)
        self.load_state_dict(state, strict=True)

        self.requires_grad_(False)
        for head in (
            self.image_model.projection_head,
            self.proj1,
            self.proj2,
        ):
            head.requires_grad_(True)
        self.logit_scale.requires_grad_(True)

        self.save_hyperparameters({
            "init_checkpoint": init_checkpoint,
            "adaptation": "heads_only",
        })
        self.train(self.training)

        print("Loaded weights only; optimizer and epoch are NOT resumed.")
        print("Trainable:", [
            (name, p.numel())
            for name, p in self.named_parameters()
            if p.requires_grad
        ])

    def train(self, mode=True):
        super().train(mode)
        self.image_model.eval()
        self.text_model.eval()
        self.text_embed.eval()
        for head in (
            self.image_model.projection_head,
            self.proj1,
            self.proj2,
        ):
            head.train(mode)
        return self

    def setup(self, stage):
        if stage == "fit":
            vocab = len(self.trainer.datamodule.tokenizer)
            if vocab != self.text_embed.num_embeddings:
                raise ValueError(
                    f"Tokenizer has {vocab} tokens; checkpoint has "
                    f"{self.text_embed.num_embeddings}. "
                    "Refusing to replace pretrained embeddings."
                )
        super().setup(stage)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return {"optimizer": self.hparams.optimizer(params=params)}
