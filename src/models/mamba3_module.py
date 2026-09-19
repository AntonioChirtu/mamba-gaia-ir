import gc
from typing import Any, Dict, Tuple  # noqa: UP035

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MaxMetric, MeanMetric


def multi_positive_cross_entropy(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != positive_mask.shape:
        raise ValueError(
            f"Shape mismatch: logits={tuple(logits.shape)}, "
            f"mask={tuple(positive_mask.shape)}"
        )

    positive_counts = positive_mask.sum(dim=1)

    if (positive_counts == 0).any():
        bad_rows = (positive_counts == 0).nonzero(as_tuple=True)[0]
        raise RuntimeError(f"Rows without a positive target: {bad_rows.tolist()}")

    # Normalize each row across all its valid positives.
    targets = positive_mask.float() / positive_counts.unsqueeze(1).float()

    # Compute in float32 for numerical stability under mixed precision.
    log_probs = F.log_softmax(logits.float(), dim=1)

    return -(targets * log_probs).sum(dim=1).mean()


# class RetrievalRecallWrapper:
#     """
#     Wrapper for official TorchMetrics RetrievalRecall that handles similarity matrices.
#     """

#     def __init__(self, k=1):
#         self.k = k
#         self.mean_metric = MeanMetric()

#     def update(self, logits):
#         """
#         Update metric with similarity matrix.

#         Args:
#             logits: [batch_size, batch_size] similarity matrix
#         """
#         batch_size = logits.shape[0]
#         target = torch.arange(batch_size, device=logits.device)

#         # Get indices of top k matches
#         _, top_k_indices = logits.topk(self.k, dim=1)

#         # Check if target is in top_k (matches along dim 1)
#         correct = (top_k_indices == target.view(-1, 1)).any(dim=1)

#         # Ensure mean_metric is on same device as input
#         if self.mean_metric.device != logits.device:
#             self.mean_metric = self.mean_metric.to(logits.device)

#         # Update your internal MeanMetric
#         self.mean_metric.update(correct.float())

#     def compute(self):
#         return self.mean_metric.compute()

#     def reset(self):
#         self.mean_metric.reset()


class RetrievalRecallWrapper:
    def __init__(self, k=1):
        self.k = k
        self.mean_metric = MeanMetric()

    def update(
        self,
        logits: torch.Tensor,
        relevant: torch.Tensor | None = None,
    ):
        if relevant is None:
            relevant = torch.eye(
                logits.shape[0],
                logits.shape[1],
                dtype=torch.bool,
                device=logits.device,
            )

        if logits.shape != relevant.shape:
            raise ValueError(
                f"Shape mismatch: logits={tuple(logits.shape)}, "
                f"relevant={tuple(relevant.shape)}"
            )

        k = min(self.k, logits.shape[1])
        top_k_indices = logits.topk(k, dim=1).indices

        correct = relevant.gather(
            dim=1,
            index=top_k_indices,
        ).any(dim=1)

        if self.mean_metric.device != logits.device:
            self.mean_metric = self.mean_metric.to(logits.device)

        self.mean_metric.update(correct.float())

    def compute(self):
        return self.mean_metric.compute()

    def reset(self):
        self.mean_metric.reset()


class Mamba3LitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        image_net: torch.nn.Module,
        text_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        patch_size: int = 16,
        vocab_size: int = 50277,
        logit_scale_init: float = 0.07,
    ) -> None:
        """Initialize a `Mamba3LitModule`.

        :param image_net: The model to use for image processing.
        :param text_net: The model to use for text processing.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        d_model = image_net.d_model

        # Multiscale augmentation parameters
        self.base_size = 224
        # self.scale_factors = [0.75, 0.85, 1.0]
        self.current_size = self.base_size

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["image_net", "text_net"])

        self.image_model = image_net
        self.text_model = text_net

        # 1. Vision "Patch" Embedding: Turns [B, 3, 224, 224] -> [B, 196, d_model]
        self.patch_embed = torch.nn.Conv2d(
            3, image_net.d_model, kernel_size=patch_size, stride=patch_size
        )

        # 2. Text Embedding: Turns [B, 77] -> [B, 77, d_model]
        self.text_embed = torch.nn.Embedding(vocab_size, text_net.d_model)

        self.proj1 = (
            torch.nn.Identity()
            if image_net.d_model == 512
            else torch.nn.Linear(image_net.d_model, 512)
        )
        self.proj2 = torch.nn.Linear(text_net.d_model, 512)

        self.logit_scale = torch.nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / logit_scale_init))
        )

        self.val_outputs = {
            "img_embs": [],
            "txt_embs": [],
            "img_ids": [],
            "txt_img_ids": [],
        }

        # TODO: Make more complicated contrastive loss?
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        # Separate metrics for I2T and T2I
        self.train_i2t_r1 = RetrievalRecallWrapper(k=1)
        self.train_t2i_r1 = RetrievalRecallWrapper(k=1)

        self.val_batch_i2t_r1 = RetrievalRecallWrapper(k=1)
        self.val_batch_t2i_r1 = RetrievalRecallWrapper(k=1)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_i2t_r1_best = MaxMetric()
        self.val_t2i_r1_best = MaxMetric()
        self.val_mean_r1_best = MaxMetric()

        # Initialize test outputs storage
        self.test_outputs = {
            "img_embs": [],
            "txt_embs": [],
            "img_ids": [],
            "txt_img_ids": [],
        }

    def forward(
        self, x: torch.Tensor, modality="image", attention_mask=None, raw_texts=None
    ) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images, or token ids for text (ignored when the
            active text_net encodes raw text directly, e.g. CLIPPretrainedTextEncoder).
        :param raw_texts: Raw caption strings, required when
            `self.text_model.encodes_raw_text` is set.
        :return: A tensor of logits.
        """
        if modality == "image":
            # Pretrained vision wrapper or ViT-only model; expects [B, 3, 224, 224]
            if hasattr(self.image_model, "vision_encoder") or hasattr(
                self.image_model, "vit"
            ):
                out = self.image_model(x)
            else:
                # Original patch embedding approach
                # [B, 3, 224, 224] -> [B, d_model, 14, 14] -> [B, 196, d_model]
                x = self.patch_embed(x).flatten(2).transpose(1, 2)
                out = self.image_model(x)

            # ViT-only models already return [B, dim]; Mamba-style returns [B, L, dim]
            if out.dim() == 3:
                out = out[:, -1, :]
            out = self.proj1(out)
        else:
            if getattr(self.text_model, "encodes_raw_text", False):
                # Self-contained pretrained text tower: owns its own tokenizer
                # and embedding, bypasses self.text_embed entirely.
                out = self.text_model(raw_texts)
            else:
                # x is [B, seq_len] token_ids -> [B, seq_len, d_model]
                x = self.text_embed(x)
                out = self.text_model(
                    x, attention_mask=attention_mask
                )  # [B, L, d_model]

                if out.dim() == 3:
                    if attention_mask is not None:
                        # Pool the last NON-PAD token (tokenizer right-pads), instead of
                        # the last position which would be a PAD token.
                        lengths = attention_mask.long().sum(dim=1) - 1  # [B]
                        lengths = lengths.clamp(min=0)
                        idx = lengths.view(-1, 1, 1).expand(-1, 1, out.size(-1))
                        out = out.gather(1, idx).squeeze(1)  # [B, d_model]
                    else:
                        out = out[:, -1, :]
            out = self.proj2(out)

        return out

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""

        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_i2t_r1_best.reset()
        self.val_t2i_r1_best.reset()
        self.val_mean_r1_best.reset()
        self.val_batch_i2t_r1.reset()
        self.val_batch_t2i_r1.reset()

    def model_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        raw_texts=None,
    ):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param raw_texts: Raw caption strings, forwarded to the text encoder when the
            active text_next encodes raw text directly

        :return: None if it explodes, else:
        A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        images, texts, attention_mask = batch

        img_emb = self.forward(images, modality="image")
        txt_emb = self.forward(
            texts, modality="text", attention_mask=attention_mask, raw_texts=raw_texts
        )

        img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=-1)
        txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=-1)

        with torch.no_grad():
            # Clamping prevents the exponential matrix from blowing up to Infinity
            self.logit_scale.clamp_(max=4.6052)

        # Scaling by temperature
        scale = self.logit_scale.exp().clamp(max=100)
        logits_i2t = (img_emb @ txt_emb.t()) * scale
        logits_t2i = logits_i2t.t()

        # Each row's positive is the diagonal, so plain InfoNCE is fine
        y = torch.arange(logits_i2t.shape[0], device=logits_i2t.device)
        loss_i2t = self.criterion(logits_i2t, y)
        loss_t2i = self.criterion(logits_t2i, y)

        loss = 0.5 * (loss_i2t + loss_t2i)

        # Defensive check
        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"WARNING: NaN/Inf loss detected! loss={loss.item()}; skipping batch!"
            )
            return None

        return loss, logits_i2t, logits_t2i, img_emb, txt_emb

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        images, texts, attention_mask, captions = batch

        # # Multiscale Augmentation Trigger
        # if batch_idx % 10 == 0:
        #     scale = random.choice(self.scale_factors)
        #     new_size = int(self.base_size * scale)
        #     self.current_size = (new_size // 16) * 16
        #
        # # Resize Images
        # if images.shape[-1] != self.current_size:
        #     images = torch.nn.functional.interpolate(
        #         images, size=(self.current_size, self.current_size),
        #         mode='bicubic', align_corners=False
        #     )

        # The model handles the embedding interpolation internally now!
        result = self.model_step((images, texts, attention_mask), raw_texts=captions)

        # Catch the NaN guard signal
        if result is None:
            return self.logit_scale * 0.0
        loss, l_i2t, l_t2i, _, _ = result

        # Update separate I2T and T2I metrics
        self.train_i2t_r1.update(l_i2t)
        self.train_t2i_r1.update(l_t2i)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        train_i2t_r1 = torch.as_tensor(
            self.train_i2t_r1.compute(), device=self.device, dtype=torch.float32
        )
        train_t2i_r1 = torch.as_tensor(
            self.train_t2i_r1.compute(), device=self.device, dtype=torch.float32
        )

        self.log("train/I2T_R1", train_i2t_r1, prog_bar=True, sync_dist=True)
        self.log("train/T2I_R1", train_t2i_r1, prog_bar=True, sync_dist=True)
        self.train_i2t_r1.reset()
        self.train_t2i_r1.reset()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # 1. Use your model_step for the loss (keep it consistent!)
        # texts/attention_mask are [B, C, L] (C = captions per image, standard
        # 5-caption protocol); anchor caption (index 0) drives loss/batch-r1
        # exactly as before, all C captions are encoded for the epoch-end
        # multi-relevant recall computation.
        images, image_ids, texts, attention_mask, text_strings = batch

        if hasattr(self.image_model, "vit"):
            self.image_model.vit.image_size = self.base_size

        B, C, L = texts.shape
        anchor_texts = texts[:, 0, :]
        anchor_mask = attention_mask[:, 0, :]
        anchor_texts_str = text_strings[0::C]  # index 0 of each image's C captions

        result = self.model_step(
            (images, anchor_texts, anchor_mask), raw_texts=anchor_texts_str
        )

        # If validation batch is broken, exit early to protect global metric tracking
        if result is None:
            return
        loss, l_i2t, l_t2i, img_emb, txt_emb = result

        self.val_batch_i2t_r1.update(l_i2t)
        self.val_batch_t2i_r1.update(l_t2i)

        if img_emb.ndim == 1:
            img_emb = img_emb.unsqueeze(0)

        # Encode all C captions per image for the full multi-relevant recall
        texts_flat = texts.reshape(B * C, L)
        mask_flat = attention_mask.reshape(B * C, L)
        txt_emb_all = self.forward(
            texts_flat,
            modality="text",
            attention_mask=mask_flat,
            raw_texts=text_strings,
        )
        txt_emb_all = torch.nn.functional.normalize(txt_emb_all, p=2, dim=-1)

        txt_img_ids = image_ids.to(img_emb.device).repeat_interleave(C)

        self.val_outputs["img_embs"].append(img_emb.detach().cpu())
        self.val_outputs["txt_embs"].append(txt_emb_all.detach().cpu())
        self.val_outputs["img_ids"].append(image_ids.detach().cpu())
        self.val_outputs["txt_img_ids"].append(txt_img_ids.detach().cpu())

        self.val_loss.update(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        # Safety guard
        has_local = torch.tensor(
            [bool(self.val_outputs["img_embs"])], device=self.device, dtype=torch.bool
        )
        has_outputs = self.all_gather(has_local).flatten()

        if not has_outputs.any():
            return
        if not has_outputs.all():
            raise RuntimeError("Some DDP ranks produced no validation embeddings")

        # Gather tensors
        local_img = torch.cat(self.val_outputs["img_embs"]).to(self.device)
        local_txt = torch.cat(self.val_outputs["txt_embs"]).to(self.device)
        local_img_ids = torch.cat(self.val_outputs["img_ids"]).to(self.device)
        local_txt_img_ids = torch.cat(self.val_outputs["txt_img_ids"]).to(self.device)

        local_n_img = torch.tensor([local_img.shape[0]], device=self.device)
        local_n_txt = torch.tensor([local_txt.shape[0]], device=self.device)
        rank_n_img = self.all_gather(local_n_img).flatten()
        rank_n_txt = self.all_gather(local_n_txt).flatten()

        if not torch.all(rank_n_img == rank_n_img[0]):
            raise RuntimeError(
                f"Unequal image counts across ranks: {rank_n_img.tolist()}"
            )
        if not torch.all(rank_n_txt == rank_n_txt[0]):
            raise RuntimeError(
                f"Unequal text counts across ranks: {rank_n_txt.tolist()}"
            )

        all_img = self.all_gather(local_img, sync_grads=False).reshape(
            -1, local_img.shape[-1]
        )
        all_txt = self.all_gather(local_txt, sync_grads=False).reshape(
            -1, local_txt.shape[-1]
        )

        all_img_ids = self.all_gather(local_img_ids).reshape(-1)
        all_txt_img_ids = self.all_gather(local_txt_img_ids).reshape(-1)

        # Translate gathered Dataset indexes into stable GAIA identifiers.
        val_dataset = self.trainer.datamodule.data_val

        all_image_metadata = [
            val_dataset.get_retrieval_metadata(int(dataset_idx))
            for dataset_idx in all_img_ids.detach().cpu().tolist()
        ]

        all_sample_ids = [metadata["sample_id"] for metadata in all_image_metadata]
        all_group_ids = [metadata["group_id"] for metadata in all_image_metadata]

        if all_img.shape[0] == 0:
            raise RuntimeError("No gathered image embeddings.")
        if all_txt.shape[0] % all_img.shape[0] != 0:
            raise RuntimeError(
                "Text embedding count is not divisible by image count: "
                f"{all_txt.shape[0]} texts vs {all_img.shape[0]} images"
            )

        captions_per_image = all_txt.shape[0] // all_img.shape[0]

        all_text_ids = []
        all_text_group_ids = []
        all_text_valid_mask = []

        for metadata in all_image_metadata:
            text_ids = list(metadata["text_ids"])
            num_valid = metadata["num_valid_captions"]

            while len(text_ids) < captions_per_image:
                padding_position = len(text_ids)
                text_ids.append(f"{metadata['sample_id']}:padding:{padding_position}")

            text_ids = text_ids[:captions_per_image]

            all_text_ids.extend(text_ids)
            all_text_group_ids.extend([metadata["group_id"]] * captions_per_image)
            all_text_valid_mask.extend(
                [
                    caption_position < num_valid
                    for caption_position in range(captions_per_image)
                ]
            )

        if len(all_sample_ids) != all_img.shape[0]:
            raise RuntimeError(
                "Stable image ID count does not match image embeddings: "
                f"{len(all_sample_ids)} IDs vs "
                f"{all_img.shape[0]} embeddings"
            )

        if len(all_text_ids) != all_txt.shape[0]:
            raise RuntimeError(
                "Stable text ID count does not match text embeddings: "
                f"{len(all_text_ids)} IDs vs "
                f"{all_txt.shape[0]} embeddings"
            )

        if all_img.ndim != 2 or all_txt.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D embeddings, got "
                f"image={tuple(all_img.shape)}, text={tuple(all_txt.shape)}"
            )

        # Normalize and compute Global Similarity Matrix
        all_img = torch.nn.functional.normalize(all_img, p=2, dim=-1)
        all_txt = torch.nn.functional.normalize(all_txt, p=2, dim=-1)

        image_norms = all_img.norm(dim=1)
        text_norms = all_txt.norm(dim=1)

        if not torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-4):
            raise RuntimeError("Image embeddings are not L2-normalized.")
        if not torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-4):
            raise RuntimeError("Text embeddings are not L2-normalized.")

        local_strings = list(self.val_outputs["raw_texts"])

        if dist.is_available() and dist.is_initialized():
            gathered_strings = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_strings, local_strings)
            all_strings = [
                text for rank_strings in gathered_strings for text in rank_strings
            ]
        else:
            all_strings = local_strings

        sim_matrix = all_img @ all_txt.t()

        if sim_matrix.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D similarity matrix, got {sim_matrix.shape}"
            )

        # Multi-relevant ground truth: text j is relevant to image i iff they
        # share the same source-image id (standard 5-captions-per-image
        # protocol), not just the diagonal
        relevant = all_img_ids.unsqueeze(1) == all_txt_img_ids.unsqueeze(
            0
        )  # [N_img, N_txt]

        if relevant.shape != sim_matrix.shape:
            raise RuntimeError(
                f"Relevance/similarity mismatch: "
                f"relevant={tuple(relevant.shape)}, "
                f"similarity={tuple(sim_matrix.shape)}"
            )

        if not relevant.any(dim=1).all():
            raise RuntimeError("At least one image has no relevant caption.")
        if not relevant.any(dim=0).all():
            raise RuntimeError("At least one caption has no owning image.")

        valid_text_mask_tensor = torch.as_tensor(
            all_text_valid_mask, dtype=torch.bool, device=sim_matrix.device
        )

        # I2T:
        # - every image is a query;
        # - padded text candidates are removed.
        i2t_scores = sim_matrix[:, valid_text_mask_tensor]
        i2t_relevant = relevant[:, valid_text_mask_tensor]

        # T2I:
        # - every valid text is a query;
        # - every image is a candidate.
        t2i_scores = sim_matrix[:, valid_text_mask_tensor].t()
        t2i_relevant = relevant[:, valid_text_mask_tensor].t()

        i2t_ranking = self._compute_ranking_metrics(i2t_scores, i2t_relevant)
        t2i_ranking = self._compute_ranking_metrics(t2i_scores, t2i_relevant)

        # 3. Calculate R@1, R@5, R@10 for both directions
        val_results = {}
        for k in [1, 5, 10, 20]:
            # Image-to-text over valid caption candidates only.
            kk_i2t = min(k, i2t_scores.shape[1])
            top_k_i2t = i2t_scores.topk(kk_i2t, dim=1).indices
            r_i2t = i2t_relevant.gather(1, top_k_i2t).any(dim=1).float().mean()
            val_results[f"val/I2T_R{k}"] = r_i2t

            # Text-to-image over valid text queries only.
            kk_t2i = min(k, t2i_scores.shape[1])
            top_k_t2i = t2i_scores.topk(kk_t2i, dim=1).indices
            r_t2i = t2i_relevant.gather(1, top_k_t2i).any(dim=1).float().mean()
            val_results[f"val/T2I_R{k}"] = r_t2i

        val_results["val/mean_R1"] = 0.5 * (
            val_results["val/I2T_R1"] + val_results["val/T2I_R1"]
        )

        for metric_name, value in i2t_ranking.items():
            val_results[f"val/I2T_{metric_name}"] = value
        for metric_name, value in t2i_ranking.items():
            val_results[f"val/T2I_{metric_name}"] = value

        for metric_name in ["MRR", "mAP", "nDCG"]:
            val_results[f"val/mean_{metric_name}"] = 0.5 * (
                i2t_ranking[metric_name] + t2i_ranking[metric_name]
            )

        # 4. Log all metrics to WandB/Progress Bar
        self.log_dict(
            val_results, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # 5. Update "Best" trackers (Usually tracked via R1)
        self.val_i2t_r1_best(val_results["val/I2T_R1"])
        self.val_t2i_r1_best(val_results["val/T2I_R1"])
        self.val_mean_r1_best(val_results["val/mean_R1"])
        self.log("val/I2T_R1_best", self.val_i2t_r1_best.compute(), sync_dist=True)
        self.log("val/T2I_R1_best", self.val_t2i_r1_best.compute(), sync_dist=True)
        self.log("val/mean_R1_best", self.val_mean_r1_best.compute(), sync_dist=True)

        batch_i2t_r1 = self.val_batch_i2t_r1.compute()
        batch_t2i_r1 = self.val_batch_t2i_r1.compute()

        # Ensure Lightning receives tensors rather than wrapper objects.
        batch_i2t_r1 = torch.as_tensor(
            batch_i2t_r1, device=self.device, dtype=torch.float32
        )
        batch_t2i_r1 = torch.as_tensor(
            batch_t2i_r1, device=self.device, dtype=torch.float32
        )

        self.log_dict(
            {"val/batch_I2T_R1": batch_i2t_r1, "val/batch_T2I_R1": batch_t2i_r1},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        # 6. Save visual results table
        if self.trainer.is_global_zero:
            self._save_retrieval_diagnostics(
                sim_matrix=sim_matrix,
                img_ids=all_img_ids,
                txt_img_ids=all_txt_img_ids,
                sample_ids=all_sample_ids,
                group_ids=all_group_ids,
                text_ids=all_text_ids,
                text_group_ids=all_text_group_ids,
                text_valid_mask=all_text_valid_mask,
                metadata=all_image_metadata,
                phase="val",
                top_k=10,
                max_queries_per_direction=15,
            )
            # self._save_misc_metrics(
            #     sim_matrix, all_strings, all_img_ids, all_txt_img_ids, phase="val"
            # )

        # # Diagnostic to check DistributedSampler repeating samples
        # if self.trainer.is_global_zero:
        #     dataset_size = len(self.trainer.datamodule.val_dataloader().dataset)
        #     world_size = self.trainer.world_size

        #     if dataset_size % world_size != 0:
        #         print(
        #             f"Warning: validation size {dataset_size} is not divisible "
        #             f"by world size {world_size}; DDP may duplicate samples."
        #         )

        # 7. Reset storage for the next epoch
        self.val_outputs = {
            "img_embs": [],
            "txt_embs": [],
            "img_ids": [],
            "txt_img_ids": [],
        }

        self.val_batch_i2t_r1.reset()
        self.val_batch_t2i_r1.reset()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        images, image_ids, texts, attention_mask, text_strings = batch

        if hasattr(self.image_model, "vit"):
            self.image_model.vit.image_size = self.base_size

        B, C, L = texts.shape
        anchor_texts = texts[:, 0, :]
        anchor_mask = attention_mask[:, 0, :]
        anchor_texts_str = text_strings[0::C]  # index 0 of each image's C captions

        result = self.model_step(
            (images, anchor_texts, anchor_mask), raw_texts=anchor_texts_str
        )

        # If validation batch is broken, exit early to protect global metric tracking
        if result is None:
            return
        loss, l_i2t, l_t2i, img_emb, txt_emb = result

        if img_emb.ndim == 1:
            img_emb = img_emb.unsqueeze(0)

        texts_flat = texts.reshape(B * C, L)
        mask_flat = attention_mask.reshape(B * C, L)
        txt_emb_all = self.forward(
            texts_flat,
            modality="text",
            attention_mask=mask_flat,
            raw_texts=text_strings,
        )
        txt_emb_all = torch.nn.functional.normalize(txt_emb_all, p=2, dim=-1)

        txt_img_ids = image_ids.to(img_emb.device).repeat_interleave(C)

        self.test_outputs["img_embs"].append(img_emb.detach().cpu())
        self.test_outputs["txt_embs"].append(txt_emb_all.detach().cpu())
        self.test_outputs["img_ids"].append(image_ids.detach().cpu())
        self.test_outputs["txt_img_ids"].append(txt_img_ids.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        has_local = torch.tensor(
            [bool(self.test_outputs["img_embs"])], device=self.device, dtype=torch.bool
        )
        has_outputs = self.all_gather(has_local).flatten()

        if not has_outputs.any():
            return
        if not has_outputs.all():
            raise RuntimeError("Some DDP ranks produced no test embeddings")

        local_img = torch.cat(self.test_outputs["img_embs"]).to(self.device)
        local_txt = torch.cat(self.test_outputs["txt_embs"]).to(self.device)
        local_img_ids = torch.cat(self.test_outputs["img_ids"]).to(self.device)
        local_txt_img_ids = torch.cat(self.test_outputs["txt_img_ids"]).to(self.device)

        if local_img.ndim != 2 or local_txt.ndim != 2:
            raise RuntimeError(
                f"Expected local [N, D] embeddings, got "
                f"image={tuple(local_img.shape)}, "
                f"text={tuple(local_txt.shape)}"
            )

        local_n_img = torch.tensor([local_img.shape[0]], device=self.device)
        local_n_txt = torch.tensor([local_txt.shape[0]], device=self.device)
        rank_n_img = self.all_gather(local_n_img).flatten()
        rank_n_txt = self.all_gather(local_n_txt).flatten()

        if not torch.all(rank_n_img == rank_n_img[0]):
            raise RuntimeError(
                f"Unequal image counts across ranks: {rank_n_img.tolist()}"
            )
        if not torch.all(rank_n_txt == rank_n_txt[0]):
            raise RuntimeError(
                f"Unequal text counts across ranks: {rank_n_txt.tolist()}"
            )

        all_img = self.all_gather(local_img, sync_grads=False).reshape(
            -1, local_img.shape[-1]
        )
        all_txt = self.all_gather(local_txt, sync_grads=False).reshape(
            -1, local_txt.shape[-1]
        )

        all_img_ids = self.all_gather(local_img_ids).reshape(-1)
        all_txt_img_ids = self.all_gather(local_txt_img_ids).reshape(-1)

        # Translate gathered Dataset indexes into stable GAIA identifiers.
        test_dataset = self.trainer.datamodule.data_test

        all_image_metadata = [
            test_dataset.get_retrieval_metadata(int(dataset_idx))
            for dataset_idx in all_img_ids.detach().cpu().tolist()
        ]

        all_sample_ids = [metadata["sample_id"] for metadata in all_image_metadata]
        all_group_ids = [metadata["group_id"] for metadata in all_image_metadata]

        if all_img.shape[0] == 0:
            raise RuntimeError("No gathered image embeddings.")
        if all_txt.shape[0] % all_img.shape[0] != 0:
            raise RuntimeError(
                "Text embedding count is not divisible by image count: "
                f"{all_txt.shape[0]} texts vs {all_img.shape[0]} images"
            )

        captions_per_image = all_txt.shape[0] // all_img.shape[0]

        all_text_ids = []
        all_text_group_ids = []
        all_text_valid_mask = []

        for metadata in all_image_metadata:
            text_ids = list(metadata["text_ids"])
            num_valid = metadata["num_valid_captions"]

            while len(text_ids) < captions_per_image:
                padding_position = len(text_ids)
                text_ids.append(f"{metadata['sample_id']}:padding:{padding_position}")

            text_ids = text_ids[:captions_per_image]

            all_text_ids.extend(text_ids)
            all_text_group_ids.extend([metadata["group_id"]] * captions_per_image)
            all_text_valid_mask.extend(
                [
                    caption_position < num_valid
                    for caption_position in range(captions_per_image)
                ]
            )

        if len(all_sample_ids) != all_img.shape[0]:
            raise RuntimeError(
                "Stable image ID count does not match image embeddings: "
                f"{len(all_sample_ids)} IDs vs "
                f"{all_img.shape[0]} embeddings"
            )

        if len(all_text_ids) != all_txt.shape[0]:
            raise RuntimeError(
                "Stable text ID count does not match text embeddings: "
                f"{len(all_text_ids)} IDs vs "
                f"{all_txt.shape[0]} embeddings"
            )

        if all_img.ndim != 2 or all_txt.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D embeddings, got "
                f"image={tuple(all_img.shape)}, text={tuple(all_txt.shape)}"
            )

        # 2. Normalize and compute Global Similarity Matrix
        all_img = torch.nn.functional.normalize(all_img, p=2, dim=-1)
        all_txt = torch.nn.functional.normalize(all_txt, p=2, dim=-1)

        image_norms = all_img.norm(dim=1)
        text_norms = all_txt.norm(dim=1)

        if not torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-4):
            raise RuntimeError("Image embeddings are not L2-normalized.")
        if not torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-4):
            raise RuntimeError("Text embeddings are not L2-normalized.")

        import torch.distributed as dist

        local_strings = list(self.test_outputs["raw_texts"])

        if dist.is_available() and dist.is_initialized():
            gathered_strings = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_strings, local_strings)
            all_strings = [
                text for rank_strings in gathered_strings for text in rank_strings
            ]
        else:
            all_strings = local_strings

        sim_matrix = all_img @ all_txt.t()

        if sim_matrix.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D similarity matrix, got {sim_matrix.shape}"
            )

        # Multi-relevant ground truth: text j is relevant to image i iff they
        # share the same source-img id (standard 5-captions-per-image
        # protocol), not just the diagonal
        relevant = all_img_ids.unsqueeze(1) == all_txt_img_ids.unsqueeze(
            0
        )  # [N_img, N_txt]

        if relevant.shape != sim_matrix.shape:
            raise RuntimeError(
                f"Relevance/similarity mismatch: "
                f"relevant={tuple(relevant.shape)}, "
                f"similarity={tuple(sim_matrix.shape)}"
            )

        if not relevant.any(dim=1).all():
            raise RuntimeError("At least one image has no relevant caption.")

        if not relevant.any(dim=0).all():
            raise RuntimeError("At least one caption has no owning image.")

        valid_text_mask_tensor = torch.as_tensor(
            all_text_valid_mask, dtype=torch.bool, device=sim_matrix.device
        )

        i2t_scores = sim_matrix[:, valid_text_mask_tensor]
        i2t_relevant = relevant[:, valid_text_mask_tensor]

        t2i_scores = sim_matrix[:, valid_text_mask_tensor].t()
        t2i_relevant = relevant[:, valid_text_mask_tensor].t()

        i2t_ranking = self._compute_ranking_metrics(i2t_scores, i2t_relevant)
        t2i_ranking = self._compute_ranking_metrics(t2i_scores, t2i_relevant)

        # 3. Calculate R@1, R@5, R@10 for both directions
        test_results = {}
        for k in [1, 5, 10, 20]:
            kk_i2t = min(k, i2t_scores.shape[1])
            top_k_i2t = i2t_scores.topk(kk_i2t, dim=1).indices
            r_i2t = i2t_relevant.gather(1, top_k_i2t).any(dim=1).float().mean()
            test_results[f"test/I2T_R{k}"] = r_i2t

            kk_t2i = min(k, t2i_scores.shape[1])
            top_k_t2i = t2i_scores.topk(kk_t2i, dim=1).indices
            r_t2i = t2i_relevant.gather(1, top_k_t2i).any(dim=1).float().mean()
            test_results[f"test/T2I_R{k}"] = r_t2i

        test_results["test/mean_R1"] = 0.5 * (
            test_results["test/I2T_R1"] + test_results["test/T2I_R1"]
        )

        for metric_name, value in i2t_ranking.items():
            test_results[f"test/I2T_{metric_name}"] = value
        for metric_name, value in t2i_ranking.items():
            test_results[f"test/T2I_{metric_name}"] = value

        for metric_name in ["MRR", "mAP", "nDCG"]:
            test_results[f"test/mean_{metric_name}"] = 0.5 * (
                i2t_ranking[metric_name] + t2i_ranking[metric_name]
            )

        # 4. Log all metrics to WandB/Progress Bar
        self.log_dict(
            test_results, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # 6. Save visual results table
        if self.trainer.is_global_zero:
            self._save_retrieval_diagnostics(
                sim_matrix=sim_matrix,
                img_ids=all_img_ids,
                txt_img_ids=all_txt_img_ids,
                sample_ids=all_sample_ids,
                group_ids=all_group_ids,
                text_ids=all_text_ids,
                text_group_ids=all_text_group_ids,
                text_valid_mask=all_text_valid_mask,
                metadata=all_image_metadata,
                phase="test",
                top_k=10,
                max_queries_per_direction=15,
            )
            # self._save_misc_metrics(
            #     sim_matrix, all_strings, all_img_ids, all_txt_img_ids, phase="test"
            # )

        # 7. Reset storage for the next epoch
        self.test_outputs = {
            "img_embs": [],
            "txt_embs": [],
            "img_ids": [],
            "txt_img_ids": [],
        }

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        # --- DYNAMIC VOCABSIZE AUTO-PATCH ---
        # Look across to see if a trainer and a datamodule with a tokenizer exist
        if (
            stage == "fit"
            and self.trainer
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "tokenizer")
        ):
            datamodule_tokenizer = self.trainer.datamodule.tokenizer
            actual_vocab_size = len(datamodule_tokenizer)

            # Check if our current embedding layer is too small or mismatched
            if self.text_embed.num_embeddings != actual_vocab_size:
                print(
                    f"🔄 Auto-Patching Text Embedding Matrix: "
                    f"{self.text_embed.num_embeddings} ➡️ {actual_vocab_size} rows "
                    f"to match the data tokenizer."
                )

                # Re-initialize the embedding layer with the exact vocabulary shape required
                self.text_embed = torch.nn.Embedding(
                    actual_vocab_size, self.text_embed.embedding_dim
                )

        if self.hparams.compile and stage == "fit":
            self.image_model = torch.compile(self.image_model)
            self.text_model = torch.compile(self.text_model)

    # def load_state_dict(self, state_dict, strict: bool = True):
    #     """
    #     Override the default load behavior to force strict=False.
    #     This prevents crashes when loading checkpoints that exclude frozen/pretrained backbones.
    #     """
    #     print("🔓 Intercepted state_dict load: Forcing strict=False to protect initialized vision weights.")
    #     return super().load_state_dict(state_dict, strict=False)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mean_R1",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def _save_retrieval_diagnostics(
        self,
        sim_matrix,
        img_ids,
        txt_img_ids,
        sample_ids,
        group_ids,
        text_ids,
        text_group_ids,
        text_valid_mask,
        metadata,
        phase="val",
        top_k=10,
        max_queries_per_direction=15,
    ):
        """Log stable-ID retrieval diagnostics for I2T and T2I."""

        if not isinstance(self.logger, WandbLogger):
            return

        num_images, num_texts = sim_matrix.shape

        # ------------------------------------------------------------------
        # Validate alignment
        # ------------------------------------------------------------------
        if len(sample_ids) != num_images:
            raise RuntimeError(
                "sample_ids/image count mismatch: "
                f"{len(sample_ids)} IDs vs {num_images} images"
            )

        if len(group_ids) != num_images:
            raise RuntimeError(
                "group_ids/image count mismatch: "
                f"{len(group_ids)} IDs vs {num_images} images"
            )

        if len(metadata) != num_images:
            raise RuntimeError(
                "metadata/image count mismatch: "
                f"{len(metadata)} rows vs {num_images} images"
            )

        if len(text_ids) != num_texts:
            raise RuntimeError(
                "text_ids/text count mismatch: "
                f"{len(text_ids)} IDs vs {num_texts} texts"
            )

        if len(text_group_ids) != num_texts:
            raise RuntimeError(
                "text_group_ids/text count mismatch: "
                f"{len(text_group_ids)} IDs vs {num_texts} texts"
            )

        if len(text_valid_mask) != num_texts:
            raise RuntimeError(
                "text_valid_mask/text count mismatch: "
                f"{len(text_valid_mask)} flags vs {num_texts} texts"
            )

        valid_text_mask = torch.as_tensor(
            text_valid_mask, dtype=torch.bool, device=sim_matrix.device
        )

        if not valid_text_mask.any():
            raise RuntimeError(
                "No valid text entries are available for retrieval diagnostics."
            )

        columns = [
            "Retrieval_Direction",
            "Query_ID",
            "Query_Group_ID",
            "Positive_IDs",
            "Positive_Rank",
            "TopK_Retrieved_IDs",
            "TopK_Cosine_Scores",
            "Hardest_Negative_ID",
            "Hardest_Negative_Cosine",
            "Top1_Correct",
            "Location",
        ]

        table = wandb.Table(columns=columns)

        # ------------------------------------------------------------------
        # Image-to-text diagnostics
        # ------------------------------------------------------------------
        num_i2t_queries = min(max_queries_per_direction, num_images)

        for image_position in range(num_i2t_queries):
            scores = sim_matrix[image_position].float()

            positive_mask = (txt_img_ids == img_ids[image_position]) & valid_text_mask

            if not positive_mask.any():
                raise RuntimeError(
                    "Image query has no valid positive captions: "
                    f"position={image_position}, "
                    f"sample_id={sample_ids[image_position]}"
                )

            # Exclude padded captions from the ranking.
            ranking_scores = scores.clone()
            ranking_scores[~valid_text_mask] = -torch.inf

            ranked_indices = torch.argsort(ranking_scores, descending=True)
            ranked_indices = ranked_indices[valid_text_mask[ranked_indices]]

            positive_positions = torch.nonzero(
                positive_mask[ranked_indices], as_tuple=True
            )[0]

            positive_rank = int(positive_positions[0].item() + 1)
            positive_indices = torch.nonzero(positive_mask, as_tuple=True)[0]
            positive_ids = [text_ids[int(index.item())] for index in positive_indices]

            current_top_k = min(top_k, ranked_indices.numel())

            top_indices = ranked_indices[:current_top_k]
            top_ids = [text_ids[int(index.item())] for index in top_indices]
            top_scores = [
                float(scores[int(index.item())].item()) for index in top_indices
            ]

            negative_ranked_indices = ranked_indices[~positive_mask[ranked_indices]]

            if negative_ranked_indices.numel() > 0:
                hardest_negative_position = int(negative_ranked_indices[0].item())
                hardest_negative_id = text_ids[hardest_negative_position]
                hardest_negative_score = float(scores[hardest_negative_position].item())
            else:
                hardest_negative_id = None
                hardest_negative_score = None

            top1_correct = bool(positive_mask[ranked_indices[0]].item())

            table.add_data(
                "I2T",
                sample_ids[image_position],
                group_ids[image_position],
                positive_ids,
                positive_rank,
                top_ids,
                top_scores,
                hardest_negative_id,
                hardest_negative_score,
                top1_correct,
                metadata[image_position].get("location"),
            )

        # ------------------------------------------------------------------
        # Text-to-image diagnostics
        # ------------------------------------------------------------------
        valid_text_positions = torch.nonzero(valid_text_mask, as_tuple=True)[0]

        num_t2i_queries = min(
            max_queries_per_direction,
            valid_text_positions.numel(),
        )

        for query_number in range(num_t2i_queries):
            text_position = int(valid_text_positions[query_number].item())

            scores = sim_matrix[:, text_position].float()

            positive_mask = img_ids == txt_img_ids[text_position]

            if not positive_mask.any():
                raise RuntimeError(
                    "Text query has no owning image: "
                    f"text_position={text_position}, "
                    f"text_id={text_ids[text_position]}"
                )

            ranked_indices = torch.argsort(scores, descending=True)

            positive_positions = torch.nonzero(
                positive_mask[ranked_indices], as_tuple=True
            )[0]
            positive_rank = int(positive_positions[0].item() + 1)
            positive_indices = torch.nonzero(positive_mask, as_tuple=True)[0]
            positive_ids = [sample_ids[int(index.item())] for index in positive_indices]

            current_top_k = min(top_k, ranked_indices.numel())

            top_indices = ranked_indices[:current_top_k]
            top_ids = [sample_ids[int(index.item())] for index in top_indices]
            top_scores = [
                float(scores[int(index.item())].item()) for index in top_indices
            ]

            negative_ranked_indices = ranked_indices[~positive_mask[ranked_indices]]

            if negative_ranked_indices.numel() > 0:
                hardest_negative_position = int(negative_ranked_indices[0].item())
                hardest_negative_id = sample_ids[hardest_negative_position]
                hardest_negative_score = float(scores[hardest_negative_position].item())
            else:
                hardest_negative_id = None
                hardest_negative_score = None

            top1_correct = bool(positive_mask[ranked_indices[0]].item())

            # Find metadata for this caption's owning image.
            owner_positions = torch.nonzero(positive_mask, as_tuple=True)[0]
            owner_position = int(owner_positions[0].item())

            table.add_data(
                "T2I",
                text_ids[text_position],
                text_group_ids[text_position],
                positive_ids,
                positive_rank,
                top_ids,
                top_scores,
                hardest_negative_id,
                hardest_negative_score,
                top1_correct,
                metadata[owner_position].get("location"),
            )

        self.logger.experiment.log({f"{phase}/retrieval_diagnostics": table})

    @torch.no_grad()
    def _save_misc_metrics(
        self, sim_matrix, all_texts, img_ids, txt_img_ids, phase="val"
    ):
        if not isinstance(self.logger, WandbLogger):
            return

        num_candidates = sim_matrix.shape[1]
        num_samples = min(15, sim_matrix.shape[0])

        if len(all_texts) != num_candidates:
            raise RuntimeError(
                f"Expected {num_candidates} captions, got {len(all_texts)}"
            )

        query_sim = sim_matrix[:num_samples].float()
        relevant = img_ids[:num_samples].unsqueeze(1) == txt_img_ids.unsqueeze(
            0
        )  # [num_samples, N_txt]

        # Top-1 and top-2 cosine similarities
        k = min(2, num_candidates)
        top_scores, top_indices = query_sim.topk(k, dim=1)

        top1_cosine = top_scores[:, 0]
        top1_indices = top_indices[:, 0]

        if k == 2:
            top2_cosine = top_scores[:, 1]
            cosine_margin = top1_cosine - top2_cosine
        else:
            top2_cosine = torch.full_like(top1_cosine, float("nan"))
            cosine_margin = torch.full_like(top1_cosine, float("nan"))

        # "True" cosine = best similarity among this image's relevant (ground truth) captions
        masked_sim = query_sim.masked_fill(~relevant, float("-inf"))
        true_cosine, true_gt_idx = masked_sim.max(dim=1)

        # Rank of the best-matching true caption: 1 means correct top-1
        true_rank = (query_sim > true_cosine[:, None]).sum(dim=1) + 1

        # Temperature-scaled score, consistent with training logits
        scale = self.logit_scale.detach().exp().clamp(max=100)
        scaled_logits = scale * query_sim
        probabilities = torch.softmax(scaled_logits, dim=1)
        top1_softmax_score = probabilities.gather(1, top1_indices[:, None]).squeeze(1)

        columns = [
            "Image_Index",
            "True_Caption",
            "Model_Top_Pick",
            "True_Rank",
            "Top1_Cosine",
            "True_Cosine",
            "Top2_Cosine",
            "Cosine_Margin",
            "Top1_Softmax_Score",
            "Correct",
        ]
        table = wandb.Table(columns=columns)

        for i in range(num_samples):
            predicted_idx = top1_indices[i].item()
            true_idx = true_gt_idx[i].item()

            table.add_data(
                i,
                all_texts[true_idx],
                all_texts[predicted_idx],
                true_rank[i].item(),
                top1_cosine[i].item(),
                true_cosine[i].item(),
                top2_cosine[i].item(),
                cosine_margin[i].item(),
                top1_softmax_score[i].item(),
                bool(relevant[i, predicted_idx].item()),
            )

        self.logger.experiment.log({f"{phase}/predictions_detailed": table})

    @staticmethod
    def _compute_ranking_metrics(
        scores: torch.Tensor,
        relevant: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute binary-relevance ranking metrics for one direction."""

        if scores.ndim != 2 or relevant.ndim != 2:
            raise RuntimeError("scores and relevant must both be 2-D.")

        if scores.shape != relevant.shape:
            raise RuntimeError(
                f"Ranking shape mismatch: "
                f"scores={tuple(scores.shape)}, "
                f"relevant={tuple(relevant.shape)}"
            )

        if not relevant.any(dim=1).all():
            raise RuntimeError("At least one query has no relevant candidate.")

        ranked_indices = torch.argsort(scores, dim=1, descending=True)
        ranked_relevant = relevant.gather(1, ranked_indices)

        num_queries, num_candidates = ranked_relevant.shape

        ranks = (
            torch.arange(
                1, num_candidates + 1, device=scores.device, dtype=torch.float32
            )
            .unsqueeze(0)
            .expand(num_queries, -1)
        )

        # Rank of the highest-ranked positive candidate.
        positive_ranks = torch.where(
            ranked_relevant, ranks, torch.full_like(ranks, float("inf"))
        )

        first_positive_rank = positive_ranks.min(dim=1).values

        # Reciprocal rank.
        reciprocal_rank = 1.0 / first_positive_rank

        # Average precision.
        cumulative_relevant = ranked_relevant.float().cumsum(dim=1)
        precision_at_rank = cumulative_relevant / ranks

        num_relevant = ranked_relevant.sum(dim=1).clamp_min(1)

        average_precision = (precision_at_rank * ranked_relevant.float()).sum(
            dim=1
        ) / num_relevant

        # Binary-relevance nDCG.
        discounts = 1.0 / torch.log2(ranks + 1.0)

        dcg = (ranked_relevant.float() * discounts).sum(dim=1)
        ideal_relevant = (ranks <= num_relevant.unsqueeze(1)).float()

        idcg = (ideal_relevant * discounts).sum(dim=1).clamp_min(1e-12)
        ndcg = dcg / idcg

        return {
            "mean_rank": first_positive_rank.mean(),
            "median_rank": torch.quantile(first_positive_rank.float(), 0.5),
            "MRR": reciprocal_rank.mean(),
            "mAP": average_precision.mean(),
            "nDCG": ndcg.mean(),
        }


if __name__ == "__main__":
    _ = Mamba3LitModule(None, None, None, None)
