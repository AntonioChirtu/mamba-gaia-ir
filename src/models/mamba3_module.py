import copy
import gc
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import Metric, MeanMetric, MaxMetric
from torchmetrics.retrieval import RetrievalRecall
import wandb
import random


class RetrievalRecallWrapper:
    """
    Wrapper for official TorchMetrics RetrievalRecall that handles similarity matrices.
    """

    def __init__(self, k=1):
        self.k = k
        self.mean_metric = MeanMetric()

    def update(self, logits):
        """
        Update metric with similarity matrix.

        Args:
            logits: [batch_size, batch_size] similarity matrix
        """
        batch_size = logits.shape[0]
        target = torch.arange(batch_size, device=logits.device)

        # Get indices of top k matches
        _, top_k_indices = logits.topk(self.k, dim=1)

        # Check if target is in top_k (matches along dim 1)
        correct = (top_k_indices == target.view(-1, 1)).any(dim=1)

        # Ensure mean_metric is on same device as input
        if self.mean_metric.device != logits.device:
            self.mean_metric = self.mean_metric.to(logits.device)

        # Update your internal MeanMetric
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

        self.proj1 = torch.nn.Linear(image_net.d_model, 512)
        self.proj2 = torch.nn.Linear(text_net.d_model, 512)

        self.logit_scale = torch.nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / logit_scale_init))
        )

        self.val_outputs = {"img_embs": [], "txt_embs": [], "raw_texts": []}

        # TODO: Make more complicated contrastive loss?
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # TODO: Add test recall, but for global set!
        # metric objects for calculating and averaging accuracy across batches
        self.train_recall = RetrievalRecallWrapper(k=1)
        # Separate metrics for I2T and T2I
        self.train_i2t_r1 = RetrievalRecallWrapper(k=1)
        self.train_t2i_r1 = RetrievalRecallWrapper(k=1)
        self.val_i2t_r1 = RetrievalRecallWrapper(k=1)
        self.val_t2i_r1 = RetrievalRecallWrapper(k=1)
        self.val_i2t_r5 = RetrievalRecallWrapper(k=5)
        self.val_t2i_r5 = RetrievalRecallWrapper(k=5)

        self.val_batch_i2t_r1 = RetrievalRecallWrapper(k=1)
        self.val_batch_t2i_r1 = RetrievalRecallWrapper(k=1)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_i2t_r1_best = MaxMetric()
        self.val_t2i_r1_best = MaxMetric()
        self.val_mean_r1_best = MaxMetric()

        self.test_r1 = RetrievalRecallWrapper(k=1)
        self.test_r5 = RetrievalRecallWrapper(k=5)
        self.test_r10 = RetrievalRecallWrapper(k=10)

        # Initialize test outputs storage
        self.test_outputs = {"img_embs": [], "txt_embs": [], "raw_texts": []}

    def forward(
        self, x: torch.Tensor, modality="image", attention_mask=None
    ) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
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
            # x is [B, seq_len] token_ids -> [B, seq_len, d_model]
            x = self.text_embed(x)
            out = self.text_model(x)  # [B, L, d_model]

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
        self.val_i2t_r1.reset()
        self.val_t2i_r1.reset()
        self.val_i2t_r5.reset()
        self.val_t2i_r5.reset()
        self.val_i2t_r1_best.reset()
        self.val_t2i_r1_best.reset()
        self.val_mean_r1_best.reset()
        self.val_batch_i2t_r1.reset()
        self.val_batch_t2i_r1.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        images, texts, attention_mask = batch

        img_emb = self.forward(images, modality="image")
        txt_emb = self.forward(texts, modality="text", attention_mask=attention_mask)

        img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=-1)
        txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=-1)

        with torch.no_grad():
            # Clamping prevents the exponential matrix from blowing up to Infinity
            self.logit_scale.clamp_(max=4.6052)

        # Scaling by temperature
        scale = self.logit_scale.exp().clamp(max=100)
        logits_i2t = (img_emb @ txt_emb.t()) * scale
        logits_t2i = logits_i2t.t()

        y = torch.arange(logits_i2t.shape[0], device=logits_i2t.device)

        # Standard InfoNCE / CLIP Loss
        loss = (self.criterion(logits_i2t, y) + self.criterion(logits_t2i, y)) / 2

        # Defensive check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss detected! loss={loss.item()}")
            # Preserve device, dtype, and requires_grad from the original loss
            loss = scale * 0.0  # Uses a parameter → keeps grad_fn
            l_i2t = torch.tensor(0.0)
            l_t2i = torch.tensor(0.0)
            y = torch.tensor(0.0)

        return loss, logits_i2t, logits_t2i, y, img_emb, txt_emb

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        images, texts, attention_mask, _ = batch

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
        loss, l_i2t, l_t2i, y, _, _ = self.model_step((images, texts, attention_mask))

        # Catch the NaN guard signal
        if loss is None:
            return None

        # Update separate I2T and T2I metrics
        self.train_i2t_r1.update(l_i2t)
        self.train_t2i_r1.update(l_t2i)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/I2T_R1",
            self.train_i2t_r1.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/T2I_R1",
            self.train_t2i_r1.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.train_i2t_r1.reset()
        self.train_t2i_r1.reset()

        gc.collect()
        torch.cuda.empty_cache()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # 1. Use your model_step for the loss (keep it consistent!)
        images, texts, attention_mask, text_strings = batch

        self.image_model.vit.image_size = self.base_size

        loss, l_i2t, l_t2i, y, img_emb, txt_emb = self.model_step(
            (images, texts, attention_mask)
        )

        # If validation batch is broken, exit early to protect global metric tracking
        if loss is None:
            return

        self.val_batch_i2t_r1.update(l_i2t)
        self.val_batch_t2i_r1.update(l_t2i)

        if img_emb.ndim == 1:
            img_emb = img_emb.unsqueeze(0)
        if txt_emb.ndim == 1:
            txt_emb = txt_emb.unsqueeze(0)

        self.val_outputs["img_embs"].append(img_emb.detach().cpu())
        self.val_outputs["txt_embs"].append(txt_emb.detach().cpu())
        self.val_outputs["raw_texts"].extend(text_strings)

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
            [bool(self.val_outputs["img_embs"])],
            device=self.device,
            dtype=torch.bool,
        )
        has_outputs = self.all_gather(has_local).flatten()

        if not has_outputs.any():
            return
        if not has_outputs.all():
            raise RuntimeError("Some DDP ranks produced no validation embeddings")

        # Gather tensors
        local_img = torch.cat(self.val_outputs["img_embs"]).to(self.device)
        local_txt = torch.cat(self.val_outputs["txt_embs"]).to(self.device)

        local_n = torch.tensor([local_img.shape[0]], device=self.device)
        rank_sizes = self.all_gather(local_n).flatten()

        if not torch.all(rank_sizes == rank_sizes[0]):
            raise RuntimeError(f"Unequal samples across ranks: {rank_sizes.tolist()}")

        all_img = self.all_gather(local_img, sync_grads=False)
        all_txt = self.all_gather(local_txt, sync_grads=False)

        all_img = all_img.reshape(-1, local_img.shape[-1])
        all_txt = all_txt.reshape(-1, local_txt.shape[-1])

        if all_img.ndim != 2 or all_txt.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D embeddings, got "
                f"image={tuple(all_img.shape)}, text={tuple(all_txt.shape)}"
            )

        # Normalize and compute Global Similarity Matrix
        all_img = torch.nn.functional.normalize(all_img, p=2, dim=-1)
        all_txt = torch.nn.functional.normalize(all_txt, p=2, dim=-1)

        import torch.distributed as dist

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

        num_images, num_texts = sim_matrix.shape

        if num_images != num_texts:
            raise RuntimeError(
                f"Diagonal targets require a square matrix, got "
                f"{num_images} images and {num_texts} texts"
            )
        targets = torch.arange(num_images, device=sim_matrix.device)

        # 3. Calculate R@1, R@5, R@10 for both directions
        val_results = {}
        for k in [1, 5, 10, 20]:
            # --- Image to Text (Rows) ---
            _, top_k_i2t = sim_matrix.topk(k, dim=1)
            r_i2t = (top_k_i2t == targets.view(-1, 1)).any(dim=1).float().mean()
            val_results[f"val/I2T_R{k}"] = r_i2t

            # --- Text to Image (Columns) ---
            _, top_k_t2i = sim_matrix.t().topk(k, dim=1)
            r_t2i = (top_k_t2i == targets.view(-1, 1)).any(dim=1).float().mean()
            val_results[f"val/T2I_R{k}"] = r_t2i

        val_results["val/mean_R1"] = 0.5 * (
            val_results["val/I2T_R1"] + val_results["val/T2I_R1"]
        )

        # 4. Log all metrics to WandB/Progress Bar
        self.log_dict(
            val_results,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
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
            batch_i2t_r1,
            device=self.device,
            dtype=torch.float32,
        )
        batch_t2i_r1 = torch.as_tensor(
            batch_t2i_r1,
            device=self.device,
            dtype=torch.float32,
        )

        self.log_dict(
            {
                "val/batch_I2T_R1": batch_i2t_r1,
                "val/batch_T2I_R1": batch_t2i_r1,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        # 6. Save visual results table
        if self.trainer.is_global_zero:
            self._save_results(sim_matrix, all_strings, phase="val")
            self._save_misc_metrics(sim_matrix, all_strings, phase="val")

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
        self.val_outputs = {"img_embs": [], "txt_embs": [], "raw_texts": []}

        self.val_batch_i2t_r1.reset()
        self.val_batch_t2i_r1.reset()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        images, texts, attention_mask, text_strings = batch

        self.image_model.vit.image_size = self.base_size

        loss, l_i2t, l_t2i, y, img_emb, txt_emb = self.model_step(
            (images, texts, attention_mask)
        )

        # If validation batch is broken, exit early to protect global metric tracking
        if loss is None:
            return

        if img_emb.ndim == 1:
            img_emb = img_emb.unsqueeze(0)
        if txt_emb.ndim == 1:
            txt_emb = txt_emb.unsqueeze(0)

        self.test_outputs["img_embs"].append(img_emb.detach().cpu())
        self.test_outputs["txt_embs"].append(txt_emb.detach().cpu())
        self.test_outputs["raw_texts"].extend(text_strings)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        has_local = torch.tensor(
            [bool(self.test_outputs["img_embs"])],
            device=self.device,
            dtype=torch.bool,
        )
        has_outputs = self.all_gather(has_local).flatten()

        if not has_outputs.any():
            return
        if not has_outputs.all():
            raise RuntimeError("Some DDP ranks produced no test embeddings")

        local_img = torch.cat(self.test_outputs["img_embs"]).to(self.device)
        local_txt = torch.cat(self.test_outputs["txt_embs"]).to(self.device)

        if local_img.ndim != 2 or local_txt.ndim != 2:
            raise RuntimeError(
                f"Expected local [N, D] embeddings, got "
                f"image={tuple(local_img.shape)}, "
                f"text={tuple(local_txt.shape)}"
            )

        if local_img.shape[0] != local_txt.shape[0]:
            raise RuntimeError(
                f"Local image/text count mismatch: "
                f"{local_img.shape[0]} vs {local_txt.shape[0]}"
            )

        local_n = torch.tensor([local_img.shape[0]], device=self.device)
        rank_sizes = self.all_gather(local_n).flatten()

        if not torch.all(rank_sizes == rank_sizes[0]):
            raise RuntimeError(f"Unequal samples across ranks: {rank_sizes.tolist()}")

        all_img = self.all_gather(local_img, sync_grads=False)
        all_txt = self.all_gather(local_txt, sync_grads=False)

        all_img = all_img.reshape(-1, local_img.shape[-1])
        all_txt = all_txt.reshape(-1, local_txt.shape[-1])

        if all_img.ndim != 2 or all_txt.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D embeddings, got "
                f"image={tuple(all_img.shape)}, text={tuple(all_txt.shape)}"
            )

        # 2. Normalize and compute Global Similarity Matrix
        all_img = torch.nn.functional.normalize(all_img, p=2, dim=-1)
        all_txt = torch.nn.functional.normalize(all_txt, p=2, dim=-1)

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

        num_images, num_texts = sim_matrix.shape

        if len(all_strings) != num_images:
            raise RuntimeError(
                f"Embedding/string count mismatch: "
                f"{num_images} embeddings vs {len(all_strings)} strings"
            )

        if num_images != num_texts:
            raise RuntimeError(
                f"Diagonal targets require a square matrix, got "
                f"{num_images} images and {num_texts} texts"
            )
        targets = torch.arange(num_images, device=sim_matrix.device)

        # 3. Calculate R@1, R@5, R@10 for both directions
        test_results = {}
        for k in [1, 5, 10, 20]:
            # --- Image to Text (Rows) ---
            _, top_k_i2t = sim_matrix.topk(k, dim=1)
            r_i2t = (top_k_i2t == targets.view(-1, 1)).any(dim=1).float().mean()
            test_results[f"test/I2T_R{k}"] = r_i2t

            # --- Text to Image (Columns) ---
            _, top_k_t2i = sim_matrix.t().topk(k, dim=1)
            r_t2i = (top_k_t2i == targets.view(-1, 1)).any(dim=1).float().mean()
            test_results[f"test/T2I_R{k}"] = r_t2i

        test_results["test/mean_R1"] = 0.5 * (
            test_results["test/I2T_R1"] + test_results["test/T2I_R1"]
        )

        # 4. Log all metrics to WandB/Progress Bar
        self.log_dict(
            test_results,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # 6. Save visual results table
        if self.trainer.is_global_zero:
            self._save_results(sim_matrix, all_strings, phase="test")
            self._save_misc_metrics(sim_matrix, all_strings, phase="test")

        # 7. Reset storage for the next epoch
        self.test_outputs = {"img_embs": [], "txt_embs": [], "raw_texts": []}

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
                    "monitor": "val/mean_R1",  # Use training metric since validation runs every 10 epochs
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        return {"optimizer": optimizer}

    def _save_results(self, sim_matrix, all_texts, phase="val"):
        """Logs a table to WandB showing what the model predicted."""
        if isinstance(self.logger, WandbLogger):
            columns = [
                "Image_Index",
                "True_Caption",
                "Model_Top_Pick",
                "Confidence",
                "Correct",
            ]
            table = wandb.Table(columns=columns)

            # Look at the first 15 images to keep the WandB payload light
            num_samples_to_log = min(15, sim_matrix.shape[0])

            # Convert raw similarities to probabilities for readability
            probs = torch.softmax(sim_matrix[:num_samples_to_log].float(), dim=1)
            confidences, indices = probs.topk(1, dim=1)

            for i in range(num_samples_to_log):
                true_caption = all_texts[i]
                predicted_idx = indices[i].item()
                predicted_caption = all_texts[predicted_idx]
                conf = confidences[i].item()
                is_correct = predicted_idx == i

                table.add_data(i, true_caption, predicted_caption, conf, is_correct)

            # This will show up in WandB under the "val/predictions_sample" tab
            self.logger.experiment.log({f"{phase}/predictions_brief": table})

        
    @torch.no_grad()
    def _save_misc_metrics(self, sim_matrix, all_texts, phase="val"):
        if not isinstance(self.logger, WandbLogger):
            return

        num_candidates = sim_matrix.shape[1]
        num_samples = min(15, sim_matrix.shape[0])

        if len(all_texts) != num_candidates:
            raise RuntimeError(
                f"Expected {num_candidates} captions, got {len(all_texts)}"
            )

        query_sim = sim_matrix[:num_samples].float()

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

        # Diagonal is the correct caption under one-to-one pairing
        query_indices = torch.arange(
            num_samples,
            device=sim_matrix.device,
        )
        true_cosine = query_sim[query_indices, query_indices]

        # Rank of the true caption: 1 means correct top-1
        true_rank = (
            (query_sim > true_cosine[:, None]).sum(dim=1) + 1
        )

        # Temperature-scaled score, consistent with training logits
        scale = self.logit_scale.detach().exp().clamp(max=100)
        scaled_logits = scale * query_sim
        probabilities = torch.softmax(scaled_logits, dim=1)
        top1_softmax_score = probabilities.gather(
            1, top1_indices[:, None]
        ).squeeze(1)

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

            table.add_data(
                i,
                all_texts[i],
                all_texts[predicted_idx],
                true_rank[i].item(),
                top1_cosine[i].item(),
                true_cosine[i].item(),
                top2_cosine[i].item(),
                cosine_margin[i].item(),
                top1_softmax_score[i].item(),
                predicted_idx == i,
            )

        self.logger.experiment.log({
            f"{phase}/predictions_detailed": table
        })


if __name__ == "__main__":
    _ = Mamba3LitModule(None, None, None, None)
