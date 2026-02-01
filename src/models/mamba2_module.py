import copy
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import Metric, MeanMetric, MaxMetric


class RetrievalRecall(Metric):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits):
        # logits shape: [batch, batch] TODO
        batch_size = logits.shape[0]
        _, top_k_indices = logits.topk(self.k, dim=1)
        labels = torch.arange(batch_size, device=logits.device).view(-1, 1)

        self.correct += (top_k_indices == labels).sum()
        self.total += batch_size

    def compute(self):
        return self.correct / self.total


class Mamba2LitModule(LightningModule):
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
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            d_model: int = 64,
            patch_size: int = 16,
            vocab_size: int = 50257
    ) -> None:
        """Initialize a `Mamba2LitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model1 = copy.deepcopy(net)
        self.model2 = copy.deepcopy(net)

        # 1. Vision "Patch" Embedding: Turns [B, 3, 224, 224] -> [B, 196, d_model]
        self.patch_embed = torch.nn.Conv2d(
            3, d_model, kernel_size=patch_size, stride=patch_size
        )

        # 2. Text Embedding: Turns [B, 77] -> [B, 77, d_model]
        self.text_embed = torch.nn.Embedding(vocab_size, d_model)

        self.proj1 = torch.nn.Linear(d_model, 512)
        self.proj2 = torch.nn.Linear(d_model, 512)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # TODO: Make more complicated contrastive loss?
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # TODO: Add test recall, but for global set!
        # metric objects for calculating and averaging accuracy across batches
        self.train_recall = RetrievalRecall(k=1)
        self.val_r1 = RetrievalRecall(k=1)
        self.val_r5 = RetrievalRecall(k=5)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_r1_best = MaxMetric()

    def forward(self, x: torch.Tensor, modality="image") -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        if modality == "image":
            # x is [batch, num_patches, patch_dim]
            # [B, 3, 224, 224] -> [B, d_model, 14, 14] -> [B, d_model, 196]
            x = self.patch_embed(x).flatten(2)
            # -> [B, 196, d_model] (The 3D shape Mamba-2 expects!)
            x = x.transpose(1, 2)
            out = self.model1(x)
        else:
            # x is [batch, seq_len, word_dim]
            # [B, seq_len] -> [B, seq_len, d_model]
            x = self.text_embed(x)
            out = self.model2(x)

        # 2. Global Pooling: Turn sequence into a single vector
        # Mamba returns [batch, length, dim]. We average across length.
        out = out.mean(dim=1)

        # 3. Project to shared IR space
        if modality == "image":
            out = self.proj1(out)
        else:
            out = self.proj2(out)

        return out

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""

        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_r1.reset()
        self.val_r5.reset()
        self.val_r1_best.reset()

    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        images, texts = batch

        img_emb = self.forward(images, modality="image")
        txt_emb = self.forward(texts, modality="text")

        img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=-1)
        txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=-1)

        # Scaling by temperature
        # t = self.logit_scale.exp().clamp(max=100) # May not need
        logits_i2t = (img_emb @ txt_emb.t()) * self.logit_scale.exp()
        logits_t2i = logits_i2t.t()

        y = torch.arange(logits_i2t.shape[0], device=logits_i2t.device)
        loss = (self.criterion(logits_i2t, y) + self.criterion(logits_t2i, y)) / 2

        return loss, logits_i2t, logits_t2i, y

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, l_i2t, l_t2i, y = self.model_step(batch)

        self.train_recall.update(l_i2t)
        self.train_recall.update(l_t2i)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/R1", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, l_i2t, l_t2i, y = self.model_step(batch)

        self.val_r1.update(l_i2t)
        self.val_r1.update(l_t2i)
        self.val_r5.update(l_i2t)
        self.val_r5.update(l_t2i)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/R1", self.val_r1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/R5", self.val_r5, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        r1 = self.val_r1.compute()  # get current val acc
        self.val_r1_best(r1)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/R1_best", self.val_r1_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, l_i2t, l_t2i, y = self.model_step(batch)

        # TODO: also initialize in init
        # update and log metrics
        # self.test_r1.update(l_i2t)
        # self.test_r1.update(l_t2i)
        #
        # self.test_r5.update(l_i2t)
        # self.test_r5.update(l_t2i)
        #
        # self.test_r10.update(l_i2t)
        # self.test_r10.update(l_t2i)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        # self.log("test/R1", self.test_r1, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/R5", self.test_r5, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/R10", self.test_r10, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model1 = torch.compile(self.model1)
            self.model2 = torch.compile(self.model2)

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
                    "monitor": "val/R1",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Mamba2LitModule(None, None, None, None)
