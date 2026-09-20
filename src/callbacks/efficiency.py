from __future__ import annotations

import time
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor

_GIB = 1024**3


def _synchronize_cuda() -> None:
    """Wait for pending CUDA work so wall-clock timings are meaningful."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _extract_batch_size(batch: Any) -> int:
    """Extract batch size from this project's batch structure."""

    if isinstance(batch, Tensor):
        return int(batch.shape[0])

    if isinstance(batch, (tuple, list)) and batch:
        first = batch[0]
        if isinstance(first, Tensor):
            return int(first.shape[0])

    if isinstance(batch, dict):
        for key in ("images", "image", "inputs", "input_ids"):
            value = batch.get(key)
            if isinstance(value, Tensor):
                return int(value.shape[0])

    raise RuntimeError(
        "EfficiencyCallback could not infer batch size from batch type "
        f"{type(batch)!r}"
    )


class EfficiencyCallback(Callback):
    """Log workload-level timing, throughput, memory, and parameter metrics."""

    def __init__(self) -> None:
        super().__init__()

        self._train_started_at: float | None = None
        self._train_elapsed_seconds = 0.0
        self._train_local_samples = 0
        self._train_peak_memory_bytes = 0

        self._val_started_at: float | None = None
        self._val_local_samples = 0
        self._val_peak_memory_bytes = 0

        self._logged_batch_size = False

    @staticmethod
    def _distributed_sum(
        trainer: Trainer,
        value: Tensor,
    ) -> Tensor:
        return trainer.strategy.reduce(value, reduce_op="sum")

    @staticmethod
    def _distributed_max(
        trainer: Trainer,
        value: Tensor,
    ) -> Tensor:
        return trainer.strategy.reduce(value, reduce_op="max")

    @staticmethod
    def _current_peak_memory(pl_module: LightningModule) -> int:
        if not torch.cuda.is_available():
            return 0

        return int(torch.cuda.max_memory_allocated(pl_module.device))

    @staticmethod
    def _reset_peak_memory(pl_module: LightningModule) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def _pause_train_timer(self) -> None:
        """Pause training timing, for example while validation executes."""

        if self._train_started_at is None:
            return

        _synchronize_cuda()

        self._train_elapsed_seconds += (
            time.perf_counter() - self._train_started_at
        )
        self._train_started_at = None

    def _resume_train_timer(self) -> None:
        """Resume training timing after an interleaved validation loop."""

        if self._train_started_at is not None:
            return

        _synchronize_cuda()
        self._train_started_at = time.perf_counter()

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        total_parameters = sum(
            parameter.numel()
            for parameter in pl_module.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in pl_module.parameters()
            if parameter.requires_grad
        )
        non_trainable_parameters = (
            total_parameters - trainable_parameters
        )

        # on_fit_start is outside an epoch loop, so use the logger directly.
        if trainer.is_global_zero and trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "perf/total_parameters": total_parameters,
                    "perf/trainable_parameters": trainable_parameters,
                    "perf/non_trainable_parameters": non_trainable_parameters,
                },
                step=trainer.global_step,
            )

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._train_elapsed_seconds = 0.0
        self._train_local_samples = 0
        self._train_peak_memory_bytes = 0

        self._reset_peak_memory(pl_module)
        self._resume_train_timer()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._logged_batch_size:
            return

        local_batch_size = _extract_batch_size(batch)
        world_size = int(trainer.world_size)

        accumulate_grad_batches = trainer.accumulate_grad_batches
        if not isinstance(accumulate_grad_batches, int):
            # This can be scheduled by epoch. Trainer normally resolves it to
            # an int, but retain a safe fallback.
            accumulate_grad_batches = 1

        data_parallel_batch_size = local_batch_size * world_size
        effective_global_batch_size = (
            data_parallel_batch_size * accumulate_grad_batches
        )

        if trainer.is_global_zero and trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "perf/local_batch_size": local_batch_size,
                    "perf/data_parallel_batch_size": (
                        data_parallel_batch_size
                    ),
                    "perf/gradient_accumulation_steps": (
                        accumulate_grad_batches
                    ),
                    "perf/global_batch_size": (
                        effective_global_batch_size
                    ),
                },
                step=trainer.global_step,
            )

        self._logged_batch_size = True

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._train_local_samples += _extract_batch_size(batch)

        self._train_peak_memory_bytes = max(
            self._train_peak_memory_bytes,
            self._current_peak_memory(pl_module),
        )

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._pause_train_timer()

        self._train_peak_memory_bytes = max(
            self._train_peak_memory_bytes,
            self._current_peak_memory(pl_module),
        )

        local_samples = torch.tensor(
            float(self._train_local_samples),
            device=pl_module.device,
            dtype=torch.float64,
        )
        local_seconds = torch.tensor(
            self._train_elapsed_seconds,
            device=pl_module.device,
            dtype=torch.float64,
        )
        local_peak_memory = torch.tensor(
            float(self._train_peak_memory_bytes),
            device=pl_module.device,
            dtype=torch.float64,
        )

        # Sum samples because each rank processes different samples.
        global_samples = self._distributed_sum(
            trainer,
            local_samples,
        )

        # Use the slowest rank's duration and largest memory allocation.
        global_seconds = self._distributed_max(
            trainer,
            local_seconds,
        )
        global_peak_memory = self._distributed_max(
            trainer,
            local_peak_memory,
        )

        throughput = global_samples / global_seconds.clamp_min(1e-12)

        pl_module.log_dict(
            {
                "perf/train_epoch_seconds": global_seconds.float(),
                "perf/train_samples_per_second": throughput.float(),
                "perf/train_processed_samples": global_samples.float(),
                "perf/train_peak_gpu_memory_gb": (
                    global_peak_memory / _GIB
                ).float(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return

        # Exclude validation from train_epoch_seconds if validation executes
        # inside the training epoch loop.
        self._pause_train_timer()

        self._val_local_samples = 0
        self._val_peak_memory_bytes = 0

        self._reset_peak_memory(pl_module)

        _synchronize_cuda()
        self._val_started_at = time.perf_counter()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return

        self._val_local_samples += _extract_batch_size(batch)

        self._val_peak_memory_bytes = max(
            self._val_peak_memory_bytes,
            self._current_peak_memory(pl_module),
        )

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if trainer.sanity_checking or self._val_started_at is None:
            return

        _synchronize_cuda()
        local_elapsed = time.perf_counter() - self._val_started_at
        self._val_started_at = None

        self._val_peak_memory_bytes = max(
            self._val_peak_memory_bytes,
            self._current_peak_memory(pl_module),
        )

        local_samples = torch.tensor(
            float(self._val_local_samples),
            device=pl_module.device,
            dtype=torch.float64,
        )
        local_seconds = torch.tensor(
            local_elapsed,
            device=pl_module.device,
            dtype=torch.float64,
        )
        local_peak_memory = torch.tensor(
            float(self._val_peak_memory_bytes),
            device=pl_module.device,
            dtype=torch.float64,
        )

        global_samples = self._distributed_sum(
            trainer,
            local_samples,
        )
        global_seconds = self._distributed_max(
            trainer,
            local_seconds,
        )
        global_peak_memory = self._distributed_max(
            trainer,
            local_peak_memory,
        )

        throughput = global_samples / global_seconds.clamp_min(1e-12)

        pl_module.log_dict(
            {
                "perf/val_epoch_seconds": global_seconds.float(),
                "perf/val_samples_per_second": throughput.float(),
                "perf/val_processed_samples": global_samples.float(),
                "perf/val_peak_gpu_memory_gb": (
                    global_peak_memory / _GIB
                ).float(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )

        # If validation happened in the middle of an epoch, resume the train
        # timer and start a fresh CUDA peak-memory window.
        if trainer.training:
            self._reset_peak_memory(pl_module)
            self._resume_train_timer()