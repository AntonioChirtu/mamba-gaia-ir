from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import multiprocessing

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"





from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature
import torch
import pytorch_lightning as pl

class GPUTemperatureGuard(pl.Callback):
    def __init__(self, max_temp=80, check_interval=10):
        """
        Args:
            max_temp: Max allowed GPU temp in °C (default 80)
            check_interval: Check every N training steps
        """
        self.max_temp = max_temp
        self.check_interval = check_interval
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            self.nvml_available = True
        except Exception:
            self.nvml_available = False
            print("⚠️ NVML not available - temperature monitoring disabled")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.nvml_available or batch_idx % self.check_interval != 0:
            return
        try:
            temp = nvmlDeviceGetTemperature(self.handle, 0)  # 0 = GPU sensor
            if temp > self.max_temp:
                raise RuntimeError(
                    f"GPU OVERHEATING: {temp}°C > {self.max_temp}°C - STOPPING TRAINING"
                )
        except Exception as e:
            print(f"⚠️ Temperature check failed: {e}")






rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

# Before trainer.fit()
torch.backends.cuda.matmul.allow_tf32 = True # Keep this for speed
torch.backends.cudnn.allow_tf32 = True

# Disable the specific optimization that breaks Mamba
import torch
torch.set_float32_matmul_precision('high')

from optuna.integration import PyTorchLightningPruningCallback

class SafePruningCallback(PyTorchLightningPruningCallback):
    def on_validation_end(self, trainer, pl_module):
        # Only report if a valid trial exists to avoid the NoneType error
        if self._trial is not None:
            super().on_validation_end(trainer, pl_module)
        else:
            # Optional: Log a warning so you know pruning is inactive for this run
            import logging
            logging.getLogger("pytorch_lightning").warning(
                "Optuna trial not found in callback. Pruning is disabled for this run."
            )


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # 1. Access the Hydra internal singleton (NOT the cfg object)
    from hydra.core.hydra_config import HydraConfig
    import optuna

    trial = None
    trial_number = None

    try:
        # This gets the global state for the current running job
        hc = HydraConfig.get()

        # In Multirun, the sweeper config is stored here
        # We use .get() to avoid 'not in struct' errors
        sweeper_conf = hc.get("sweeper")

        if sweeper_conf:
            storage = sweeper_conf.get("storage")
            study_name = sweeper_conf.get("study_name")
            trial_number = hc.job.num  # The current trial index (0, 1, 2...)

            if storage and study_name:
                # Connect to the SQLite DB
                study = optuna.load_study(study_name=study_name, storage=storage)

                # Filter trials to find the one matching our current job number
                trials = study.get_trials()
                trial = next((t for t in trials if t.number == trial_number), None)
    except Exception as e:
        log.warning(f"Could not connect to Optuna DB: {e}")

    # 2. Inject into the callback
    for i, cb in enumerate(callbacks):
        # We check for the base class or our safe wrapper
        if isinstance(cb, (PyTorchLightningPruningCallback, SafePruningCallback)):
            # If we found a trial, we update the callback
            # Note: Using getattr to safely get the monitor name
            monitor = getattr(cb, "monitor", getattr(cb, "_monitor", "val/I2T_R1"))

            callbacks[i] = SafePruningCallback(trial, monitor)

            if trial is not None:
                log.info(f"✅ SUCCESSFULLY connected to Optuna Trial #{trial_number} via SQLite!")
                # Print the hyperparameters for this specific trial
                log.info(f"Parameters for Trial #{trial_number}:")
                for key, value in trial.params.items():
                    log.info(f"    - {key}: {value}")
            else:
                log.warning("Optuna trial object not found. Pruning is disabled for this run.")
    
    callbacks.append(GPUTemperatureGuard(max_temp=85, check_interval=10))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, num_sanity_val_steps=0)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), weights_only=False)
    elif cfg.get("validate_only"):
        log.info("Running validation only!")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), weights_only=False)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        # trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
