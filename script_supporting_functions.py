# ==============================================================================
# LOGGING UTILITIES
# Functions for logging model complexity and GPU utilization metrics.
# ==============================================================================

import logging
import mlflow
import pynvml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from fvcore.nn import FlopCountAnalysis

_logger = logging.getLogger(__name__)


def log_dataset_sizes(training_dataloader, validation_dataloader, pl_logger=None):
    """Log the exact number of events in train and validation sets.

    Uses len(dataset) rather than len(dataloader)*batch_size, which would
    overcount the last partial batch.  Optionally logs as hyperparams so the
    counts appear in MLflow / TensorBoard alongside other run metadata.
    """
    n_train = len(training_dataloader.dataset)
    n_val   = len(validation_dataloader.dataset)
    n_total = n_train + n_val
    _logger.info(
        f"Dataset sizes — train: {n_train:,}  val: {n_val:,}  total: {n_total:,}"
    )
    if pl_logger is not None:
        pl_logger.log_hyperparams({
            "n_train_events": n_train,
            "n_val_events":   n_val,
            "n_total_events": n_total,
        })


def log_model_complexity(backbone, training_dataloader, pl_logger=None):
    """Compute and log model parameter counts and FLOPs per batch.

    Moves backbone to CPU for FLOPs analysis, then returns it to CUDA.
    Logs total_params, trainable_params, and flops_per_batch via pl_logger
    (supports both MLFlowLogger and TensorBoardLogger).
    """
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    _logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Log parameter counts eagerly — before FLOPs analysis — so they always land
    # in the logger even if FlopCountAnalysis fails.  Use mlflow.log_params()
    # directly for MLFlowLogger: PL's log_hyperparams abstraction can silently
    # swallow errors (same workaround as GPUUtilizationLogger.on_train_start).
    if pl_logger is not None:
        param_metrics = {
            "total_params":     total_params,
            "trainable_params": trainable_params,
        }
        if isinstance(pl_logger, MLFlowLogger):
            mlflow.log_params(param_metrics)
        else:
            pl_logger.log_hyperparams(param_metrics)

    # FLOPs analysis — wrapped in try/except because transformers with dynamic
    # shapes / attention masks often trigger unsupported-op errors in fvcore.
    total_flops = None
    try:
        with torch.no_grad():
            sample_batch_cpu = {k: v[:1].cpu() if isinstance(v, torch.Tensor) else v
                                for k, v in next(iter(training_dataloader)).items()}
            backbone.cpu().eval()
            flops = FlopCountAnalysis(backbone, sample_batch_cpu)
            flops.unsupported_ops_warnings(False)
            total_flops = flops.total()
            del flops, sample_batch_cpu
        backbone.to("cuda").train()
        _logger.info(f"FLOPs per batch: {total_flops:,}")
    except Exception as e:
        _logger.warning(f"FLOPs analysis failed ({type(e).__name__}: {e}); skipping flops_per_batch.")
        backbone.to("cuda").train()

    if pl_logger is not None and total_flops is not None:
        flops_metric = {"flops_per_batch": total_flops}
        if isinstance(pl_logger, MLFlowLogger):
            mlflow.log_params(flops_metric)
        else:
            pl_logger.log_hyperparams(flops_metric)


class GPUUtilizationLogger(pl.Callback):
    """Logs mean GPU compute utilization % and memory % to MLflow once per epoch."""

    def __init__(self, log_every_n_steps: int = 50, device_index: int = 0):
        self.log_every_n_steps = log_every_n_steps
        self.device_index = device_index
        self._handle = None
        self._util_samples: list = []
        self._mem_samples: list = []

    def on_train_start(self, trainer, pl_module):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

        # Log GPU specs as hyperparams so they appear alongside model/data metadata.
        n_gpus = pynvml.nvmlDeviceGetCount()
        gpu_name = pynvml.nvmlDeviceGetName(self._handle)
        # pynvml < 11.0 returns bytes; decode defensively
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        mem_total_gb = pynvml.nvmlDeviceGetMemoryInfo(self._handle).total / 1024**3
        _logger.info(
            f"GPU specs — device {self.device_index}: {gpu_name}, "
            f"{mem_total_gb:.1f} GB VRAM, {n_gpus} GPU(s) visible"
        )
        params = {
            "gpu_name":       gpu_name,
            "gpu_vram_gb":    str(round(mem_total_gb, 1)),
            "n_gpus_visible": str(n_gpus),
        }
        if isinstance(trainer.logger, MLFlowLogger):
            # Call mlflow directly — PL's log_hyperparams abstraction can
            # silently swallow errors or require params to be nested.
            mlflow.log_params(params)
        elif trainer.logger is not None:
            trainer.logger.log_hyperparams(params)

    def on_train_epoch_start(self, trainer, pl_module):
        self._util_samples.clear()
        self._mem_samples.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0 and self._handle is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            mem_used_pct = 100 * mem_info.used / mem_info.total
            self._util_samples.append(float(util))
            self._mem_samples.append(float(mem_used_pct))

    def on_train_epoch_end(self, trainer, pl_module):
        if self._util_samples:
            pl_module.log("gpu/utilization_pct", sum(self._util_samples) / len(self._util_samples), on_step=False, on_epoch=True)
            pl_module.log("gpu/memory_used_pct", sum(self._mem_samples) / len(self._mem_samples), on_step=False, on_epoch=True)

    def on_train_end(self, trainer, pl_module):
        if self._handle is not None:
            pynvml.nvmlShutdown()


# ==============================================================================
# BACKBONE FACTORY
# Instantiate the backbone model from the config dict.
# To add a new model: import it and add it to _BACKBONE_REGISTRY below.
# ==============================================================================

from .nuT_model_no_graphnet import nuT_vanilla

_BACKBONE_REGISTRY = {
    "nuT_vanilla": nuT_vanilla,
}


def build_backbone(config, features, updated_features, idx_dict):
    """Instantiate the backbone specified by ``config["backbone"]["name"]``.

    ``nuT`` / ``nuT_PROMETHEUS``  — uses ``updated_features`` (metadata stripped)
                                    and pairwise-mask parameters.
    ``nuT_vanilla``               — uses all ``features`` and ``no_blocks``.
    """
    bb = config["backbone"]
    name = bb.get("name", "nuT_PROMETHEUS")
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. Choose from {list(_BACKBONE_REGISTRY.keys())}."
        )

    common = dict(
        emb_dims=bb["emb_dims"],
        seq_length=config["node_definition"]["max_hits"],
        emb_type=bb["emb_type"],
        abs_position_encoding=bb["abs_position_encoding"],
        num_heads=bb["num_heads"],
        dropout_attn=bb["dropout_attn"],
        hidden_dim=bb["hidden_dim"],
        dropout_FFNN=bb["dropout_FFNN"],
        use_varlen=bb.get("use_varlen", False),
    )

    if name == "nuT_vanilla":
        return nuT_vanilla(n_features=len(features), no_blocks=bb["no_blocks"], **common)

    return _BACKBONE_REGISTRY[name](
        idx_dict=idx_dict,
        n_features=len(updated_features),
        mode=bb["mode"],
        masks=bb.get("masks"),
        refractive_index=bb.get("refractive_index"),
        pairwise_dims=bb["pairwise_dims"],
        no_hits_blocks=bb["no_hits_blocks"],
        no_evt_blocks=bb["no_evt_blocks"],
        **common,
    )
